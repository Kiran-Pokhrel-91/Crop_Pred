"""API views for crop prediction and prediction history."""

from __future__ import annotations

import logging

from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import PredictionRequest
from .serializers import PredictionCreateSerializer, PredictionRequestSerializer
from .services import run_prediction

logger = logging.getLogger(__name__)


class PredictView(APIView):
    """Create a crop yield prediction from location, date, and optional crop."""

    def post(self, request):
        serializer = PredictionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        payload = serializer.validated_data
        payload["date"] = payload["date"].isoformat()

        record = PredictionRequest.objects.create(
            latitude=payload.get("latitude") if payload.get("latitude") is not None else 0.0,
            longitude=payload.get("longitude") if payload.get("longitude") is not None else 0.0,
            request_date=payload["date"],
            selected_crop=payload.get("crop", ""),
            request_payload=payload,
            status=PredictionRequest.STATUS_SUCCESS,
        )

        try:
            result = run_prediction(payload)
            recommended = result.get("recommended_crop") or {}
            location = result.get("location", {})
            resolved_lat = location.get("latitude", payload.get("latitude") or 0.0)
            resolved_lon = location.get("longitude", payload.get("longitude") or 0.0)
            record.latitude = float(resolved_lat)
            record.longitude = float(resolved_lon)
            record.grid_id = location.get("grid_id", "")
            record.district = location.get("district", "")
            record.province = location.get("province", "")
            record.recommended_crop = recommended.get("crop", "")
            record.predicted_yield_kg_ha = recommended.get("predicted_yield_kg_ha")
            record.confidence = recommended.get("confidence")
            health = recommended.get("health") or {}
            record.health_label = health.get("label", "")
            record.model_name = (result.get("model") or {}).get("name", "")
            record.response_payload = result
            record.save(
                update_fields=[
                    "latitude",
                    "longitude",
                    "grid_id",
                    "district",
                    "province",
                    "recommended_crop",
                    "predicted_yield_kg_ha",
                    "confidence",
                    "health_label",
                    "model_name",
                    "response_payload",
                ]
            )
            return Response(result, status=status.HTTP_200_OK)
        except ValueError as exc:
            logger.warning("Prediction request validation failed: %s", exc)
            record.status = PredictionRequest.STATUS_ERROR
            record.error_message = str(exc)
            record.response_payload = {"status": "error", "error": {"message": str(exc), "type": "validation_error"}}
            record.save(update_fields=["status", "error_message", "response_payload"])
            return Response(
                {"status": "error", "error": {"message": str(exc), "type": "validation_error"}},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            logger.exception("Prediction failed")
            record.status = PredictionRequest.STATUS_ERROR
            record.error_message = str(exc)
            record.response_payload = {"status": "error", "error": {"message": "Prediction failed.", "type": "server_error"}}
            record.save(update_fields=["status", "error_message", "response_payload"])
            return Response(
                {
                    "status": "error",
                    "error": {
                        "message": "Prediction failed. Check server logs for details.",
                        "type": "server_error",
                        "detail": str(exc),
                    },
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class PredictionHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only prediction history."""

    queryset = PredictionRequest.objects.all()
    serializer_class = PredictionRequestSerializer

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        serializer = self.get_serializer(page if page is not None else queryset, many=True)
        data = serializer.data
        if page is not None:
            paginated = self.get_paginated_response(data)
            paginated.data = {"status": "success", **paginated.data}
            return paginated
        return Response({"status": "success", "count": len(data), "results": data})

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response({"status": "success", "result": serializer.data})

    @action(detail=False, methods=["get"], url_path="latest")
    def latest(self, request):
        instance = self.get_queryset().first()
        if not instance:
            return Response({"status": "success", "result": None})
        serializer = self.get_serializer(instance)
        return Response({"status": "success", "result": serializer.data})
