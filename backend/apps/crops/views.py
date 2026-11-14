"""API views for crop catalogue and metadata."""

from rest_framework.response import Response
from rest_framework.views import APIView

from .services import get_crop_catalogue, get_crop_seasons, get_model_info


class CropListView(APIView):
    def get(self, request):
        return Response({"status": "success", "count": 9, "results": get_crop_catalogue()})


class CropSeasonView(APIView):
    def get(self, request):
        crop = request.query_params.get("crop")
        seasons = get_crop_seasons(crop)
        return Response({"status": "success", "count": len(seasons), "results": seasons})


class ModelInfoView(APIView):
    def get(self, request):
        return Response({"status": "success", "model": get_model_info()})
