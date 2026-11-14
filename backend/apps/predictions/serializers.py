"""Serializers for prediction API requests and responses."""

from rest_framework import serializers

from .models import PredictionRequest


class PredictionCreateSerializer(serializers.Serializer):
    latitude = serializers.FloatField(required=False)
    longitude = serializers.FloatField(required=False)
    district = serializers.CharField(required=False, allow_blank=False)
    date = serializers.DateField()
    crop = serializers.CharField(required=False, allow_blank=False)
    crops = serializers.ListField(
        child=serializers.CharField(allow_blank=False),
        required=False,
        allow_empty=False,
    )
    use_realtime = serializers.BooleanField(required=False, default=True)

    def validate(self, attrs):
        if attrs.get("crop") and attrs.get("crops"):
            raise serializers.ValidationError("Provide either 'crop' or 'crops', not both.")
        has_coords = attrs.get("latitude") is not None and attrs.get("longitude") is not None
        has_district = bool(attrs.get("district"))
        if not has_coords and not has_district:
            raise serializers.ValidationError("Provide a district or both latitude and longitude.")
        if (attrs.get("latitude") is None) != (attrs.get("longitude") is None):
            raise serializers.ValidationError("Provide both latitude and longitude together.")
        return attrs


class PredictionRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionRequest
        fields = [
            "id",
            "created_at",
            "status",
            "latitude",
            "longitude",
            "request_date",
            "grid_id",
            "district",
            "province",
            "selected_crop",
            "recommended_crop",
            "predicted_yield_kg_ha",
            "confidence",
            "health_label",
            "model_name",
            "error_message",
            "request_payload",
            "response_payload",
        ]
