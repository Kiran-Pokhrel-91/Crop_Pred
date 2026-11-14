"""Database models for prediction requests and history."""

from django.db import models


class PredictionRequest(models.Model):
    """Persisted prediction request and response summary."""

    STATUS_SUCCESS = "success"
    STATUS_ERROR = "error"
    STATUS_CHOICES = [
        (STATUS_SUCCESS, "Success"),
        (STATUS_ERROR, "Error"),
    ]

    created_at = models.DateTimeField(auto_now_add=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    request_date = models.DateField()
    grid_id = models.CharField(max_length=32, blank=True)
    district = models.CharField(max_length=128, blank=True)
    province = models.CharField(max_length=128, blank=True)
    selected_crop = models.CharField(max_length=64, blank=True)
    recommended_crop = models.CharField(max_length=64, blank=True)
    predicted_yield_kg_ha = models.FloatField(null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    health_label = models.CharField(max_length=32, blank=True)
    model_name = models.CharField(max_length=64, blank=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_SUCCESS)
    error_message = models.TextField(blank=True)
    request_payload = models.JSONField(default=dict, blank=True)
    response_payload = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.grid_id or 'unknown'} {self.request_date} ({self.status})"
