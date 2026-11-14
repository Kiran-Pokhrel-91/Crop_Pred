"""Root URL configuration for the Nepal Crop Prediction API."""

from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def health_check(request):
    return JsonResponse({"status": "ok", "service": "nepal-crop-prediction-api"})


def api_root(request):
    return JsonResponse(
        {
            "status": "success",
            "service": "Nepal Crop Prediction API",
            "endpoints": {
                "health": "/api/health/",
                "predict": "/api/predict/",
                "predictions": "/api/predictions/history/",
                "prediction_latest": "/api/predictions/history/latest/",
                "crops": "/api/crops/",
                "crop_seasons": "/api/crops/seasons/",
                "model": "/api/model/",
                "grid": "/api/satellite/grid/",
                "grid_nearest": "/api/satellite/grid/nearest/?latitude=..&longitude=..",
                "grid_detail": "/api/satellite/grid/<grid_id>/",
                "grid_features": "/api/satellite/features/?grid_id=..",
                "districts": "/api/satellite/districts/",
                "district_dashboard": "/api/satellite/district-dashboard/?district=Kathmandu",
            },
        }
    )


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", api_root, name="api-root"),
    path("api/health/", health_check, name="health-check"),
    path("api/", include("apps.predictions.urls")),
    path("api/", include("apps.crops.urls")),
    path("api/", include("apps.satellite.urls")),
]
