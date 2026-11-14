"""URL routes for prediction endpoints."""

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import PredictView, PredictionHistoryViewSet

router = DefaultRouter()
router.register(r"history", PredictionHistoryViewSet, basename="prediction-history")

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict"),
    path("predictions/", include(router.urls)),
]
