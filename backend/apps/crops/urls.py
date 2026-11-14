"""URL routes for crop endpoints."""

from django.urls import path

from .views import CropListView, CropSeasonView, ModelInfoView

urlpatterns = [
    path("crops/", CropListView.as_view(), name="crop-list"),
    path("crops/seasons/", CropSeasonView.as_view(), name="crop-seasons"),
    path("model/", ModelInfoView.as_view(), name="model-info"),
]
