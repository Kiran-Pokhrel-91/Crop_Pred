"""URL routes for satellite/grid endpoints."""

from django.urls import path

from .views import DistrictDashboardView, DistrictListView, GridDetailView, GridFeaturesView, GridListView, GridNearestView

urlpatterns = [
    path("satellite/grid/", GridListView.as_view(), name="grid-list"),
    path("satellite/grid/nearest/", GridNearestView.as_view(), name="grid-nearest"),
    path("satellite/grid/<str:grid_id>/", GridDetailView.as_view(), name="grid-detail"),
    path("satellite/features/", GridFeaturesView.as_view(), name="grid-features"),
    path("satellite/districts/", DistrictListView.as_view(), name="district-list"),
    path("satellite/district-dashboard/", DistrictDashboardView.as_view(), name="district-dashboard"),
]
