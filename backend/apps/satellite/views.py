"""API views for grid and satellite feature data."""

from rest_framework.response import Response
from rest_framework.views import APIView

from .services import find_nearest_cell, get_cell_features, get_district_dashboard, get_grid_cell, list_districts, load_grid


class GridListView(APIView):
    def get(self, request):
        district = request.query_params.get("district")
        province = request.query_params.get("province")
        limit = min(int(request.query_params.get("limit", 100)), 1000)
        offset = max(int(request.query_params.get("offset", 0)), 0)
        grid = load_grid()
        if district:
            grid = grid[grid["district"].str.lower() == district.lower()]
        if province:
            grid = grid[grid["province"].str.lower() == province.lower()]
        total = len(grid)
        page = grid.iloc[offset: offset + limit]
        return Response(
            {
                "status": "success",
                "count": total,
                "limit": limit,
                "offset": offset,
                "results": page.to_dict("records"),
            }
        )


class GridDetailView(APIView):
    def get(self, request, grid_id: str):
        cell = get_grid_cell(grid_id)
        if not cell:
            return Response({"status": "error", "error": {"message": "Grid cell not found"}}, status=404)
        return Response({"status": "success", "result": cell})


class GridNearestView(APIView):
    def get(self, request):
        try:
            latitude = float(request.query_params.get("latitude"))
            longitude = float(request.query_params.get("longitude"))
        except (TypeError, ValueError):
            return Response(
                {"status": "error", "error": {"message": "latitude and longitude query params are required"}},
                status=400,
            )
        cell = find_nearest_cell(latitude, longitude)
        if not cell:
            return Response({"status": "error", "error": {"message": "Grid not available"}}, status=404)
        return Response({"status": "success", "result": cell})


class DistrictListView(APIView):
    def get(self, request):
        districts = list_districts()
        return Response({"status": "success", "count": len(districts), "results": districts})


class DistrictDashboardView(APIView):
    def get(self, request):
        district = request.query_params.get("district")
        if not district:
            return Response({"status": "error", "error": {"message": "district query parameter is required"}}, status=400)
        data = get_district_dashboard(district)
        if not data:
            return Response({"status": "error", "error": {"message": "District not found"}}, status=404)
        return Response({"status": "success", "dashboard": data})


class GridFeaturesView(APIView):
    def get(self, request):
        grid_id = request.query_params.get("grid_id")
        if not grid_id:
            return Response({"status": "error", "error": {"message": "grid_id is required"}}, status=400)
        year = request.query_params.get("year")
        month = request.query_params.get("month")
        year_i = int(year) if year is not None else None
        month_i = int(month) if month is not None else None
        features = get_cell_features(grid_id, year_i, month_i)
        return Response({"status": "success", "grid_id": grid_id, "features": features})
