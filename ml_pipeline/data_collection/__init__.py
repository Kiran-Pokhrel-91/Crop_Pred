"""Numbered Earth Engine extraction scripts, run directly in pipeline order.

01 generates the grid, 02-06 extract Sentinel-2, Sentinel-1, ERA5, terrain, and
land-cover features. Each script reads the grid contract and writes a per-source
table keyed by ``grid_id``.
"""
