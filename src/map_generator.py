"""Folium map generation utilities."""

from __future__ import annotations

import html
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster

CONGESTION_COLORS = {"Low": "green", "Medium": "orange", "High": "red", "Unknown": "gray"}
INDIA_CENTER = [20.5937, 78.9629]


def latest_per_location(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep only the newest observation for each location."""

    if dataframe.empty:
        return dataframe
    return (
        dataframe.sort_values("timestamp")
        .groupby("location_name", sort=False, dropna=False)
        .tail(1)
    )


def create_congestion_map(dataframe: pd.DataFrame, output_path: Path | None = None) -> folium.Map:
    """Create a marker-clustered map with the latest observation per location."""

    latest = latest_per_location(dataframe)
    if not latest.empty:
        latest = latest.dropna(subset=["latitude", "longitude"])
    traffic_map = folium.Map(location=INDIA_CENTER, zoom_start=5)

    if not latest.empty:
        marker_cluster = MarkerCluster().add_to(traffic_map)
        for _, row in latest.iterrows():
            raw_level = str(row.get("congestion_level"))
            level = raw_level if raw_level in CONGESTION_COLORS else "Unknown"
            color = CONGESTION_COLORS[level]
            popup = (
                f"<b>{html.escape(str(row.get('location_name', 'Unknown')))}</b><br>"
                f"Congestion: {html.escape(level)}<br>"
                f"Speed: {html.escape(str(row.get('current_speed', 0)))} km/h<br>"
                f"Free flow: {html.escape(str(row.get('free_flow_speed', 0)))} km/h"
            )
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=8,
                popup=popup,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
            ).add_to(marker_cluster)

        lat_min, lon_min = latest[["latitude", "longitude"]].min().astype(float)
        lat_max, lon_max = latest[["latitude", "longitude"]].max().astype(float)
        traffic_map.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]], max_zoom=12)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traffic_map.save(str(output_path))
    return traffic_map
