"""Folium map generation utilities."""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
from folium.plugins import MarkerCluster

CONGESTION_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}


def create_congestion_map(dataframe: pd.DataFrame, output_path: Path | None = None) -> folium.Map:
    """Create a marker-clustered congestion map."""

    if dataframe.empty:
        traffic_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    else:
        center = [float(dataframe["latitude"].mean()), float(dataframe["longitude"].mean())]
        traffic_map = folium.Map(location=center, zoom_start=11)
        marker_cluster = MarkerCluster().add_to(traffic_map)

        for _, row in dataframe.iterrows():
            level = row.get("congestion_level", "Low")
            color = CONGESTION_COLORS.get(str(level), "blue")
            popup = (
                f"<b>{row.get('location_name', 'Unknown')}</b><br>"
                f"Congestion: {level}<br>"
                f"Speed: {row.get('current_speed', 0)} km/h<br>"
                f"Free flow: {row.get('free_flow_speed', 0)} km/h"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                popup=popup,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
            ).add_to(marker_cluster)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        traffic_map.save(str(output_path))
    return traffic_map
