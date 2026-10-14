"""Tests for the congestion map builder."""

import pandas as pd
from folium import CircleMarker
from folium.plugins import MarkerCluster

from src.map_generator import INDIA_CENTER, create_congestion_map, latest_per_location


def observation(timestamp, location="Pune", speed=30.0, level="Medium", latitude=18.5):
    return {
        "timestamp": pd.Timestamp(timestamp, tz="UTC"),
        "location_name": location,
        "latitude": latitude,
        "longitude": 73.8,
        "current_speed": speed,
        "free_flow_speed": 60.0,
        "congestion_level": level,
    }


def markers(traffic_map):
    clusters = [c for c in traffic_map._children.values() if isinstance(c, MarkerCluster)]
    if not clusters:
        return []
    return [child for child in clusters[0]._children.values() if isinstance(child, CircleMarker)]


def test_latest_per_location_keeps_newest_row():
    frame = pd.DataFrame(
        [
            observation("2026-06-08T09:00:00", speed=45.0),
            observation("2026-06-08T08:00:00", speed=10.0),
            observation("2026-06-08T08:00:00", location="Delhi", speed=20.0),
        ]
    )

    latest = latest_per_location(frame)

    assert len(latest) == 2
    assert latest.loc[latest["location_name"] == "Pune", "current_speed"].item() == 45.0


def test_map_plots_one_marker_per_location_and_fits_bounds():
    frame = pd.DataFrame(
        [
            observation("2026-06-08T08:00:00", speed=10.0),
            observation("2026-06-08T09:00:00", speed=45.0),
            observation("2026-06-08T08:00:00", location="Mumbai", latitude=19.07),
        ]
    )

    traffic_map = create_congestion_map(frame)
    html = traffic_map.get_root().render()

    assert len(markers(traffic_map)) == 2
    assert "Speed: 45.0 km/h" in html
    assert "Speed: 10.0 km/h" not in html
    assert "fitBounds" in html


def test_popup_values_are_escaped():
    frame = pd.DataFrame([observation("2026-06-08T08:00:00", location="<script>alert(1)</script>")])

    html = create_congestion_map(frame).get_root().render()

    assert "<script>alert(1)" not in html
    assert "&lt;script&gt;" in html


def test_unknown_congestion_level_is_neutral_grey():
    frame = pd.DataFrame([observation("2026-06-08T08:00:00", level=None)])

    traffic_map = create_congestion_map(frame)

    assert markers(traffic_map)[0].options["color"] == "gray"


def test_known_levels_use_their_colours():
    frame = pd.DataFrame([observation("2026-06-08T08:00:00", level="High")])

    assert markers(create_congestion_map(frame))[0].options["color"] == "red"


def test_empty_frame_uses_default_view_without_markers():
    traffic_map = create_congestion_map(pd.DataFrame(columns=["timestamp", "location_name"]))
    html = traffic_map.get_root().render()

    assert markers(traffic_map) == []
    assert traffic_map.location == INDIA_CENTER
    assert "fitBounds" not in html
