"""Streamlit dashboard for traffic analytics."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from src.config import get_settings
from src.data_storage import TrafficDataRepository
from src.feature_engineering import add_traffic_features
from src.map_generator import create_congestion_map
from src.predict import TrafficPredictionInput, TrafficPredictor

st.set_page_config(page_title="Real-Time Traffic Analytics", layout="wide")


@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    """Load and feature-engineer traffic data for dashboard rendering."""

    settings = get_settings()
    repository = TrafficDataRepository(settings.raw_data_path, settings.processed_data_path)
    return add_traffic_features(repository.read_raw())


def render_metrics(dataframe: pd.DataFrame) -> None:
    """Render top-line traffic metrics."""

    average_speed = dataframe["current_speed"].mean() if not dataframe.empty else 0
    congested = int((dataframe["congestion_level"] == "High").sum()) if not dataframe.empty else 0
    free_flow = int((dataframe["congestion_level"] == "Low").sum()) if not dataframe.empty else 0
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Speed", f"{average_speed:.1f} km/h")
    col2.metric("Congested Locations", congested)
    col3.metric("Free Flow Locations", free_flow)


def render_analytics(dataframe: pd.DataFrame) -> None:
    """Render Plotly analytics charts."""

    if dataframe.empty:
        st.info("No traffic records available yet. Run the scheduler or fetch data first.")
        return

    trends = dataframe.sort_values("timestamp")
    st.plotly_chart(
        px.line(
            trends,
            x="timestamp",
            y="current_speed",
            color="location_name",
            title="Traffic Trends",
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        px.histogram(
            dataframe,
            x="congestion_level",
            color="congestion_level",
            title="Congestion Distribution",
        ),
        use_container_width=True,
    )
    hourly = dataframe.groupby("hour", as_index=False)["current_speed"].mean()
    col2.plotly_chart(
        px.bar(hourly, x="hour", y="current_speed", title="Hourly Average Speed"),
        use_container_width=True,
    )


def render_prediction_form() -> None:
    """Render model prediction inputs."""

    settings = get_settings()
    st.subheader("Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        location_name = col1.text_input("Location Name", "New Delhi")
        latitude = col1.number_input("Latitude", value=28.6139, format="%.6f")
        longitude = col1.number_input("Longitude", value=77.2090, format="%.6f")
        current_speed = col1.number_input("Current Speed", min_value=0.0, value=25.0)
        free_flow_speed = col2.number_input("Free Flow Speed", min_value=1.0, value=55.0)
        confidence = col2.slider("Confidence", min_value=0.0, max_value=1.0, value=0.9)
        travel_time = col2.number_input("Travel Time", min_value=0.0, value=180.0)
        road_closure = col2.checkbox("Road Closure")
        submitted = st.form_submit_button("Predict Congestion")

    if submitted:
        try:
            predictor = TrafficPredictor(settings.model_path)
            result = predictor.predict(
                TrafficPredictionInput(
                    location_name=location_name,
                    latitude=latitude,
                    longitude=longitude,
                    current_speed=current_speed,
                    free_flow_speed=free_flow_speed,
                    confidence=confidence,
                    travel_time=travel_time,
                    road_closure=road_closure,
                    timestamp=datetime.now(UTC).isoformat(),
                )
            )
            st.success(f"Predicted congestion level: {result['prediction']}")
            if result["probabilities"]:
                st.json(result["probabilities"])
        except FileNotFoundError:
            st.error("Model file not found. Train the model before using predictions.")


def main() -> None:
    """Run the Streamlit dashboard."""

    st.title("Real-Time Traffic Congestion Prediction and Geospatial Analytics")
    dataframe = load_data()
    render_metrics(dataframe)

    st.subheader("Interactive Map")
    traffic_map = create_congestion_map(dataframe)
    st_folium(traffic_map, width=None, height=520)

    render_analytics(dataframe)
    render_prediction_form()


if __name__ == "__main__":
    main()
