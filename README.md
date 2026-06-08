# Real-Time Traffic Congestion Prediction and Geospatial Analytics Platform

Production-oriented ML engineering project for collecting live traffic flow data from the TomTom Traffic Flow API, storing historical observations, training congestion prediction models, exposing FastAPI endpoints, and visualizing geospatial analytics in Streamlit.

## Architecture

```text
TomTom Traffic Flow API
        |
        v
Scheduler / main.py
        |
        v
CSV Repository  --->  Feature Engineering  --->  Model Training / Evaluation
        |                         |                         |
        v                         v                         v
FastAPI Service             Streamlit Dashboard        models/traffic_model.pkl
```

The current storage layer uses CSV files through `TrafficDataRepository`. API, dashboard, and ML modules depend on that repository boundary so the project can migrate to PostgreSQL later with minimal changes.

## Features

- TomTom Flow Segment Data API integration with environment-based credentials.
- Incremental CSV storage for raw and processed traffic data.
- Scheduler that collects live data every 10 minutes.
- Feature engineering for speed ratio, time fields, weekend and peak-hour flags, and congestion labels.
- Model comparison across Logistic Regression, Random Forest, XGBoost, and SVM.
- Persisted best model at `models/traffic_model.pkl`.
- FastAPI endpoints for health, latest traffic, history, and prediction.
- Streamlit dashboard with live metrics, Folium marker clustering, Plotly analytics, and prediction form.
- Docker and Docker Compose for API, dashboard, and scheduler services.
- GitHub Actions CI for dependency installation, linting, and tests.

## Tech Stack

- Backend: Python 3.11, FastAPI, Uvicorn
- Machine Learning: Pandas, NumPy, Scikit-Learn, XGBoost, Joblib
- Visualization: Folium, Plotly, Streamlit
- Data Storage: CSV now, repository pattern for PostgreSQL migration later
- Deployment: Docker, Docker Compose, GitHub Actions

## Installation

```bash
cd ~/Desktop/Real-Time-Traffic-Analysis
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `TOMTOM_API_KEY`.

`LOCATIONS` format:

```text
Location Name:latitude:longitude,Another Location:latitude:longitude
```

## Usage

Collect one batch through the scheduler entry point:

```bash
python main.py
```

Train a model after enough historical records are collected:

```bash
python -m src.train
```

Run the FastAPI service:

```bash
uvicorn api.main:app --reload
```

Open API docs:

```text
http://localhost:8000/docs
```

Run the dashboard:

```bash
streamlit run src/app.py
```

Open dashboard:

```text
http://localhost:8501
```

## API Documentation

### `GET /health`

Returns service status and current UTC timestamp.

### `GET /traffic/latest?limit=20`

Returns the latest feature-enriched traffic observations.

### `GET /traffic/history?limit=100`

Returns historical feature-enriched traffic observations.

### `POST /predict`

Request body:

```json
{
  "location_name": "New Delhi",
  "latitude": 28.6139,
  "longitude": 77.209,
  "current_speed": 25,
  "free_flow_speed": 55,
  "confidence": 0.9,
  "travel_time": 180,
  "road_closure": false
}
```

Response:

```json
{
  "prediction": "Medium",
  "probabilities": {
    "High": 0.12,
    "Low": 0.18,
    "Medium": 0.70
  },
  "features": {}
}
```

## Docker Deployment

```bash
cp .env.example .env
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Dashboard: `http://localhost:8501`
- Scheduler: background collection service

## Testing

```bash
pytest -q
ruff check .
```

## Dashboard Screenshots

Add screenshots to `outputs/reports/` after running the dashboard:

- `outputs/reports/dashboard-overview.png`
- `outputs/reports/map-view.png`
- `outputs/reports/prediction-form.png`

## Project Structure

```text
Real-Time-Traffic-Analysis/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── maps/
│   ├── reports/
│   └── metrics/
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── data_fetcher.py
│   ├── data_storage.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── map_generator.py
│   └── app.py
├── api/
│   └── main.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── main.py
```

## Future Improvements

- Replace CSV storage with PostgreSQL and SQLAlchemy migrations.
- Add a model registry and experiment tracking.
- Add authenticated API access.
- Add live dashboard refresh and background API-triggered collection.
- Add geohash-based spatial aggregation.
- Add drift monitoring and retraining automation.
