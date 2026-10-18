# Real-Time Traffic Congestion Prediction and Geospatial Analytics Platform

Project for collecting live traffic flow data from the TomTom Traffic Flow API, storing historical observations, training congestion prediction models, exposing FastAPI endpoints, and visualizing geospatial analytics in Streamlit.

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

The storage layer uses CSV files through `TrafficDataRepository`. API, dashboard, and ML modules depend on that repository boundary, so the project can move to PostgreSQL later with minimal changes.

## Features

- TomTom Flow Segment Data API integration, with the key read from the environment. Readings with a missing `roadClosure` or a `confidence` outside 0-1 are rejected.
- Raw observations appended to a CSV file. Repeated readings are stored with an `is_repeat` flag, and training and evaluation skip them.
- Scheduler that collects data every `FETCH_INTERVAL_MINUTES` (default 10).
- Feature engineering for speed ratio, local-time fields (Asia/Kolkata), weekend and peak-hour flags, and congestion labels.
- Model comparison across Logistic Regression, Random Forest, XGBoost, and SVM, with a chronological train/test split.
- Persisted best model at `models/traffic_model.pkl`, with a SHA-256 sidecar and a metadata file that are checked before loading. The API reloads the model when the file or its sidecars change.
- FastAPI endpoints for health, latest traffic, history, and prediction.
- Streamlit dashboard with metrics, a Folium map of the latest reading per location, Plotly charts, and a prediction form.
- Docker and Docker Compose for the API, dashboard, and scheduler services.
- GitHub Actions CI (`.github/workflows/ci.yml`) that runs `ruff check .` and `pytest`.

## Tech Stack

- Backend: Python 3.12 or 3.13 (3.14 fails to build pydantic-core), FastAPI, Uvicorn
- Machine Learning: Pandas, NumPy, Scikit-Learn, XGBoost, Joblib
- Visualization: Folium, Plotly, Streamlit
- Data Storage: CSV now, repository pattern for PostgreSQL migration later
- Deployment: Docker, Docker Compose, GitHub Actions

## Installation

Run these commands from the cloned repository folder:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # runtime only
pip install -r requirements-dev.txt    # adds pytest, ruff and httpx for tests and linting
cp .env.example .env
```

Edit `.env` and set `TOMTOM_API_KEY`. Never commit `.env`; it is in `.gitignore`.

The Bangalore notebook (`BangaloreTraffic.ipynb`) does not read `.env`. Export `TOMTOM_API_KEY` in your shell before running it. It builds its history in `bangalore_traffic_data.csv`.

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

Open API docs at `http://localhost:8000/docs`.

Run the dashboard:

```bash
python -m streamlit run src/app.py
```

Open the dashboard at `http://localhost:8501`.

Note: the plain `streamlit run src/app.py` command fails with `ModuleNotFoundError: No module named 'src'` on the pinned Streamlit 1.41.1, because Streamlit only puts `src/` on the import path. Use `python -m streamlit run src/app.py` (the working directory is then on the path), or `PYTHONPATH=. streamlit run src/app.py`.

## API Documentation

All traffic endpoints return JSON records. Rows with missing or non-numeric coordinates are left out, and empty values are returned as `null`.

### `GET /health`

Returns service status and the current UTC timestamp.

### `GET /traffic/latest?limit=20`

Returns the newest observation for each location, newest first. `limit` (1-500) caps the number of locations returned.

### `GET /traffic/history?limit=100&since=2026-06-08T10:00:00+00:00`

Returns the newest observations, newest first. `limit` (1-5000, default 100) caps the number of rows. `since` is an optional ISO 8601 datetime; only rows at or after it are returned. A value without a timezone is read as UTC. An invalid `since` returns 422.

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
  "road_closure": false,
  "timestamp": "2026-06-08T08:00:00+00:00"
}
```

`timestamp` is optional and defaults to now. The `features` object in the response holds only the model input columns. Latitude must be between -90 and 90, longitude between -180 and 180.

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

Error responses:

| Status | Meaning |
|---|---|
| 422 | Invalid input: a bad `timestamp`, out-of-range latitude, or missing/zero speeds. The body names the problem. |
| 503 | The model file is missing. Train the model first. |
| 500 | Unexpected error, including a model whose checksum or metadata does not match. The body is `{"detail": "Internal server error"}`; details are in the server log only. |

## Docker Deployment

```bash
cp .env.example .env
docker compose up --build
```

Services:

- API: `http://127.0.0.1:8000`
- Dashboard: `http://127.0.0.1:8501`
- Scheduler: background collection service, the only service that reads `.env`

Ports are published on the loopback interface only, so the API and dashboard are not reachable from other machines. To expose them, change the `127.0.0.1:` prefix in `docker-compose.yml`, and first add authentication to the API.

The API and dashboard mount `data/`, `models/` and `outputs/` read-only. The scheduler writes to `data/`, and the container runs as user `app` (UID 1000). On Linux, make the host `data/` folder writable by UID 1000 (for example `sudo chown -R 1000:1000 data`). Only the scheduler needs `.env`; the API and dashboard use the default paths and need `MODEL_PATH` only if you changed it.

## Testing and Linting

```bash
pytest -q
ruff check .
```

CI runs both commands on every push and pull request.

## Dashboard Screenshots

Add screenshots to `outputs/reports/` after running the dashboard:

- `outputs/reports/dashboard-overview.png`
- `outputs/reports/map-view.png`
- `outputs/reports/prediction-form.png`

## Project Structure

```text
Real-Time-Traffic-Analysis/
├── .github/workflows/ci.yml
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
│   └── EDA.ipynb
├── outputs/
│   ├── maps/
│   ├── metrics/
│   └── reports/
├── src/
│   ├── app.py
│   ├── config.py
│   ├── data_fetcher.py
│   ├── data_storage.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── map_generator.py
│   ├── model_artifacts.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
├── tests/
├── .dockerignore
├── .env.example
├── .gitignore
├── BangaloreTraffic.ipynb
├── Dockerfile
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Future Improvements

- Replace CSV storage with PostgreSQL and SQLAlchemy migrations.
- Add a model registry and experiment tracking.
- Add authenticated API access.
- Add live dashboard refresh and background API-triggered collection.
- Add geohash-based spatial aggregation.
- Add drift monitoring and retraining automation.
