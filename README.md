# Nepal Crop Prediction Platform

An end-to-end, reproducible geospatial machine-learning platform for predicting
crop yield across Nepal. Earth observation and climate data are transformed into
a national grid dataset, models are trained offline, and predictions are served
through a Django REST API with a React dashboard.

> **Status:** Phases 1–5 of the data pipeline are implemented: the national 5 km
> grid, Sentinel-2 optical features, Sentinel-1 SAR features, ERA5-Land climate
> features, and the historical ground-truth merge. Later modelling and serving
> stages consume these outputs.

## Architecture

```text
                    ┌──────────────────────────────────────────┐
                    │              Data sources                │
                    │ Sentinel-2 · Sentinel-1 · ERA5 · DEM     │
                    │ ESA WorldCover · administrative boundary │
                    └────────────────────┬─────────────────────┘
                                         │ Google Earth Engine
                    ┌────────────────────▼─────────────────────┐
                    │            ml_pipeline/                  │
                    │ grid → extract → merge → features        │
                    │ train → evaluate → persist artifacts     │
                    └────────────┬─────────────────┬───────────┘
                                 │                 │
                    data/final/  │                 │ models/
                    (GeoParquet) │                 │ (joblib/JSON)
                                 ▼                 ▼
                    ┌──────────────────────────────────────────┐
                    │        backend/ (Django + DRF)           │
                    │ satellite · crops · predictions          │
                    └────────────────────┬─────────────────────┘
                                         │ REST /api/*
                    ┌────────────────────▼─────────────────────┐
                    │      frontend/ (React + Vite + Tailwind) │
                    │ dashboard · prediction · map             │
                    └──────────────────────────────────────────┘
```

### Design boundaries

- The ML pipeline never imports Django and can run independently of the web app.
- `ml_pipeline/prediction/predictor.py` is a standalone inference class that
  loads persisted artifacts and returns plain Python objects.
- Django views stay thin: validation, service/predictor call, serialisation.
- Configuration is centralised in `ml_pipeline/config.py` and
  `backend/config/settings.py`; secrets come from environment variables.

## Repository layout

```text
.
├── AGENT.md                    # Architecture rules and working agreements
├── .env.example                # Environment template (copy to .env)
├── data/
│   ├── raw/                    # Downloaded source data (ignored)
│   ├── processed/              # Intermediate per-source tables (ignored)
│   └── final/                  # Model-ready datasets (ignored)
├── models/                     # Trained artifacts (ignored)
├── ml_pipeline/
│   ├── config.py               # Env, paths, GEE init, AOI/CRS constants
│   ├── requirements.txt        # Geospatial + ML dependencies
│   ├── utils/                  # Earth Engine and geospatial helpers
│   ├── data_collection/        # Numbered extraction scripts (01-06)
│   ├── processing/             # Merge (07) and feature engineering
│   ├── training/               # Training and evaluation
│   └── prediction/             # Standalone predictor
├── backend/
│   ├── pyproject.toml          # uv project and Django/DRF dependencies
│   ├── uv.lock                 # Locked backend dependency versions
│   ├── .python-version         # Backend Python version
│   ├── manage.py
│   ├── config/                 # Settings, URLs, WSGI
│   └── apps/
│       ├── satellite/          # Grid and satellite feature endpoints
│       ├── crops/              # Crop catalogue and seasonal metadata
│       └── predictions/        # Prediction endpoint and history
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── api/client.js       # Axios client using VITE_API_BASE_URL
        ├── components/         # Reusable UI
        └── pages/              # Dashboard, Prediction, MapVisualization
```

## Pipeline flow

1. **Grid generation** — build a regular grid clipped to Nepal, assign stable
   `grid_id`s, and store centroids plus geometries.
2. **Feature extraction** — sample Sentinel-2 (optical indices), Sentinel-1
   (SAR backscatter), ERA5 (climate), terrain (DEM derivatives), and land cover
   at each grid cell.
3. **Merge and engineer** — join all sources by `grid_id`, run QA, derive
   seasonal and interaction features, and write the model-ready table.
4. **Train and evaluate** — compare baselines and gradient-boosted models with
   spatially aware validation; persist the best pipeline and metrics.
5. **Predict** — load artifacts through the standalone predictor, expose the
   result via the API, and visualise it in the dashboard.

All geospatial storage uses **EPSG:4326**; metric operations use a projected CRS
configured in `config.py` (default UTM zone 45N, EPSG:32645).

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for the backend environment
- Node.js 20+ and npm
- A Google Earth Engine account with a service account
- Git

### 1. Clone and create the ML virtual environment

```bash
git clone <repository-url>
cd NepalCropPrediction
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

The ML pipeline uses a pip-managed virtual environment. The backend is a
standalone uv project with locked dependencies; `uv sync` creates its own
environment at `backend/.venv`.

```bash
pip install -r ml_pipeline/requirements.txt
uv sync --project backend
```

### 3. Configure the environment

```bash
cp .env.example .env
```

Fill in the Earth Engine service-account details, Django secret, database URL,
and frontend API base URL. Place the service-account JSON key outside Git
(for example `secrets/gee-service-account.json`) and share the Earth Engine
registered project with that account.

### 4. Earth Engine access

1. Create a service account in Google Cloud and enable the Earth Engine API.
2. Register the Google Cloud project for Earth Engine (non-commercial use).
3. Download the JSON key and set `GEE_SERVICE_ACCOUNT`, `GEE_PRIVATE_KEY_PATH`,
   and `GEE_PROJECT_ID` in `.env`.

### 5. Phases 1–5: data collection and merge

Run the data pipeline in order:

```bash
python ml_pipeline/data_collection/01_generate_grid.py
python ml_pipeline/data_collection/02_extract_sentinel2.py
python ml_pipeline/data_collection/03_extract_sentinel1.py
python ml_pipeline/data_collection/04_extract_era5.py
python ml_pipeline/data_collection/08_extract_and_merge_historical.py
```

Set `TEST_MODE=True` in `.env` to restrict extraction to 100 grid cells and
2024 for a quick verification. Set `TEST_MODE=False` for the full 2018–2024
extraction over every grid cell.

Extraction is batched and resumable. Sentinel-2 and Sentinel-1 are reduced three
months at a time; ERA5-Land is reduced one year at a time. Monthly checkpoints
are written under `data/processed/{sentinel2,sentinel1,era5}/monthly/`, so an
interrupted run resumes without recomputing completed months.

Primary outputs from the full 2018–2024 run:

- `data/processed/sentinel2_features.csv` — 528,864 grid-month rows
  (6,296 cells × 84 months) with 13 optical bands and indices.
- `data/processed/sentinel1_features.csv` — 528,864 rows with VV/VH mean,
  median, standard deviation, and linear-power polarization ratios.
- `data/processed/era5_climate_features.csv` — 528,864 rows with temperature,
  rainfall, soil moisture, relative humidity, solar radiation, and wind speed.
- `data/processed/nepal_all_77_districts_dataset.csv` — 539 district-year rows
  (77 districts × 7 years) with climate, area, production, and crop yields.
- `data/final/nepal_crop_ml_training_dataset.csv` — 127,677 grid-year-crop rows
  covering 9 crops, with annual and quarterly features (152 columns).

Missing values reflect real source gaps; they are never filled with constants.
Sentinel-2 has almost no usable scenes over Nepal in July 2018 and none in
August 2018, so those monthly and quarterly columns remain null.

Parquet copies and run metadata (grid size, date range, scene counts, scale,
columns) are written alongside each CSV.

ERA5-Land uses `ECMWF/ERA5_LAND/HOURLY`; monthly temperature and soil moisture
are means, rainfall and solar radiation are sums, and rainfall is converted from
metres to millimetres. Historical yields combine the Open Data Nepal CKAN
2024/25 district tables with the cached Zenodo 2018–2023 maize/wheat panel.
Missing labels are left as nulls; they are never imputed with constants.

### 6. Backend

```bash
uv run --project backend backend/manage.py migrate
uv run --project backend backend/manage.py runserver
```

Use `uv sync --project backend --frozen` in deployment to install exactly the
versions recorded in `backend/uv.lock`.

### 7. Frontend

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server runs on `http://localhost:5173` and reads
`VITE_API_BASE_URL` (default `http://localhost:8000/api`).

## Development phases

| Phase | Deliverable | Key files |
| --- | --- | --- |
| 1 | National 5 km prediction grid | `01_generate_grid.py` |
| 2 | Sentinel-2 optical features | `02_extract_sentinel2.py` |
| 3 | Sentinel-1 SAR features | `03_extract_sentinel1.py` |
| 4 | ERA5-Land climate features | `04_extract_era5.py` |
| 5 | Historical ground truth and final merge | `08_extract_and_merge_historical.py` |
| 6 | Optional terrain and land-cover features | `05_extract_terrain.py`, `06_extract_landcover.py` |
| 7 | Feature engineering | `processing/feature_engineering.py` |
| 8 | Trained models and report | `training/train_models.py`, `training/evaluate.py` |
| 9 | REST API | `backend/apps/*` |
| 10 | Dashboard and map | `frontend/src/pages/*` |

Build in this order. Later phases must consume earlier outputs rather than
hard-coding substitutes.

## Planned API surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/satellite/grid/` | Grid metadata and cell lookup |
| `GET` | `/api/satellite/features/` | Satellite/climate features for a cell |
| `GET` | `/api/crops/` | Supported crops and metadata |
| `GET` | `/api/crops/seasons/` | Planting/harvest windows by crop |
| `POST` | `/api/predict/` | Predict yield for a grid cell or region |
| `GET` | `/api/predictions/` | Prediction history |

## Data and model artifacts

- `data/raw/` — immutable source downloads and GEE exports.
- `data/processed/` — per-source tables keyed by `grid_id`.
- `data/final/` — versioned model-ready datasets (GeoParquet).
- `models/` — preprocessing pipeline, trained model, feature schema, metadata.

Contents of `data/**` and `models/**` are Git-ignored; only `.gitkeep` files are
tracked. Back up generated artifacts to external/object storage rather than
committing them.

### Model validation

`training/train_models.py` predicts `yield_kg_ha` from numeric satellite/climate
features plus `crop` as a categorical input. Ground-truth yield is constant
within each `district × crop × year` and repeated across grid cells, so a random
row split leaks the target. Training therefore reports three schemes:

| Scheme | Purpose |
| --- | --- |
| `grouped_by_district` | Primary score: 20% of districts are held out entirely. |
| `temporal_year` | Forward-in-time check: the latest year is held out. |
| `random_reference` | Diagnostic only; quantifies how much a random split inflates scores. |

Metrics per model, per scheme, and per year are written to
`models/model_evaluation_metrics.csv`, `models/model_evaluation_by_year.json`,
and `models/training_metadata.json`.

## Testing and quality

```bash
pytest                      # pipeline tests
uv run --project backend backend/manage.py test
cd frontend && npm run build
```

Follow the conventions in [`AGENT.md`](AGENT.md): typed public functions,
docstrings, central configuration, no secrets or data in Git, and a regression
test for every bug fix.
