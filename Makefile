# Nepal Crop Prediction Platform - Makefile
# Uses the Python venv at .venv and pip-managed dependencies.
# Run all targets from the repository root.

PYTHON := python
PIP := pip
VENV := .venv
ML_DIR := ml_pipeline
DATA_DIR := data

.PHONY: all help install dev-install test lint typecheck clean grid s2 s1 era5 terrain landcover merge train evaluate predict api frontend

all: help

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make [target]\n\nTargets:\n"} \
	/^[a-zA-Z_-]+:.*?##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------- Environment ----------

install:  ## Install ML pipeline dependencies into .venv
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && $(PIP) install -r $(ML_DIR)/requirements.txt

dev-install: install  ## Install dev dependencies (pytest, black, ruff)
	. $(VENV)/bin/activate && $(PIP) install pytest black ruff

# ---------- Pipeline phases ----------

grid:  ## Phase 1: Generate 5km x 5km spatial grid
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/01_generate_grid.py

s2:  ## Phase 2: Extract Sentinel-2 optical features
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/02_extract_sentinel2.py

s1:  ## Phase 3: Extract Sentinel-1 SAR features
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/03_extract_sentinel1.py

era5:  ## Phase 4: Extract ERA5-Land climate features
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/04_extract_era5.py

terrain:  ## Phase 5: Extract terrain features
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/05_extract_terrain.py

landcover:  ## Phase 6: Extract land cover features
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/06_extract_landcover.py

merge:  ## Phase 5: Historical ground truth and final merge
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/08_extract_and_merge_historical.py

historical:  ## Phase 8: Extract & merge historical ground truth
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/data_collection/08_extract_and_merge_historical.py

train:  ## Phase 9: Train models
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/training/train_models.py

evaluate:  ## Phase 10: Evaluate trained models
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/training/evaluate.py

predict:  ## Phase 11: Run inference
	. $(VENV)/bin/activate && $(PYTHON) $(ML_DIR)/prediction/predictor.py

# ---------- Full pipeline ----------

pipeline: grid s2 s1 era5 terrain landcover merge historical  ## Run the full data pipeline
	@echo "Pipeline complete. See $(DATA_DIR)/processed/ and $(DATA_DIR)/final/"

# ---------- Test mode (fast) ----------

test-s2:  ## Run Sentinel-2 extraction in TEST_MODE (100 cells, 2024 only)
	. $(VENV)/bin/activate && TEST_MODE=True $(PYTHON) $(ML_DIR)/data_collection/02_extract_sentinel2.py

test-s1:  ## Run Sentinel-1 extraction in TEST_MODE
	. $(VENV)/bin/activate && TEST_MODE=True $(PYTHON) $(ML_DIR)/data_collection/03_extract_sentinel1.py

test-era5:  ## Run ERA5 extraction in TEST_MODE
	. $(VENV)/bin/activate && TEST_MODE=True $(PYTHON) $(ML_DIR)/data_collection/04_extract_era5.py

test-pipeline: test-s2 test-s1 test-era5  ## Run pipeline phases in TEST_MODE

# ---------- Backend ----------

backend-install:  ## Install backend dependencies with uv
	cd backend && uv sync

backend-migrate:  ## Run Django migrations
	cd backend && uv run python manage.py migrate

backend-run:  ## Run Django dev server
	cd backend && uv run python manage.py runserver

api: backend-run  ## Alias for backend-run

# ---------- Frontend ----------

frontend-install:  ## Install frontend dependencies
	cd frontend && npm install

frontend-dev:  ## Run Vite dev server
	cd frontend && npm run dev

frontend-build:  ## Build frontend for production
	cd frontend && npm run build

# ---------- Testing & Quality ----------

test:  ## Run pytest on the ML pipeline
	. $(VENV)/bin/activate && $(PYTHON) -m pytest -v

lint:  ## Lint with ruff
	. $(VENV)/bin/activate && ruff check $(ML_DIR)/

format:  ## Format code with black
	. $(VENV)/bin/activate && black $(ML_DIR)/

typecheck:  ## Type-check with mypy
	. $(VENV)/bin/activate && mypy $(ML_DIR)/

# ---------- Cleanup ----------

clean:  ## Remove generated data and models (keep .gitkeep)
	rm -rf $(DATA_DIR)/raw/grid.* $(DATA_DIR)/processed/* $(DATA_DIR)/final/*
	rm -rf models/*.joblib models/*.json models/*.onnx
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-venv:  ## Remove the virtual environment
	rm -rf $(VENV)