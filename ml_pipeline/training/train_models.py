"""Train and persist a crop-yield model.

Uses grouped cross-validation by district to avoid spatial leakage, a log1p
target transform to stabilise the wide per-crop yield scale, and one-standard-
error model selection to prefer simpler models when scores are comparable.

Outputs under ``models/``:
    best_crop_model.joblib
    model_evaluation_metrics.csv
    training_metadata.json
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_pipeline.config import FINAL_DIR, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "latitude", "longitude",
    "B2_mean", "B3_mean", "B4_mean", "B8_mean", "B11_mean", "B12_mean",
    "NDVI_mean", "EVI_mean", "NDWI_mean", "SAVI_mean", "MSAVI_mean", "GNDVI_mean", "NDBI_mean",
    "VV_mean", "VV_median", "VV_std", "VH_mean", "VH_median", "VH_std", "VV_VH_mean", "VH_VV_mean",
    "temperature_c_mean", "rainfall_mm_mean", "soil_moisture_mean",
    "relative_humidity_mean", "solar_radiation_mean", "wind_speed_mean",
]
CATEGORICAL_FEATURES = ["crop"]
TARGET = "yield_kg_ha"
GROUP_COLUMN = "district"
CV_FOLDS = 3
SIMPLICITY = {"Ridge": 0, "Random Forest": 1, "LightGBM": 2, "CatBoost": 3}


def _load_data() -> pd.DataFrame:
    path = FINAL_DIR / "nepal_crop_ml_training_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run the merge step first.")
    df = pd.read_csv(path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, GROUP_COLUMN]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def _build_models() -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    estimators = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMRegressor
        estimators["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbose=-1)
    except ImportError:
        logger.warning("LightGBM not installed; skipping.")
    try:
        from catboost import CatBoostRegressor
        estimators["CatBoost"] = CatBoostRegressor(iterations=200, learning_rate=0.1, depth=6, verbose=0, random_state=42)
    except ImportError:
        logger.warning("CatBoost not installed; skipping.")

    return {
        name: Pipeline([
            ("preprocess", clone(preprocess)),
            ("model", TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)),
        ])
        for name, estimator in estimators.items()
    }


def _score(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.ptp(y_true) > 0 else float("nan")
    return {
        "R2": r2,
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def _cross_validate(models, X, y, groups):
    splitter = GroupKFold(n_splits=CV_FOLDS)
    fold_scores = {name: [] for name in models}
    oof = {name: np.empty(len(y)) for name in models}

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        logger.info("Fold %d/%d: train=%d, test=%d", fold, CV_FOLDS, len(train_idx), len(test_idx))
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]
        for name, model in models.items():
            pred = clone(model).fit(Xtr, ytr).predict(Xte)
            oof[name][test_idx] = pred
            fold_scores[name].append(_score(yte, pred))

    rows = []
    for name, scores in fold_scores.items():
        rows.append({
            "Model": name,
            "CV_R2_Mean": float(np.mean([s["R2"] for s in scores])),
            "CV_R2_Std": float(np.std([s["R2"] for s in scores], ddof=1)),
            "CV_RMSE": float(np.mean([s["RMSE"] for s in scores])),
            "CV_MAE": float(np.mean([s["MAE"] for s in scores])),
            "Folds": CV_FOLDS,
            "Simplicity": SIMPLICITY.get(name, 99),
        })
    return pd.DataFrame(rows).sort_values("CV_R2_Mean", ascending=False), oof


def _select_model(results: pd.DataFrame) -> str:
    best = results.iloc[0]
    se = best["CV_R2_Std"] / np.sqrt(best["Folds"])
    eligible = results[results["CV_R2_Mean"] >= best["CV_R2_Mean"] - se]
    selected = eligible.sort_values(["Simplicity", "CV_R2_Mean"], ascending=[True, False]).iloc[0]["Model"]
    logger.info("Selected %s (one-SE threshold %.4f)", selected, best["CV_R2_Mean"] - se)
    return str(selected)


def _per_crop_metrics(y, pred, crops) -> list[dict]:
    rows = []
    for crop in sorted(pd.unique(crops)):
        mask = crops == crop
        rows.append({"Crop": crop, "Rows": int(mask.sum()), **_score(y[mask], pred[mask])})
    return rows


def main() -> None:
    logger.info("=== Training crop-yield model ===")
    df = _load_data()
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, GROUP_COLUMN]
    data = df[cols].dropna()
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET]
    logger.info("Rows: %d | districts: %d | crops: %d", len(data), data[GROUP_COLUMN].nunique(), data["crop"].nunique())

    models = _build_models()
    results, oof = _cross_validate(models, X, y, data[GROUP_COLUMN])
    best_name = _select_model(results)
    results["Selected"] = results["Model"] == best_name
    for _, row in results.iterrows():
        logger.info("  %-14s R2=%.4f +/- %.4f | RMSE=%.1f | MAE=%.1f", row["Model"], row["CV_R2_Mean"], row["CV_R2_Std"], row["CV_RMSE"], row["CV_MAE"])

    per_crop = _per_crop_metrics(y.to_numpy(), oof[best_name], data["crop"].to_numpy())
    for row in per_crop:
        logger.info("  %-12s n=%5d | R2=%7.4f | MAE=%8.1f", row["Crop"], row["Rows"], row["R2"], row["MAE"])

    final_model = clone(models[best_name]).fit(X, y)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "model_name": best_name,
        "target": TARGET,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_transform": "log1p",
        "fitted_on_rows": len(data),
        "cv_summary": results.loc[results["Selected"]].drop(columns=["Selected", "Simplicity"]).to_dict("records")[0],
        "per_crop_metrics": per_crop,
    }
    joblib.dump(artifact, MODELS_DIR / "best_crop_model.joblib")
    results.drop(columns=["Simplicity"]).to_csv(MODELS_DIR / "model_evaluation_metrics.csv", index=False)
    (MODELS_DIR / "training_metadata.json").write_text(json.dumps({
        "best_model": best_name,
        "target": TARGET,
        "target_transform": "log1p",
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "validation": f"GroupKFold by {GROUP_COLUMN}, {CV_FOLDS} folds",
        "selection_rule": "one-standard-error, prefer simpler model",
        "cv_metrics": results.drop(columns=["Simplicity"]).to_dict("records"),
        "per_crop_metrics": per_crop,
        "usable_rows": len(data),
        "districts": int(data[GROUP_COLUMN].nunique()),
        "crops": int(data["crop"].nunique()),
    }, indent=2), encoding="utf-8")
    logger.info("=== Saved %s ===", MODELS_DIR / "best_crop_model.joblib")


if __name__ == "__main__":
    main()

