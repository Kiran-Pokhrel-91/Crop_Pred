"""Print saved training/evaluation metrics."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml_pipeline.config import MODELS_DIR


def main() -> None:
    meta_path = MODELS_DIR / "training_metadata.json"
    metrics_path = MODELS_DIR / "model_evaluation_metrics.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Training metadata not found at {meta_path}. Run training first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"Best model: {meta.get('best_model')}")
    print(f"Target: {meta.get('target')} ({meta.get('target_transform', 'none')})")
    print(f"Validation: {meta.get('validation')}")
    if metrics_path.exists():
        print("\nModel comparison:")
        print(metrics_path.read_text(encoding="utf-8").strip())
    print("\nPer-crop (selected model):")
    for row in meta.get("per_crop_metrics", []):
        r2 = row.get("R2")
        r2_str = f"{r2:.4f}" if r2 == r2 else "nan"
        print(f"  {row['Crop']:<12} n={row['Rows']:>5} R2={r2_str:>8} MAE={row['MAE']:.1f} RMSE={row['RMSE']:.1f}")


if __name__ == "__main__":
    main()

