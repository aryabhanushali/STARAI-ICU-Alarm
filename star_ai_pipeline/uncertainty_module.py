# uncertainty_module.py — Stage 4
# Uncertainty = std deviation across the 5 fold predictions.
# Patients with std > 0.15 are flagged for human review.
# The Kaggle submission still uses the weighted-mean point estimate.

import json
import logging

import joblib
import numpy as np
import pandas as pd

import config
from data_loader import load_test_ids, setup_logging
from agents.prediction_agent import PredictionAgent

log = logging.getLogger(__name__)


def run(rebuild_features=False):
    log.info("[Stage 4] Running uncertainty_module...")

    model_paths = sorted(config.MODELS_DIR.glob("lgbm_fold*.pkl"))
    if not model_paths:
        raise FileNotFoundError(f"No models in {config.MODELS_DIR}. Run train.py first.")
    models = [joblib.load(p) for p in model_paths]
    log.info(f"[Stage 4] Loaded {len(models)} fold models")

    auc_path = config.RESULTS_DIR / "fold_aucs.json"
    fold_aucs = json.load(open(auc_path))["fold_aucs"] if auc_path.exists() else [1.0] * len(models)

    if not config.FEATURES_TEST.exists():
        from predict import get_test_features
        get_test_features(rebuild=rebuild_features)

    test_df  = pd.read_parquet(config.FEATURES_TEST)
    test_ids = load_test_ids()

    imp_path  = config.RESULTS_DIR / "feature_importance.csv"
    feat_cols = pd.read_csv(imp_path)["feature"].tolist() if imp_path.exists() else [
        c for c in test_df.columns if c not in (config.PID_COL, "age", "sex_encoded")
    ]

    for col in feat_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test   = test_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test_pid = test_df[config.PID_COL].astype(str).tolist()

    fold_preds = np.vstack([m.predict(X_test) for m in models])

    agent  = PredictionAgent()
    unc_df = agent.predict_with_uncertainty(fold_preds, fold_aucs)
    unc_df.insert(0, config.PID_COL, test_pid)

    fallback = float(open(config.RESULTS_DIR / "train_base_rate.txt").read()) \
               if (config.RESULTS_DIR / "train_base_rate.txt").exists() else 0.061

    full_rows = []
    n_missing = 0
    for pid in test_ids:
        row = unc_df[unc_df[config.PID_COL] == pid]
        if len(row) == 0:
            full_rows.append({config.PID_COL: pid, "point_estimate": fallback,
                              "uncertainty": 0.0, "high_uncertainty": False})
            n_missing += 1
        else:
            full_rows.append(row.iloc[0].to_dict())

    if n_missing:
        log.warning(f"[Stage 4] {n_missing} missing test patients → fallback {fallback:.4f}")

    result   = pd.DataFrame(full_rows)
    out_path = config.RESULTS_DIR / "predictions_with_uncertainty.csv"
    result.to_csv(out_path, index=False)

    n_high = int(result["high_uncertainty"].sum())
    log.info(f"[Stage 4] High-uncertainty patients: {n_high} / {len(result)} ({n_high/len(result)*100:.1f}%)")
    return result


if __name__ == "__main__":
    setup_logging()
    run()
