# predict.py
# Load fold models, predict on test set, write submission.csv.
# Blend: 10% LightGBM ensemble + 90% logistic regression (best on public LB).

import json
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config
from data_loader import load_test_ids, setup_logging
from multi_task_encoder import build_features

LGBM_WEIGHT = 0.10  # tuned on Kaggle public LB

log = logging.getLogger(__name__)


def load_models():
    paths = sorted(config.MODELS_DIR.glob("lgbm_fold*.pkl"))
    models = [joblib.load(p) for p in paths]
    with open(config.RESULTS_DIR / "fold_aucs.json") as f:
        fold_aucs = json.load(f)["fold_aucs"]
    log.info(f"Loaded {len(models)} fold models, AUCs={[f'{a:.4f}' for a in fold_aucs]}")
    return models, fold_aucs


def get_test_features(rebuild=False):
    if not rebuild and config.FEATURES_TEST.exists():
        return pd.read_parquet(config.FEATURES_TEST)
    return build_features("test", save_path=config.FEATURES_TEST)


def predict(rebuild_features=False):
    test_ids          = load_test_ids()
    models, fold_aucs = load_models()
    test_df           = get_test_features(rebuild=rebuild_features)
    test_pid          = test_df[config.PID_COL].astype(str).tolist()

    feat_cols = pd.read_csv(config.RESULTS_DIR / "feature_importance.csv")["feature"].tolist()
    for col in feat_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0
    X_test = test_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # AUC-weighted average of rank-normalised fold predictions
    weights = np.array(fold_aucs) / sum(fold_aucs)
    lgbm_probs = np.zeros(len(X_test))
    for m, w in zip(models, weights):
        raw = m.predict(X_test)
        lgbm_probs += w * (rankdata(raw) / len(raw))

    # Logistic regression baseline
    logreg = joblib.load(config.MODELS_DIR / "logreg_full.pkl")
    lr_cols = logreg.named_steps["scaler"].feature_names_in_.tolist()
    X_lr = X_test.copy()
    for col in lr_cols:
        if col not in X_lr.columns:
            X_lr[col] = 0.0
    lr_probs = logreg.predict_proba(X_lr[lr_cols])[:, 1]

    # Final blend
    probs = LGBM_WEIGHT * lgbm_probs + (1 - LGBM_WEIGHT) * lr_probs
    probs = np.clip(probs, 0.0, 1.0)
    log.info(f"Blended predictions: mean={probs.mean():.4f}, std={probs.std():.4f}")

    # Patients with no features get the training base rate
    id_to_prob    = dict(zip(test_pid, probs.tolist()))
    fallback_prob = float(open(config.RESULTS_DIR / "train_base_rate.txt").read().strip())
    final_probs   = [id_to_prob.get(pid, fallback_prob) for pid in test_ids]

    submission = pd.DataFrame({"id": test_ids, "mortality": final_probs})
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    log.info(f"Saved {config.SUBMISSION_PATH} ({len(submission)} rows)")
    return submission


if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    predict(rebuild_features=args.rebuild_features)
