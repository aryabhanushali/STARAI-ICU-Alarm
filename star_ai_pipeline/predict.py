# predict.py
# Load saved fold models, predict on test set, write submission.csv.
# Uses AUC-weighted rank-normalized ensemble + logistic regression blend.
# 499 test patients with no features get the training base rate as prediction.

import json
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import config
from data_loader import load_test_ids, setup_logging
from multi_task_encoder import build_features

log = logging.getLogger(__name__)


def load_models():
    paths = sorted(config.MODELS_DIR.glob("lgbm_fold*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No model files in {config.MODELS_DIR} — run train.py first")
    models = [joblib.load(p) for p in paths]
    log.info(f"Loaded {len(models)} fold models")

    auc_path = config.RESULTS_DIR / "fold_aucs.json"
    if auc_path.exists():
        with open(auc_path) as f:
            fold_aucs = json.load(f)["fold_aucs"]
        log.info(f"Loaded fold AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
    else:
        fold_aucs = [1.0] * len(models)
        log.warning("fold_aucs.json not found — using equal weights")

    return models, fold_aucs


def get_test_features(rebuild=False):
    if not rebuild and config.FEATURES_TEST.exists():
        log.info("Loading cached test features...")
        return pd.read_parquet(config.FEATURES_TEST)
    log.info("Building test features from scratch...")
    return build_features("test", save_path=config.FEATURES_TEST)


def predict(rebuild_features=False):
    test_ids          = load_test_ids()
    models, fold_aucs = load_models()

    test_df  = get_test_features(rebuild=rebuild_features)
    test_pid = test_df[config.PID_COL].astype(str).tolist()

    imp_path = config.RESULTS_DIR / "feature_importance.csv"
    if imp_path.exists():
        feat_cols = pd.read_csv(imp_path)["feature"].tolist()
    else:
        feat_cols = [c for c in test_df.columns
                     if c not in (config.PID_COL, "age", "sex_encoded")]

    for col in feat_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    X_test = test_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    assert X_test.isnull().sum().sum() == 0

    log.info(f"Predicting on {len(X_test)} test patients (AUC-weighted rank ensemble)...")

    auc_arr = np.array(fold_aucs)
    weights  = auc_arr / auc_arr.sum()

    log.info("Ensemble weights (proportional to fold validation AUC):")
    for i, (auc, w) in enumerate(zip(fold_aucs, weights)):
        log.info(f"  Fold {i+1}: val_AUC={auc:.4f}  weight={w:.4f}")

    probs = np.zeros(len(X_test))
    for i, (m, w) in enumerate(zip(models, weights)):
        raw_preds = m.predict(X_test)
        # convert to percentile ranks so different folds are on the same scale
        ranked    = rankdata(raw_preds) / len(raw_preds)
        probs    += w * ranked
        log.info(
            f"  Model {i+1}: raw_mean={raw_preds.mean():.4f} | "
            f"ranked_mean={ranked.mean():.4f} | weight={w:.4f}"
        )

    print("\n=== Prediction Distribution (rank-normalised) ===")
    print(f"  N: {len(probs):,}")
    print(f"  Mean:   {probs.mean():.4f}")
    print(f"  Std:    {probs.std():.4f}")
    print(f"  Min:    {probs.min():.4f}")
    print(f"  Max:    {probs.max():.4f}")
    print(f"  Median: {float(np.median(probs)):.4f}")
    for t in [0.3, 0.5, 0.7]:
        n = int((probs >= t).sum())
        print(f"  >= {t}: {n} ({n/len(probs)*100:.1f}%)")
    print()

    # logistic regression blend — captures linear structure LightGBM can underfit
    logreg_path = config.MODELS_DIR / "logreg_full.pkl"
    if logreg_path.exists():
        logreg_model = joblib.load(logreg_path)
        logreg_feat_cols = logreg_model.named_steps["scaler"].feature_names_in_.tolist()
        X_test_lr = X_test.copy()
        for col in logreg_feat_cols:
            if col not in X_test_lr.columns:
                X_test_lr[col] = 0.0
        logreg_probs = logreg_model.predict_proba(X_test_lr[logreg_feat_cols])[:, 1]
        log.info(f"Loaded logreg model, mean prediction: {logreg_probs.mean():.4f}")

        # select best blend ratio using OOF (no test-set tuning)
        best_ratio, best_oof_auc = 0.85, -1.0
        oof_path = config.RESULTS_DIR / "oof_predictions.csv"
        logreg_oof_path = config.RESULTS_DIR / "logreg_oof_predictions.csv"
        if oof_path.exists() and logreg_oof_path.exists():
            oof_df        = pd.read_csv(oof_path)
            logreg_oof_df = pd.read_csv(logreg_oof_path)
            lgbm_oof  = oof_df["oof_pred"].values
            lr_oof    = logreg_oof_df["logreg_oof_pred"].values
            y_oof     = oof_df["mortality"].values

            for ratio in [0.90, 0.85, 0.80]:
                blended_oof = ratio * lgbm_oof + (1 - ratio) * lr_oof
                auc = roc_auc_score(y_oof, blended_oof)
                log.info(f"  Blend ratio {ratio:.0%} LGBM / {1-ratio:.0%} LR -> OOF AUC: {auc:.4f}")
                if auc > best_oof_auc:
                    best_oof_auc = auc
                    best_ratio   = ratio

            log.info(f"Selected blend ratio: {best_ratio:.0%} LGBM / {1-best_ratio:.0%} LR "
                     f"(OOF AUC: {best_oof_auc:.4f})")
        else:
            log.warning("OOF files not found — defaulting to 85/15 blend without tuning")

        probs = best_ratio * probs + (1 - best_ratio) * logreg_probs
        probs = np.clip(probs, 0.0, 1.0)
        log.info(f"Blended predictions: mean={probs.mean():.4f}, std={probs.std():.4f}")
    else:
        log.info("logreg_full.pkl not found — using LGBM ensemble only (run train.py to enable blend)")

    id_to_prob = dict(zip(test_pid, probs.tolist()))

    # use training base rate for patients with no data (corrupted file tail)
    base_rate_path = config.RESULTS_DIR / "train_base_rate.txt"
    if base_rate_path.exists():
        fallback_prob = float(open(base_rate_path).read().strip())
    else:
        fallback_prob = 0.061
        log.warning("train_base_rate.txt not found — using hard-coded fallback 0.061")

    n_missing = sum(1 for pid in test_ids if pid not in id_to_prob)
    if n_missing:
        log.warning(
            f"{n_missing} test patients had no features (corrupted file tail). "
            f"Assigning training base rate {fallback_prob:.4f}."
        )

    final_probs = [id_to_prob.get(pid, fallback_prob) for pid in test_ids]

    submission = pd.DataFrame({"id": test_ids, "mortality": final_probs})

    assert len(submission) == config.N_TEST,            f"Wrong row count: {len(submission)}"
    assert list(submission.columns) == ["id", "mortality"], "Wrong columns"
    assert submission.isnull().sum().sum() == 0,        "NaNs in submission"
    assert submission["mortality"].between(0, 1).all(), "Probabilities out of [0,1]"
    assert list(submission["id"]) == test_ids,          "Row order doesn't match test.json"

    config.SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    log.info(f"Submission saved: {config.SUBMISSION_PATH}  ({len(submission)} rows)")

    return submission


if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    predict(rebuild_features=args.rebuild_features)
