# equity_dashboard.py — Stage 6
# Checks AUC separately for age groups, sex, and ICU stay length.
# Does not affect the Kaggle submission.

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from data_loader import setup_logging
from agents.policy_agent import PolicyAgent

log = logging.getLogger(__name__)


def run():
    log.info("[Stage 6] Running equity_dashboard...")

    oof_path  = config.RESULTS_DIR / "oof_predictions.csv"
    feat_path = config.FEATURES_TRAIN

    if not oof_path.exists():
        log.warning("[Stage 6] oof_predictions.csv not found — run train.py first")
        return pd.DataFrame()
    if not feat_path.exists():
        log.warning("[Stage 6] features_train.parquet not found — run train.py first")
        return pd.DataFrame()

    oof_df  = pd.read_csv(oof_path)
    feat_df = pd.read_parquet(feat_path).reset_index(drop=True)

    assert len(oof_df) == len(feat_df), "OOF and feature row counts differ"

    oof_preds = oof_df["oof_pred"].values
    labels    = oof_df["mortality"].values.astype(int)

    age         = feat_df.get("age",              pd.Series(np.full(len(feat_df), np.nan)))
    sex_encoded = feat_df.get("sex_encoded",      pd.Series(np.full(len(feat_df), -1)))
    icu_hours   = feat_df.get("icu_duration_hours",
                  feat_df.get("icu_stay_hours",   pd.Series(np.zeros(len(feat_df)))))

    agent  = PolicyAgent()
    result = agent.subgroup_auc(oof_preds, labels, age, sex_encoded, icu_hours)

    if result.empty:
        log.warning("[Stage 6] No subgroup results")
        return result

    overall_auc = roc_auc_score(labels, oof_preds)
    max_gap     = result["auc"].max() - result["auc"].min()

    print("\n" + "=" * 65)
    print("SUBGROUP FAIRNESS ANALYSIS (OOF predictions, training set)")
    print("=" * 65)
    print(f"  Overall AUC : {overall_auc:.4f}")
    print(f"  Max Gap     : {max_gap:.4f}  "
          f"('{result.loc[result.auc.idxmax(), 'category']}' vs "
          f"'{result.loc[result.auc.idxmin(), 'category']}')")
    print()
    print(result.to_string(index=False))
    print("=" * 65 + "\n")

    out_path = config.RESULTS_DIR / "equity_analysis.csv"
    result.to_csv(out_path, index=False)
    log.info(f"[Stage 6] Saved {out_path} — max gap: {max_gap:.4f}")
    return result


if __name__ == "__main__":
    setup_logging()
    run()
