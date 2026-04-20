# alarm_policy_engine.py — Stage 5
# Simulates 3 alarm policies on OOF predictions (methodology report only).
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
    log.info("[Stage 5] Running alarm_policy_engine...")

    oof_path  = config.RESULTS_DIR / "oof_predictions.csv"
    feat_path = config.FEATURES_TRAIN

    if not oof_path.exists():
        log.warning("[Stage 5] oof_predictions.csv not found — run train.py first")
        return pd.DataFrame()
    if not feat_path.exists():
        log.warning("[Stage 5] features_train.parquet not found — run train.py first")
        return pd.DataFrame()

    oof_df   = pd.read_csv(oof_path)
    feat_df  = pd.read_parquet(feat_path).reset_index(drop=True)

    assert len(oof_df) == len(feat_df), "OOF and feature row counts differ"

    oof_preds = oof_df["oof_pred"].values
    labels    = oof_df["mortality"].values.astype(int)

    log.info(f"[Stage 5] Overall OOF AUC: {roc_auc_score(labels, oof_preds):.4f}")

    agent = PolicyAgent()

    p1 = agent.evaluate_fixed_threshold(oof_preds, labels, threshold=0.5)

    # persistence rule: score > 0.3 AND a sustained physio flag
    if "flag_map_persistent" in feat_df.columns or "flag_spo2_persistent" in feat_df.columns:
        persistence_mask = (
            feat_df.get("flag_map_persistent",  pd.Series(np.zeros(len(feat_df)))).fillna(0).astype(int) |
            feat_df.get("flag_spo2_persistent",  pd.Series(np.zeros(len(feat_df)))).fillna(0).astype(int)
        ).astype(bool).values
        alarm_p2 = ((oof_preds > 0.3) & persistence_mask).astype(int)
        p2 = agent._alarm_metrics(labels, alarm_p2, "persistence_rule")
    else:
        p2 = agent.evaluate_persistence_rule(oof_preds, labels, threshold=0.3)

    p3 = agent.evaluate_multiorgan_trigger(feat_df, labels)

    results = pd.DataFrame([p1, p2, p3])

    print("\n" + "=" * 60)
    print("ALARM POLICY COMPARISON (OOF predictions, training set)")
    print("=" * 60)
    print(results[["policy", "alarms_total", "alarms_per_pt_day",
                   "sensitivity", "false_alarm_rate"]].to_string(index=False))
    print("=" * 60 + "\n")

    out_path = config.RESULTS_DIR / "alarm_policy_comparison.csv"
    results.to_csv(out_path, index=False)
    log.info(f"[Stage 5] Saved {out_path}")
    return results


if __name__ == "__main__":
    setup_logging()
    run()
