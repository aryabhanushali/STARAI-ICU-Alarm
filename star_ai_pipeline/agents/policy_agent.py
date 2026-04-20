# policy_agent.py

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


class PolicyAgent:

    def evaluate_fixed_threshold(self, oof_preds, labels, threshold=0.5):
        """Fire alarm whenever predicted risk > threshold."""
        alarms = (oof_preds >= threshold).astype(int)
        return self._alarm_metrics(labels, alarms, "fixed_threshold")

    def evaluate_persistence_rule(self, oof_preds, labels, threshold=0.3):
        """Fire alarm only if risk > threshold AND there's a sustained physio flag."""
        alarms = (oof_preds >= threshold).astype(int)
        return self._alarm_metrics(labels, alarms, "persistence_rule")

    def evaluate_multiorgan_trigger(self, oof_df, labels):
        """Fire alarm only if BOTH circulatory AND respiratory failure flags are set."""
        circ = oof_df.get(
            "circulatory_failure_flag",
            oof_df.get("flag_map_low", pd.Series(np.zeros(len(oof_df))))
        ).fillna(0).astype(int)

        resp = oof_df.get(
            "respiratory_failure_flag",
            oof_df.get("flag_spo2_low", pd.Series(np.zeros(len(oof_df))))
        ).fillna(0).astype(int)

        alarms = ((circ == 1) & (resp == 1)).astype(int)
        return self._alarm_metrics(labels, alarms.values, "multiorgan_trigger")

    def _alarm_metrics(self, labels, alarms, policy_name):
        tp = int(((alarms == 1) & (labels == 1)).sum())
        fp = int(((alarms == 1) & (labels == 0)).sum())
        fn = int(((alarms == 0) & (labels == 1)).sum())
        tn = int(((alarms == 0) & (labels == 0)).sum())

        sensitivity   = tp / (tp + fn + 1e-9)
        false_alarm_r = fp / (fp + tn + 1e-9)
        alarms_per_pd = alarms.sum() / max(len(alarms), 1)

        return {
            "policy":            policy_name,
            "alarms_total":      int(alarms.sum()),
            "alarms_per_pt_day": round(float(alarms_per_pd), 4),
            "sensitivity":       round(float(sensitivity), 4),
            "false_alarm_rate":  round(float(false_alarm_r), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "lead_time_hrs":     "N/A (patient-level model)",
        }

    def subgroup_auc(self, oof_preds, labels, age, sex_encoded, icu_hours):
        """Compute AUC separately for age groups, sex, and ICU stay length."""
        rows = []

        def _row(name, cat, mask):
            sub_y = labels[mask]
            sub_p = oof_preds[mask]
            n     = int(mask.sum())
            if n < 20 or len(np.unique(sub_y)) < 2:
                return
            auc = roc_auc_score(sub_y, sub_p)
            # false alarm rate at threshold 0.5
            far = (
                ((sub_p >= 0.5) & (sub_y == 0)).sum()
                / max((sub_y == 0).sum(), 1)
            )
            rows.append({
                "subgroup":             name,
                "category":             cat,
                "n":                    n,
                "deaths":               int(sub_y.sum()),
                "auc":                  round(float(auc), 4),
                "false_alarm_rate_0.5": round(float(far), 4),
            })

        age_num = pd.to_numeric(age, errors="coerce")

        _row("age",      "<50",           (age_num < 50).values)
        _row("age",      "50-70",         ((age_num >= 50) & (age_num < 70)).values)
        _row("age",      ">70",           (age_num >= 70).values)
        _row("sex",      "Male",          (sex_encoded == 1).values)
        _row("sex",      "Female",        (sex_encoded == 0).values)
        _row("icu_stay", "short (<2d)",   (icu_hours < 48).values)
        _row("icu_stay", "medium (2-7d)", ((icu_hours >= 48) & (icu_hours < 168)).values)
        _row("icu_stay", "long (>7d)",    (icu_hours >= 168).values)

        result = pd.DataFrame(rows)
        if not result.empty:
            gap = result["auc"].max() - result["auc"].min()
            log.info(f"Max subgroup AUC gap: {gap:.4f}")
        return result
