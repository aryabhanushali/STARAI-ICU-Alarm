# prediction_agent.py

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class PredictionAgent:

    def predict_with_uncertainty(self, fold_predictions, fold_aucs):
        """
        fold_predictions: shape (5, n_patients)
        fold_aucs: list of 5 validation AUCs used as blend weights

        Returns DataFrame with point_estimate, uncertainty, high_uncertainty.
        uncertainty = std across folds; flagged if std > 0.15.
        """
        weights = np.array(fold_aucs)
        weights = weights / weights.sum()

        point_estimate   = np.average(fold_predictions, axis=0, weights=weights)
        uncertainty      = np.std(fold_predictions, axis=0)
        high_uncertainty = uncertainty > 0.15

        n_high = int(high_uncertainty.sum())
        log.info(
            f"Uncertainty: mean={uncertainty.mean():.4f}, "
            f"max={uncertainty.max():.4f}, "
            f"flagged (>0.15): {n_high} ({n_high/len(uncertainty)*100:.1f}%)"
        )

        return pd.DataFrame({
            "point_estimate":  point_estimate,
            "uncertainty":     uncertainty,
            "high_uncertainty": high_uncertainty,
        })

    def blend_with_logreg(self, lgbm_preds, logreg_preds, lgbm_weight=0.85):
        """Mix LightGBM and logistic regression predictions."""
        return np.clip(
            lgbm_weight * lgbm_preds + (1 - lgbm_weight) * logreg_preds,
            0.0, 1.0
        )
