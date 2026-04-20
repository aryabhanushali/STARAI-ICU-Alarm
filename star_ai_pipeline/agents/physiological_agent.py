# physiological_agent.py

import logging
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

log = logging.getLogger(__name__)


class PhysiologicalAgent:

    def load_and_resample(self, patient_df):
        """Resample to 2-min windows by averaging readings per epoch."""
        pid = patient_df[config.PID_COL].iloc[0]

        df = patient_df.copy()
        df["_ts"] = pd.to_datetime(df[config.TRAIN_TIME], errors="coerce")
        df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)

        if df.empty:
            return df

        t0 = df["_ts"].iloc[0]
        df["_epoch"] = ((df["_ts"] - t0).dt.total_seconds() // 120).astype(int)

        num_cols = [c for c in config.FEATURE_COLS if c in df.columns]
        agg_dict = {c: "mean" for c in num_cols}
        agg_dict["_ts"] = "first"

        resampled = (
            df.groupby("_epoch", sort=True)
            .agg(agg_dict)
            .reset_index(drop=True)
        )
        resampled[config.PID_COL] = pid
        resampled[config.TRAIN_TIME] = resampled["_ts"]
        resampled = resampled.drop(columns=["_ts"])

        return resampled

    def flag_missingness(self, df):
        """Compute missing rate per variable for a single patient."""
        pid = df[config.PID_COL].iloc[0]
        rows = []
        for col in config.FEATURE_COLS:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce")
                missing_rate = series.isna().mean()
            else:
                missing_rate = 1.0
            rows.append({
                "patientid":    pid,
                "variable":     col,
                "missing_rate": round(float(missing_rate), 4),
            })
        return pd.DataFrame(rows)

    def merge_demographics(self, df, general_table):
        """Join age and sex onto the feature table. Unknown sex gets encoded as -1."""
        gen = general_table[["patientid", "age", "sex"]].copy()
        gen["patientid"]   = gen["patientid"].astype(str)
        gen["sex_encoded"] = gen["sex"].map({"M": 1, "F": 0}).fillna(-1).astype(int)
        gen["age"]         = pd.to_numeric(gen["age"], errors="coerce")
        gen = gen.drop(columns=["sex"])

        df = df.copy()
        df["patientid"] = df["patientid"].astype(str)
        df = df.merge(gen, on="patientid", how="left")

        # fill missing age with cohort median
        df["age"]         = df["age"].fillna(df["age"].median())
        df["sex_encoded"] = df["sex_encoded"].fillna(-1).astype(int)

        return df
