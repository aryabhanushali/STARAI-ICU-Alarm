# features.py
# Collapse each patient's time-series into a single feature row.

import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import config
from data_loader import (
    iter_test_patients,
    iter_train_patients,
    load_test_ids,
    load_train_labels,
    setup_logging,
)

log = logging.getLogger(__name__)


def compute_slope(values, times):
    if len(values) < config.MIN_TREND_OBS:
        return 0.0
    try:
        slope, _, _, _, _ = sp_stats.linregress(times, values)
        return float(slope) if np.isfinite(slope) else 0.0
    except Exception:
        return 0.0


def agg_variable(series, times, name):
    out = {}
    n_total = len(series)
    valid   = series.dropna()
    n_valid = len(valid)

    out[f"{name}__missing_rate"] = 1.0 - (n_valid / n_total) if n_total > 0 else 1.0

    if n_valid == 0:
        for s in config.AGG_STATS:
            out[f"{name}__{s}"] = 0.0
        return out

    v = valid.values.astype(float)
    t = times[series.notna()].values.astype(float)

    out[f"{name}__mean"]   = float(np.mean(v))
    out[f"{name}__std"]    = float(np.std(v)) if n_valid > 1 else 0.0
    out[f"{name}__min"]    = float(np.min(v))
    out[f"{name}__max"]    = float(np.max(v))
    out[f"{name}__median"] = float(np.median(v))
    out[f"{name}__p25"]    = float(np.percentile(v, 25))
    out[f"{name}__p75"]    = float(np.percentile(v, 75))
    out[f"{name}__first"]  = float(v[0])
    out[f"{name}__last"]   = float(v[-1])
    out[f"{name}__trend"]  = compute_slope(v, t)

    return out


def delta_features(series, times, name):
    # biggest single-step change and fastest rate of change
    valid = series.dropna()
    if len(valid) < 2:
        return {f"{name}__max_delta": 0.0, f"{name}__max_rate": 0.0}

    v  = valid.values.astype(float)
    t  = times[series.notna()].values.astype(float)
    dv = np.abs(np.diff(v))
    dt = np.diff(t)
    dt[dt == 0] = 1.0  # avoid divide by zero

    return {
        f"{name}__max_delta": float(np.max(dv)),
        f"{name}__max_rate":  float(np.max(dv / dt)),
    }


def clinical_flags(df):
    # binary flags for deterioration events that are known mortality predictors
    flags = {}

    if config.MAP_COL in df.columns:
        map_vals = pd.to_numeric(df[config.MAP_COL], errors="coerce").dropna()
        flags["flag_map_low"]        = int((map_vals < config.MAP_LOW).any())
        consec = (map_vals < config.MAP_LOW).astype(int)
        flags["flag_map_persistent"] = int((consec.rolling(2).sum() >= 2).any())
    else:
        flags["flag_map_low"] = flags["flag_map_persistent"] = 0

    if config.SPO2_COL in df.columns:
        spo2_vals = pd.to_numeric(df[config.SPO2_COL], errors="coerce").dropna()
        flags["flag_spo2_low"]        = int((spo2_vals < config.SPO2_LOW).any())
        consec = (spo2_vals < config.SPO2_LOW).astype(int)
        flags["flag_spo2_persistent"] = int((consec.rolling(2).sum() >= 2).any())
    else:
        flags["flag_spo2_low"] = flags["flag_spo2_persistent"] = 0

    vaso_cols = [c for c in config.VASOPRESSOR_COLS if c in df.columns]
    if vaso_cols:
        vaso_any = df[vaso_cols].apply(pd.to_numeric, errors="coerce").gt(0).any(axis=None)
        flags["flag_vasopressor"] = int(bool(vaso_any))
    else:
        flags["flag_vasopressor"] = 0

    if config.MV_COL in df.columns:
        mv_vals = pd.to_numeric(df[config.MV_COL], errors="coerce").dropna()
        flags["flag_mech_vent"] = int((mv_vals > 0).any())
    else:
        flags["flag_mech_vent"] = 0

    return flags


def early_window_features(df, times):
    # features from just the first 24h
    FIRST_24H  = 24 * 3600
    early_mask = times <= FIRST_24H
    df_early   = df[early_mask.values]

    feats = {}

    # min MAP in first 24h (low MAP = circulatory compromise)
    if config.MAP_COL in df.columns and not df_early.empty:
        map_vals = pd.to_numeric(df_early[config.MAP_COL], errors="coerce").dropna()
        feats["early_map_min"] = float(map_vals.min()) if len(map_vals) > 0 else 0.0
    else:
        feats["early_map_min"] = 0.0

    # min SpO2 in first 24h
    if config.SPO2_COL in df.columns and not df_early.empty:
        spo2_vals = pd.to_numeric(df_early[config.SPO2_COL], errors="coerce").dropna()
        feats["early_spo2_min"] = float(spo2_vals.min()) if len(spo2_vals) > 0 else 0.0
    else:
        feats["early_spo2_min"] = 0.0

    # max heart rate in first 24h
    HR_COL = "vm1"
    if HR_COL in df.columns and not df_early.empty:
        hr_vals = pd.to_numeric(df_early[HR_COL], errors="coerce").dropna()
        feats["early_hr_max"] = float(hr_vals.max()) if len(hr_vals) > 0 else 0.0
    else:
        feats["early_hr_max"] = 0.0

    map_critical  = feats["early_map_min"]  > 0 and feats["early_map_min"]  < config.MAP_LOW
    spo2_critical = feats["early_spo2_min"] > 0 and feats["early_spo2_min"] < config.SPO2_LOW
    feats["early_deterioration_flag"] = int(map_critical or spo2_critical)

    return feats


def icu_duration(df, time_col):
    ts = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if ts.empty:
        return 0.0
    return max(0.0, (ts.max() - ts.min()).total_seconds() / 3600.0)


def patient_features(df, time_col):
    pid = df[config.PID_COL].iloc[0]
    feats = {config.PID_COL: pid}

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    times  = (parsed - parsed.min()).dt.total_seconds().fillna(0.0)

    for col in config.FEATURE_COLS:
        if col not in df.columns:
            for s in config.AGG_STATS:
                feats[f"{col}__{s}"] = 0.0
            feats[f"{col}__missing_rate"] = 1.0
        else:
            series = pd.to_numeric(df[col], errors="coerce")
            feats.update(agg_variable(series, times, col))

    for col in config.DELTA_COLS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            feats.update(delta_features(series, times, col))
        else:
            feats[f"{col}__max_delta"] = 0.0
            feats[f"{col}__max_rate"]  = 0.0

    feats.update(clinical_flags(df))
    feats.update(early_window_features(df, times))

    feats["icu_duration_hours"] = icu_duration(df, time_col)
    feats["n_observations"]     = len(df)

    return feats


def build_features(split, save_path=None):
    assert split in ("train", "test"), f"split must be 'train' or 'test', got {split}"
    log.info(f"Building {split} features...")

    iterator = iter_train_patients() if split == "train" else iter_test_patients()
    time_col = config.TRAIN_TIME    if split == "train" else config.TEST_TIME

    rows = []
    n = 0
    for df in iterator:
        rows.append(patient_features(df, time_col))
        n += 1
        if n % 1000 == 0:
            log.info(f"  processed {n} patients")

    log.info(f"Done: {n} patients")

    features_df = pd.DataFrame(rows)
    assert len(features_df) > 0, "feature matrix is empty"

    if split == "train":
        labels = load_train_labels()
        labels["patientid"] = labels["patientid"].astype(str)
        features_df["patientid"] = features_df["patientid"].astype(str)

        features_df = features_df.merge(
            labels[["patientid", "age", "sex", "mortality"]],
            on="patientid",
            how="left",
        )
        features_df["sex_encoded"] = features_df["sex"].map({"M": 1, "F": 0}).fillna(-1).astype(int)
        features_df = features_df.drop(columns=["sex"])
        features_df["age"] = pd.to_numeric(features_df["age"], errors="coerce")

        assert features_df["mortality"].notna().all(), "some patients missing mortality label"
        log.info(f"Train: {len(features_df)} patients x {features_df.shape[1]} features")

    else:
        # merge age + sex from general_table so test features match train
        if config.GENERAL_TABLE.exists():
            gen = pd.read_csv(config.GENERAL_TABLE)
            gen["patientid"] = gen["patientid"].astype(str)
            gen["sex_encoded"] = gen["sex"].map({"M": 1, "F": 0}).fillna(-1).astype(int)
            gen["age"] = pd.to_numeric(gen["age"], errors="coerce")
            features_df["patientid"] = features_df["patientid"].astype(str)
            features_df = features_df.merge(
                gen[["patientid", "age", "sex_encoded"]],
                on="patientid",
                how="left",
            )
            features_df["age"] = features_df["age"].fillna(features_df["age"].median())
            features_df["sex_encoded"] = features_df["sex_encoded"].fillna(-1).astype(int)
            log.info("  Merged age/sex from general_table for test patients")
        else:
            log.warning("general_table.csv not found - test patients will have no demographics")

        test_ids = load_test_ids()
        missing  = set(test_ids) - set(features_df["patientid"].astype(str))
        if missing:
            log.warning(f"{len(missing)} test patients not in CSV (corrupted file tail) - will use mean prediction")
        log.info(f"Test: {len(features_df)} patients x {features_df.shape[1]} features")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(save_path, index=False)
        log.info(f"Saved to {save_path}")

    return features_df


if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.split in ("train", "both"):
        build_features("train", save_path=config.FEATURES_TRAIN)
    if args.split in ("test", "both"):
        build_features("test",  save_path=config.FEATURES_TEST)
