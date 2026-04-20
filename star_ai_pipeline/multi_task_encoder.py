# multi_task_encoder.py — Stage 2
# Turns each patient's time-series into a single feature row.
# Features are grouped by organ system: circulatory, respiratory, general, late trends.

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
    """Summary stats for one variable over a patient's stay."""
    out     = {}
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
    # largest single-step drop and fastest rate of change for MAP and SpO2
    valid = series.dropna()
    if len(valid) < 2:
        return {f"{name}__max_delta": 0.0, f"{name}__max_rate": 0.0}

    v  = valid.values.astype(float)
    t  = times[series.notna()].values.astype(float)
    dv = np.abs(np.diff(v))
    dt = np.diff(t)
    dt[dt == 0] = 1.0

    return {
        f"{name}__max_delta": float(np.max(dv)),
        f"{name}__max_rate":  float(np.max(dv / dt)),
    }


def _late_slope(series, times, n_windows=180):
    """Slope over the last n_windows 2-min windows (~6 hours)."""
    valid = series.dropna()
    if len(valid) < config.MIN_TREND_OBS:
        return 0.0
    tail_v = valid.values[-n_windows:].astype(float)
    tail_t = times[series.notna()].values[-n_windows:].astype(float)
    if len(tail_v) < config.MIN_TREND_OBS:
        return 0.0
    try:
        slope = float(np.polyfit(tail_t, tail_v, 1)[0])
        return slope if np.isfinite(slope) else 0.0
    except Exception:
        return 0.0


def circulatory_features(df, times):
    feats    = {}
    EARLY_24H = 24 * 3600

    if config.MAP_COL in df.columns:
        map_s = pd.to_numeric(df[config.MAP_COL], errors="coerce")
        map_v = map_s.dropna().values.astype(float)

        feats["map_mean"]       = float(map_v.mean()) if len(map_v) else 0.0
        feats["map_min"]        = float(map_v.min())  if len(map_v) else 0.0
        feats["map_last"]       = float(map_v[-1])    if len(map_v) else 0.0
        feats["map_slope_late"] = _late_slope(map_s, times)

        # min MAP in first 24h
        early_map = map_s[(times <= EARLY_24H).values].dropna()
        feats["early_map_min"] = float(early_map.min()) if len(early_map) else 0.0

        # MAP < 65 mmHg = circulatory failure threshold
        feats["circulatory_failure_flag"] = int((map_v < config.MAP_LOW).any()) if len(map_v) else 0
    else:
        for k in ("map_mean", "map_min", "map_last", "map_slope_late",
                  "early_map_min", "circulatory_failure_flag"):
            feats[k] = 0.0

    vaso_cols = [c for c in config.VASOPRESSOR_COLS if c in df.columns]
    if vaso_cols:
        vaso_df     = df[vaso_cols].apply(pd.to_numeric, errors="coerce")
        vaso_on     = vaso_df.gt(0).any(axis=1)
        row_interval = max(
            (times.max() - times.min()) / max(len(times) - 1, 1) / 60.0, 1.0
        )
        feats["vasopressor_flag"]         = int(vaso_on.any())
        feats["vasopressor_duration_hrs"] = round(int(vaso_on.sum()) * row_interval / 60.0, 2)
    else:
        feats["vasopressor_flag"]         = 0
        feats["vasopressor_duration_hrs"] = 0.0

    return feats


def respiratory_features(df, times):
    feats     = {}
    EARLY_24H = 24 * 3600

    if config.SPO2_COL in df.columns:
        spo2_s = pd.to_numeric(df[config.SPO2_COL], errors="coerce")
        spo2_v = spo2_s.dropna().values.astype(float)

        feats["spo2_mean"]       = float(spo2_v.mean()) if len(spo2_v) else 0.0
        feats["spo2_min"]        = float(spo2_v.min())  if len(spo2_v) else 0.0
        feats["spo2_last"]       = float(spo2_v[-1])    if len(spo2_v) else 0.0
        feats["spo2_slope_late"] = _late_slope(spo2_s, times)

        early_spo2 = spo2_s[(times <= EARLY_24H).values].dropna()
        feats["early_spo2_min"] = float(early_spo2.min()) if len(early_spo2) else 0.0

        # SpO2 < 90% = respiratory failure threshold
        feats["respiratory_failure_flag"] = int((spo2_v < config.SPO2_LOW).any()) if len(spo2_v) else 0
    else:
        for k in ("spo2_mean", "spo2_min", "spo2_last", "spo2_slope_late",
                  "early_spo2_min", "respiratory_failure_flag"):
            feats[k] = 0.0

    if config.MV_COL in df.columns:
        mv_s = pd.to_numeric(df[config.MV_COL], errors="coerce")
        mv_on = mv_s.gt(0)
        row_interval = max(
            (times.max() - times.min()) / max(len(times) - 1, 1) / 60.0, 1.0
        )
        feats["ventilation_flag"]         = int(mv_on.any())
        feats["ventilation_duration_hrs"] = round(int(mv_on.sum()) * row_interval / 60.0, 2)
    else:
        feats["ventilation_flag"]         = 0
        feats["ventilation_duration_hrs"] = 0.0

    return feats


def general_features(df, times):
    feats = {}

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
            feats.update(delta_features(pd.to_numeric(df[col], errors="coerce"), times, col))
        else:
            feats[f"{col}__max_delta"] = 0.0
            feats[f"{col}__max_rate"]  = 0.0

    ts     = pd.to_datetime(df[config.TRAIN_TIME], errors="coerce").dropna()
    icu_h  = max(0.0, (ts.max() - ts.min()).total_seconds() / 3600.0) if not ts.empty else 0.0
    feats["icu_duration_hours"] = icu_h
    feats["n_observations"]     = len(df)

    # early deterioration: did MAP or SpO2 cross their thresholds in the first 24h?
    FIRST_24H  = 24 * 3600
    df_early   = df[(times <= FIRST_24H).values]

    if not df_early.empty:
        map_e  = pd.to_numeric(df_early.get(config.MAP_COL,  pd.Series(dtype=float)), errors="coerce").dropna()
        spo2_e = pd.to_numeric(df_early.get(config.SPO2_COL, pd.Series(dtype=float)), errors="coerce").dropna()
        feats["early_deterioration_flag"] = int(
            (len(map_e)  > 0 and map_e.min()  < config.MAP_LOW) or
            (len(spo2_e) > 0 and spo2_e.min() < config.SPO2_LOW)
        )
        hr_e = pd.to_numeric(df_early.get("vm1", pd.Series(dtype=float)), errors="coerce").dropna()
        feats["early_hr_max"] = float(hr_e.max()) if len(hr_e) else 0.0
    else:
        feats["early_deterioration_flag"] = 0
        feats["early_hr_max"]             = 0.0

    # persistent flags (2 consecutive low readings)
    for col, flag_low, flag_persist in [
        (config.MAP_COL,  "flag_map_low",  "flag_map_persistent"),
        (config.SPO2_COL, "flag_spo2_low", "flag_spo2_persistent"),
    ]:
        threshold = config.MAP_LOW if col == config.MAP_COL else config.SPO2_LOW
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            feats[flag_low]     = int((v < threshold).any())
            feats[flag_persist] = int(((v < threshold).astype(int).rolling(2).sum() >= 2).any())
        else:
            feats[flag_low] = feats[flag_persist] = 0

    vaso_cols = [c for c in config.VASOPRESSOR_COLS if c in df.columns]
    feats["flag_vasopressor"] = int(
        df[vaso_cols].apply(pd.to_numeric, errors="coerce").gt(0).any(axis=None)
    ) if vaso_cols else 0

    if config.MV_COL in df.columns:
        feats["flag_mech_vent"] = int((pd.to_numeric(df[config.MV_COL], errors="coerce") > 0).any())
    else:
        feats["flag_mech_vent"] = 0

    return feats


def late_trend_features(df, times, top_cols=None):
    """Slopes over the last 6 hours for the 5 most important variables."""
    DEFAULT_TOP = ["vm5", "vm20", "vm172", "vm1", "vm174"]

    if top_cols is None:
        imp_path = config.RESULTS_DIR / "feature_importance.csv"
        if imp_path.exists():
            imp_df    = pd.read_csv(imp_path)
            base_vars = []
            for col in config.FEATURE_COLS:
                mask = imp_df["feature"].str.startswith(col + "__")
                if mask.any():
                    base_vars.append((col, imp_df.loc[mask, "importance"].sum()))
            base_vars.sort(key=lambda x: x[1], reverse=True)
            top_cols = [v for v, _ in base_vars[:5]] if base_vars else DEFAULT_TOP
        else:
            top_cols = DEFAULT_TOP

    feats = {}
    for col in top_cols:
        if col in df.columns:
            feats[f"{col}__late_slope"] = _late_slope(
                pd.to_numeric(df[col], errors="coerce"), times
            )
        else:
            feats[f"{col}__late_slope"] = 0.0
    return feats


def patient_features(df, time_col):
    pid  = df[config.PID_COL].iloc[0]
    row  = {config.PID_COL: pid}

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    times  = (parsed - parsed.min()).dt.total_seconds().fillna(0.0)

    row.update(general_features(df, times))
    row.update(circulatory_features(df, times))
    row.update(respiratory_features(df, times))
    row.update(late_trend_features(df, times))

    return row


def build_features(split, save_path=None):
    assert split in ("train", "test")
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
    assert len(features_df) > 0

    if split == "train":
        labels = load_train_labels()
        labels["patientid"]   = labels["patientid"].astype(str)
        features_df["patientid"] = features_df["patientid"].astype(str)
        features_df = features_df.merge(
            labels[["patientid", "age", "sex", "mortality"]],
            on="patientid", how="left",
        )
        features_df["sex_encoded"] = features_df["sex"].map({"M": 1, "F": 0}).fillna(-1).astype(int)
        features_df = features_df.drop(columns=["sex"])
        features_df["age"] = pd.to_numeric(features_df["age"], errors="coerce")
        assert features_df["mortality"].notna().all()
        log.info(f"Train: {len(features_df)} patients x {features_df.shape[1]} features")

    else:
        if config.GENERAL_TABLE.exists():
            gen = pd.read_csv(config.GENERAL_TABLE)
            gen["patientid"]   = gen["patientid"].astype(str)
            gen["sex_encoded"] = gen["sex"].map({"M": 1, "F": 0}).fillna(-1).astype(int)
            gen["age"]         = pd.to_numeric(gen["age"], errors="coerce")
            features_df["patientid"] = features_df["patientid"].astype(str)
            features_df = features_df.merge(
                gen[["patientid", "age", "sex_encoded"]], on="patientid", how="left"
            )
            features_df["age"]         = features_df["age"].fillna(features_df["age"].median())
            features_df["sex_encoded"] = features_df["sex_encoded"].fillna(-1).astype(int)
            log.info("  Merged age/sex from general_table for test patients")

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


def run(rebuild=False):
    if not rebuild and config.FEATURES_TRAIN.exists() and config.FEATURES_TEST.exists():
        log.info("[Stage 2] Feature parquets exist — skipping (use --rebuild-features to force)")
        return
    build_features("train", save_path=config.FEATURES_TRAIN)
    build_features("test",  save_path=config.FEATURES_TEST)


if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   choices=["train", "test", "both"], default="both")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.split in ("train", "both"):
        build_features("train", save_path=config.FEATURES_TRAIN)
    if args.split in ("test", "both"):
        build_features("test",  save_path=config.FEATURES_TEST)
