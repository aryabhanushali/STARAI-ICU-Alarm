# data_loader.py
# CSVs are huge (~3GB uncompressed) so we read in chunks instead of all at once.

import json
import logging
import sys
import zlib
from pathlib import Path

import pandas as pd

import config

def setup_logging():
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE, mode="a"),
        ],
    )

log = logging.getLogger(__name__)


def load_train_labels():
    """Load mortality labels + demographics from train_labels.json."""
    log.info("Loading train labels...")
    with open(config.TRAIN_LABELS) as f:
        data = json.load(f)

    df = pd.DataFrame(data["patients"])
    df["patientid"] = df["patientid"].astype(int)
    df["mortality"]  = df["mortality"].astype(int)
    df["age"]        = pd.to_numeric(df["age"], errors="coerce")

    log.info(f"  {len(df)} patients, mortality rate = {df['mortality'].mean()*100:.1f}%")
    return df


def load_test_ids():
    """Load the list of test patient IDs from test.json."""
    with open(config.TEST_IDS) as f:
        data = json.load(f)
    ids = data["patient_ids"]
    assert len(ids) == config.N_TEST, f"Expected {config.N_TEST} test IDs, got {len(ids)}"
    log.info(f"Loaded {len(ids)} test patient IDs")
    return ids


def load_general_table():
    df = pd.read_csv(config.GENERAL_TABLE)
    df["patientid"] = df["patientid"].astype(int)
    return df


def iter_patients(path):
    """Yield one complete patient's DataFrame at a time.

    Keeps a leftover buffer for patients split across chunks.
    Handles corrupted gzip tails gracefully.
    """
    leftover = None

    reader = pd.read_csv(
        path,
        chunksize=config.CHUNK_SIZE,
        dtype={config.PID_COL: str},
        low_memory=False,
    )

    try:
        for chunk in reader:
            chunk = chunk[chunk[config.PID_COL].notna()]
            chunk = chunk[chunk[config.PID_COL].str.strip() != ""]
            if chunk.empty:
                continue

            if leftover is not None and not leftover.empty:
                chunk = pd.concat([leftover, chunk], ignore_index=True)

            last_pid = chunk[config.PID_COL].iloc[-1]

            complete = chunk[chunk[config.PID_COL] != last_pid]
            leftover  = chunk[chunk[config.PID_COL] == last_pid]

            for _, grp in complete.groupby(config.PID_COL, sort=False):
                yield grp

    except (zlib.error, EOFError, OSError) as e:
        log.warning(f"File read error in {Path(path).name} (probably truncated): {e}")
        log.warning("Flushing buffer and stopping - some patients at end of file will be missing")

    if leftover is not None and not leftover.empty:
        for _, grp in leftover.groupby(config.PID_COL, sort=False):
            yield grp


def iter_train_patients():
    return iter_patients(config.TRAIN_CSV)

def iter_test_patients():
    return iter_patients(config.TEST_CSV)


def validate_ids():
    """Check that all patients in train_labels.json exist in the train CSV."""
    log.info("Validating train patient IDs (full scan)...")
    labels_df = load_train_labels()
    label_ids = set(labels_df["patientid"].astype(str).tolist())

    csv_ids = set()
    for grp in iter_train_patients():
        csv_ids.add(grp[config.PID_COL].iloc[0])

    missing = label_ids - csv_ids
    if missing:
        log.warning(f"{len(missing)} label IDs not found in CSV: {list(missing)[:5]}")
    else:
        log.info(f"All {len(label_ids)} train patients found in CSV")


def print_summary(path, n_patients=100):
    frames = []
    for i, grp in enumerate(iter_patients(path)):
        frames.append(grp)
        if i >= n_patients - 1:
            break

    sample = pd.concat(frames, ignore_index=True)
    cols = [c for c in config.FEATURE_COLS if c in sample.columns]
    sub = sample[cols].apply(pd.to_numeric, errors="coerce")

    print(f"\n=== Summary: {Path(path).name} (first {n_patients} patients) ===")
    print(f"Rows: {len(sample):,}  |  Unique patients: {sample[config.PID_COL].nunique()}")
    print(f"{'Variable':<10}  {'Null%':>6}  {'Min':>8}  {'Max':>8}  {'Mean':>8}")
    print("-" * 50)
    for col in cols:
        pct = sub[col].isna().mean() * 100
        vals = sub[col].dropna()
        if len(vals) == 0:
            print(f"{col:<10}  {pct:>6.1f}%  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
        else:
            print(f"{col:<10}  {pct:>6.1f}%  {vals.min():>8.2f}  {vals.max():>8.2f}  {vals.mean():>8.2f}")
    print()


if __name__ == "__main__":
    setup_logging()
    load_train_labels()
    load_test_ids()
    print_summary(config.TRAIN_CSV, config.TRAIN_TIME)
