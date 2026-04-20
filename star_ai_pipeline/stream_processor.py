# stream_processor.py — Stage 1
# Resamples raw minute-level ICU data into 2-min windows and saves parquet.

import logging
import pandas as pd

import config
from data_loader import iter_train_patients, iter_test_patients, setup_logging
from agents.physiological_agent import PhysiologicalAgent

log = logging.getLogger(__name__)

STREAM_TRAIN = config.RESULTS_DIR / "stream_train.parquet"
STREAM_TEST  = config.RESULTS_DIR / "stream_test.parquet"


def process_split(split, rebuild=False):
    out_path = STREAM_TRAIN if split == "train" else STREAM_TEST

    if not rebuild and out_path.exists():
        log.info(f"[Stage 1] Using cached stream_{split}.parquet")
        return out_path

    log.info(f"[Stage 1] Processing {split} split...")
    agent    = PhysiologicalAgent()
    iterator = iter_train_patients() if split == "train" else iter_test_patients()

    resampled_frames = []
    missingness_rows = []
    n = 0

    for patient_df in iterator:
        resampled = agent.load_and_resample(patient_df)
        if resampled.empty:
            continue

        resampled_frames.append(resampled)

        # run missingness on the original df (before resampling) to get true rates
        miss_df = agent.flag_missingness(patient_df)
        missingness_rows.append(miss_df)

        n += 1
        if n % 1000 == 0:
            log.info(f"  [Stage 1] {split}: {n} patients")

    log.info(f"[Stage 1] {split}: {n} patients resampled")

    gen_table = pd.read_csv(config.GENERAL_TABLE)
    gen_table["patientid"] = gen_table["patientid"].astype(str)

    all_resampled = pd.concat(resampled_frames, ignore_index=True)
    all_resampled = agent.merge_demographics(all_resampled, gen_table)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_resampled.to_parquet(out_path, index=False)
    log.info(f"[Stage 1] Saved {out_path}  ({len(all_resampled):,} rows)")

    if missingness_rows:
        miss_combined = pd.concat(missingness_rows, ignore_index=True)
        miss_path = config.RESULTS_DIR / f"missingness_{split}.csv"
        miss_combined.to_csv(miss_path, index=False)
        mean_miss = miss_combined.groupby("variable")["missing_rate"].mean()
        log.info(f"[Stage 1] Top-5 missing variables ({split}):")
        for var, rate in mean_miss.nlargest(5).items():
            log.info(f"  {var}: {rate*100:.1f}% missing")

    return out_path


def run(rebuild=False):
    process_split("train", rebuild=rebuild)
    process_split("test",  rebuild=rebuild)
    log.info("[Stage 1] stream_processor complete.")


if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.split in ("train", "both"):
        process_split("train", rebuild=args.rebuild)
    if args.split in ("test", "both"):
        process_split("test",  rebuild=args.rebuild)
