# run_pipeline.py
# Runs all 6 pipeline stages in order.

import argparse
import logging
import time

import config
from data_loader import load_train_labels, load_test_ids, setup_logging
from train import get_train_features, prep_data, run_cv, set_seeds
from predict import get_test_features, predict

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="HiRID ICU mortality pipeline (multi-agent)")
    parser.add_argument("--skip-validation",  action="store_true",
                        help="skip ID scan validation (much faster)")
    parser.add_argument("--rebuild-features", action="store_true",
                        help="re-extract features even if parquet cache exists")
    parser.add_argument("--rebuild-stream",   action="store_true",
                        help="re-run Stage 1 resampling even if cache exists")
    parser.add_argument("--skip-stream",      action="store_true",
                        help="skip Stage 1 resampling (use raw data directly)")
    parser.add_argument("--with-analysis",    action="store_true",
                        help="run analysis plots after training")
    args = parser.parse_args()

    setup_logging()
    set_seeds()

    log.info("=" * 60)
    log.info("STAR AI Track 4 — HiRID ICU Mortality (Multi-Agent Pipeline)")
    log.info("=" * 60)
    t_start = time.perf_counter()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True,  exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True,    exist_ok=True)

    # Stage 1: Stream Processor
    log.info("[1/6] Stage 1: stream_processor (resample + missingness)...")
    if not args.skip_stream:
        from stream_processor import run as run_stream
        run_stream(rebuild=args.rebuild_stream)
    else:
        log.info("  (skipping Stage 1 — using raw data)")

    # Data validation
    log.info("[2/6] Loading labels and IDs...")
    load_train_labels()
    load_test_ids()

    if not args.skip_validation:
        from data_loader import validate_ids
        validate_ids()
    else:
        log.info("  (skipping ID validation)")

    # Stage 2: Multi-Task Encoder
    log.info("[3/6] Stage 2: multi_task_encoder (organ-grouped features)...")
    train_df = get_train_features(rebuild=args.rebuild_features)
    get_test_features(rebuild=args.rebuild_features)

    # Stage 3: Train + Predict
    log.info("[4/6] Stage 3: training 5-fold CV + logistic regression blend...")
    X, y, feat_cols = prep_data(train_df)
    results = run_cv(X, y, feat_cols)
    predict(rebuild_features=False)

    # Stage 4: Uncertainty Module
    log.info("[5/6] Stage 4: uncertainty_module (epistemic uncertainty)...")
    try:
        from uncertainty_module import run as run_uncertainty
        run_uncertainty()
    except Exception as e:
        log.warning(f"Stage 4 uncertainty_module failed (non-critical): {e}")

    # Stage 5: Alarm Policy Engine
    log.info("[6a/6] Stage 5: alarm_policy_engine (methodology report)...")
    try:
        from alarm_policy_engine import run as run_alarm
        run_alarm()
    except Exception as e:
        log.warning(f"Stage 5 alarm_policy_engine failed (non-critical): {e}")

    # Stage 6: Equity Dashboard
    log.info("[6b/6] Stage 6: equity_dashboard (subgroup fairness report)...")
    try:
        from equity_dashboard import run as run_equity
        run_equity()
    except Exception as e:
        log.warning(f"Stage 6 equity_dashboard failed (non-critical): {e}")

    if args.with_analysis:
        log.info("[+] Running analysis plots...")
        from analysis import plot_roc, plot_importance, alarm_burden_table, subgroup_auc
        plot_roc()
        plot_importance()
        alarm_burden_table()
        subgroup_auc()

    elapsed = time.perf_counter() - t_start
    log.info("=" * 60)
    log.info(f"Pipeline complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"  CV AUC  : {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
    log.info(f"  OOF AUC : {results['oof_auc']:.4f}")
    log.info(f"  Output  : {config.SUBMISSION_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
