# analysis.py
# Generates plots and tables for the methodology report.
# Run after training — needs oof_predictions.csv and feature_importance.csv.

import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so it works without a display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import config
from data_loader import setup_logging

log = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.1)


def save_fig(fig, name):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {path}")
    return path


def plot_roc():
    oof_path = config.RESULTS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        log.warning("oof_predictions.csv not found - run train.py first")
        return

    df      = pd.read_csv(oof_path)
    y_true  = df["mortality"].values
    y_score = df["oof_pred"].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"OOF AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Out-of-Fold, LightGBM)")
    ax.legend(loc="lower right")
    save_fig(fig, "roc_curve.png")


def plot_importance(top_n=30):
    imp_path = config.RESULTS_DIR / "feature_importance.csv"
    if not imp_path.exists():
        log.warning("feature_importance.csv not found")
        return

    df = pd.read_csv(imp_path).nlargest(top_n, "importance")
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))
    sns.barplot(data=df, y="feature", x="importance", ax=ax, orient="h")
    ax.set_title(f"Top {top_n} Features (LightGBM gain, avg across folds)")
    ax.set_xlabel("Mean Gain")
    ax.set_ylabel("")
    save_fig(fig, "feature_importance.png")


def alarm_burden_table():
    oof_path = config.RESULTS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        log.warning("oof_predictions.csv not found")
        return pd.DataFrame()

    df      = pd.read_csv(oof_path)
    y_true  = df["mortality"].values
    y_score = df["oof_pred"].values

    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold":   round(float(t), 2),
            "sensitivity": round(tp / (tp + fn + 1e-9), 4),
            "specificity": round(tn / (tn + fp + 1e-9), 4),
            "FAR":         round(fp / (fp + tn + 1e-9), 4),
            "PPV":         round(tp / (tp + fp + 1e-9), 4),
            "F1":          round(2*tp / (2*tp + fp + fn + 1e-9), 4),
        })

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "alarm_burden.csv", index=False)

    print("\n=== Alarm Burden Table ===")
    print(table.to_string(index=False))
    print()
    return table


def subgroup_auc():
    oof_path   = config.RESULTS_DIR / "oof_predictions.csv"
    feat_path  = config.FEATURES_TRAIN
    if not oof_path.exists() or not feat_path.exists():
        log.warning("Missing files for subgroup analysis")
        return

    oof  = pd.read_csv(oof_path).reset_index(drop=True)
    feat = pd.read_parquet(feat_path).reset_index(drop=True)
    assert len(oof) == len(feat), "OOF and feature matrix length mismatch"

    merged = pd.concat([feat[["patientid", "age", "sex_encoded"]], oof], axis=1)

    print("\n=== Subgroup AUC Analysis ===")

    # by age group
    merged["age_group"] = pd.cut(
        pd.to_numeric(merged["age"], errors="coerce"),
        bins=[0, 50, 70, 999], labels=["<50", "50-70", ">70"], right=False
    )
    print("\nBy age group:")
    for grp in ["<50", "50-70", ">70"]:
        sub = merged[merged["age_group"] == grp]
        if len(sub) < 20 or sub["mortality"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["mortality"], sub["oof_pred"])
        print(f"  {grp:>6}: AUC={auc:.4f}  (n={len(sub)}, deaths={int(sub['mortality'].sum())})")

    # by sex
    print("\nBy sex:")
    for code, label in [(1, "Male"), (0, "Female"), (-1, "Unknown")]:
        sub = merged[merged["sex_encoded"] == code]
        if len(sub) < 20 or sub["mortality"].nunique() < 2:
            continue
        auc = roc_auc_score(sub["mortality"], sub["oof_pred"])
        print(f"  {label:>8}: AUC={auc:.4f}  (n={len(sub)}, deaths={int(sub['mortality'].sum())})")
    print()


def early_warning(hours_list=None):
    """AUC vs first N hours of data."""
    import joblib
    from features import patient_features
    from data_loader import iter_train_patients, load_train_labels

    if hours_list is None:
        hours_list = [1, 2, 4, 8, 12, 24]

    model_paths = sorted(config.MODELS_DIR.glob("lgbm_fold*.pkl"))
    if not model_paths:
        log.warning("No models found - run train.py first")
        return

    models    = [joblib.load(p) for p in model_paths]
    labels_df = load_train_labels()
    label_map = dict(zip(labels_df["patientid"].astype(str), labels_df["mortality"]))
    aucs      = {}

    for max_hours in hours_list:
        max_sec = max_hours * 3600.0
        log.info(f"Early warning: first {max_hours}h...")
        rows = []
        for df in iter_train_patients():
            pid = df["patientid"].iloc[0]
            ts  = pd.to_numeric(df[config.TRAIN_TIME], errors="coerce")
            truncated = df[ts <= max_sec]
            if len(truncated) == 0:
                continue
            feats = patient_features(truncated, config.TRAIN_TIME)
            feats["mortality"] = label_map.get(str(pid), np.nan)
            rows.append(feats)

        sub_df = pd.DataFrame(rows).dropna(subset=["mortality"])
        if sub_df.empty or sub_df["mortality"].nunique() < 2:
            continue

        labels_str = labels_df.copy()
        labels_str["patientid"] = labels_str["patientid"].astype(str)
        sub_df["patientid"] = sub_df["patientid"].astype(str)
        sub_df = sub_df.merge(labels_str[["patientid", "age", "sex"]], on="patientid", how="left")
        sub_df["sex_encoded"] = sub_df["sex"].map({"M": 1, "F": 0}).fillna(-1).astype(int)
        sub_df["age"] = pd.to_numeric(sub_df["age"], errors="coerce")
        sub_df = sub_df.drop(columns=["sex"], errors="ignore")

        drop_cols = {"patientid", "mortality", "discharge_status"}
        feat_cols = [c for c in sub_df.columns if c not in drop_cols]
        X = sub_df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = sub_df["mortality"].astype(int)

        preds = np.zeros(len(X))
        n_ok  = 0
        for m in models:
            try:
                m_cols = m.feature_name()
                for c in m_cols:
                    if c not in X.columns:
                        X[c] = 0.0
                preds += m.predict(X[m_cols])
                n_ok  += 1
            except Exception as e:
                log.warning(f"Model predict failed for {max_hours}h: {e}")
        if n_ok == 0:
            continue
        preds /= n_ok

        auc = roc_auc_score(y, preds)
        aucs[max_hours] = auc
        log.info(f"  {max_hours}h -> AUC={auc:.4f}  (n={len(sub_df)})")

    if not aucs:
        return

    xs = sorted(aucs.keys())
    ys = [aucs[h] for h in xs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "o-", lw=2, ms=8)
    ax.set_xlabel("First N hours of ICU data")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Early Warning: Performance vs Data Horizon")
    ax.set_xticks(xs)
    ax.set_ylim(0.5, 1.0)
    for h, a in zip(xs, ys):
        ax.annotate(f"{a:.3f}", (h, a), textcoords="offset points", xytext=(0, 8), ha="center")
    save_fig(fig, "early_warning.png")

    print("\n=== Early Warning AUC ===")
    for h, a in zip(xs, ys):
        print(f"  {h:>3}h -> {a:.4f}")
    print()


if __name__ == "__main__":
    import argparse
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", nargs="+",
                        choices=["roc", "imp", "alarm", "subgroup", "early"],
                        default=["roc", "imp", "alarm", "subgroup"])
    args = parser.parse_args()

    if "roc"      in args.plots: plot_roc()
    if "imp"      in args.plots: plot_importance()
    if "alarm"    in args.plots: alarm_burden_table()
    if "subgroup" in args.plots: subgroup_auc()
    if "early"    in args.plots: early_warning()
