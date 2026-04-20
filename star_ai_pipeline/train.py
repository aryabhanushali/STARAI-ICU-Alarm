# train.py
# 5-fold stratified CV with LightGBM, plus a logistic regression baseline

import json
import logging
import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from data_loader import setup_logging
from multi_task_encoder import build_features

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
log = logging.getLogger(__name__)


def set_seeds(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_train_features(rebuild=False):
    if not rebuild and config.FEATURES_TRAIN.exists():
        log.info("Loading cached train features...")
        return pd.read_parquet(config.FEATURES_TRAIN)
    log.info("Building train features from scratch...")
    return build_features("train", save_path=config.FEATURES_TRAIN)


def prep_data(df):
    drop = ["patientid", "mortality", "discharge_status"]
    feat_cols = [c for c in df.columns if c not in drop]

    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["mortality"].astype(int)

    assert X.isnull().sum().sum() == 0, "NaNs still in feature matrix"
    assert y.isin([0, 1]).all(), "Labels contain non-binary values"

    imp_path = config.RESULTS_DIR / "feature_importance.csv"
    if imp_path.exists():
        imp_df      = pd.read_csv(imp_path)
        zero_feats  = set(imp_df.loc[imp_df["importance"] == 0, "feature"].tolist())
        to_drop     = [c for c in zero_feats if c in feat_cols]
        if to_drop:
            log.info(f"Dropping {len(to_drop)} zero-importance features: {to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")
            feat_cols = [c for c in feat_cols if c not in zero_feats]
            X = X[feat_cols]
        else:
            log.info("No zero-importance features found to drop")

    log.info(f"X shape: {X.shape}  |  mortality rate: {y.mean()*100:.1f}%")
    return X, y, feat_cols


def train_lgbm_fold(X_tr, y_tr, X_val, y_val, fold_num):
    import lightgbm as lgb

    # dataset is ~94% negative so upweight positives
    pos_weight = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    log.info(f"  Fold {fold_num} | scale_pos_weight={pos_weight:.2f}")

    lgb_train = lgb.Dataset(X_tr, label=y_tr)
    lgb_val   = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        "objective":        "binary",
        "metric":           "auc",
        "num_leaves":       config.N_LEAVES,
        "learning_rate":    config.LR,
        "feature_fraction": config.COLSAMPLE_TREE,
        "bagging_fraction": config.SUBSAMPLE,
        "bagging_freq":     config.BAGGING_FREQ,
        "scale_pos_weight": pos_weight,
        "verbose":          -1,
        "seed":             config.SEED,
        "n_jobs":           -1,
    }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.N_TREES,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(config.EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds   = model.predict(X_val)
    val_auc = roc_auc_score(y_val, preds)
    log.info(f"  Fold {fold_num} | best_iter={model.best_iteration} | AUC={val_auc:.4f}")

    return model, val_auc


def train_logreg_baseline(X, y, cv):
    # sanity check - if logreg beats lgbm something is wrong
    log.info("Training logistic regression baseline...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="lbfgs", C=0.1, random_state=config.SEED
        )),
    ])
    fold_aucs = []
    for _, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = pipe.predict_proba(X.iloc[val_idx])[:, 1]
        auc   = roc_auc_score(y.iloc[val_idx], preds)
        fold_aucs.append(auc)
    mean_auc = float(np.mean(fold_aucs))
    log.info(f"LogReg baseline: {mean_auc:.4f} ± {float(np.std(fold_aucs)):.4f}")
    return mean_auc


def run_cv(X, y, feat_cols):
    set_seeds()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)

    fold_aucs = []
    models    = []
    oof_preds = np.zeros(len(y))

    log.info(f"Starting {config.N_FOLDS}-fold CV...")

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr,  y_tr  = X.iloc[tr_idx],  y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model, val_auc = train_lgbm_fold(X_tr, y_tr, X_val, y_val, fold + 1)

        fold_aucs.append(val_auc)
        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)

        joblib.dump(model, config.MODELS_DIR / f"lgbm_fold{fold+1}.pkl")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    oof_auc  = roc_auc_score(y, oof_preds)

    print("\n" + "="*45)
    print("CROSS-VALIDATION RESULTS")
    print("="*45)
    for i, auc in enumerate(fold_aucs):
        print(f"  Fold {i+1}: {auc:.4f}")
    print(f"  Mean ± Std : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  OOF AUC    : {oof_auc:.4f}")
    print("="*45 + "\n")

    pd.DataFrame({"oof_pred": oof_preds, "mortality": y.values}).to_csv(
        config.RESULTS_DIR / "oof_predictions.csv", index=False
    )

    # average feature importance across folds
    importances = np.mean([m.feature_importance("gain") for m in models], axis=0)
    pd.DataFrame({"feature": feat_cols, "importance": importances}) \
      .sort_values("importance", ascending=False) \
      .to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)

    with open(config.RESULTS_DIR / "fold_aucs.json", "w") as f:
        json.dump({"fold_aucs": fold_aucs}, f, indent=2)
    log.info("Saved fold AUCs to fold_aucs.json")

    # save training mortality rate as fallback for missing test patients
    train_base_rate = float(y.mean())
    with open(config.RESULTS_DIR / "train_base_rate.txt", "w") as f:
        f.write(str(train_base_rate))
    log.info(f"Training mortality base rate: {train_base_rate:.4f}")

    train_logreg_baseline(X, y, cv)

    log.info("Training full-data logistic regression for blend...")
    full_logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000, class_weight="balanced",
            solver="lbfgs", C=0.1, random_state=config.SEED
        )),
    ])
    full_logreg.fit(X, y)
    logreg_path = config.MODELS_DIR / "logreg_full.pkl"
    joblib.dump(full_logreg, logreg_path)
    log.info(f"Saved full-data logistic regression to {logreg_path}")

    # OOF logreg predictions for blend ratio selection
    logreg_oof = np.zeros(len(y))
    for _, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        tmp = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                max_iter=1000, class_weight="balanced",
                solver="lbfgs", C=0.1, random_state=config.SEED
            )),
        ])
        tmp.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        logreg_oof[val_idx] = tmp.predict_proba(X.iloc[val_idx])[:, 1]

    logreg_oof_auc = roc_auc_score(y, logreg_oof)
    log.info(f"Logistic regression OOF AUC: {logreg_oof_auc:.4f}")

    pd.DataFrame({"logreg_oof_pred": logreg_oof}).to_csv(
        config.RESULTS_DIR / "logreg_oof_predictions.csv", index=False
    )

    return {"fold_aucs": fold_aucs, "mean_auc": mean_auc, "std_auc": std_auc,
            "oof_auc": oof_auc, "models": models, "oof_preds": oof_preds,
            "logreg_oof_preds": logreg_oof, "feat_cols": feat_cols}


if __name__ == "__main__":
    import argparse
    setup_logging()
    set_seeds()

    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    df = get_train_features(rebuild=args.rebuild_features)
    X, y, feat_cols = prep_data(df)
    run_cv(X, y, feat_cols)
