"""
modeling.py - shared harness for comparing classifiers on the Adult Census data.

Project rules this file enforces
  * Only TRAIN rows are loaded. The test rows are never read into memory.
  * Every learned step (scaler, one-hot categories, model weights) is fitted
    inside each CV fold, on that fold's training part only: the whole model is a
    Pipeline(preprocessor + classifier) that is cloned and refitted per fold.
  * Results go to results/model_results.csv. An experiment_id that already
    exists is skipped, never overwritten.
  * Out-of-fold (OOF) predictions are saved to results/oof/<experiment_id>.csv
    so thresholds and ensembles can be studied later without retraining.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import preprocessing as prep

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "adult.csv"
RESULTS_DIR = ROOT / "results"
SPLIT_PATH = RESULTS_DIR / "split_indices.csv"
DROPPED_PATH = RESULTS_DIR / "dropped_duplicates.csv"
FOLDS_PATH = RESULTS_DIR / "cv_folds.csv"
RESULTS_PATH = RESULTS_DIR / "model_results.csv"
OOF_DIR = RESULTS_DIR / "oof"

DATA_VERSION = "v2_dedup"
POSITIVE_LABEL = ">50K"
SEX_GROUPS = ["Female", "Male"]

METRIC_NAMES = ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1"]


# --------------------------------------------------------------------------
# Loading (train rows only)
# --------------------------------------------------------------------------
@dataclass
class TrainData:
    """Everything a model run needs. All arrays are aligned to the same rows."""

    frame: pd.DataFrame      # raw feature columns of the TRAIN rows only
    y: np.ndarray            # 1 = ">50K", 0 = "<=50K"
    sex: np.ndarray          # "Female" / "Male", used only for reporting by group
    fold: np.ndarray         # fold number of every train row (one fold column)
    row_id: np.ndarray       # position of each row in data/adult.csv
    fold_col: str


def load_train(fold_col: str = "fold_r0") -> TrainData:
    for path in [DATA_PATH, SPLIT_PATH, DROPPED_PATH, FOLDS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")

    df = pd.read_csv(DATA_PATH, header=None, names=prep.RAW_COLUMNS, keep_default_na=False)
    if df.shape != (48842, 15):
        raise ValueError(f"Unexpected shape of data/adult.csv: {df.shape}")

    split_df = pd.read_csv(SPLIT_PATH)
    if list(split_df.columns) != ["row_id", "split"] or not split_df["row_id"].is_unique:
        raise ValueError("split_indices.csv does not have the expected layout")
    if set(split_df["split"].unique()) != {"train", "test"}:
        raise ValueError("split_indices.csv: 'split' must be train/test")

    dropped_df = pd.read_csv(DROPPED_PATH)
    expected_ids = set(range(len(df))) - set(dropped_df["row_id"])
    if set(split_df["row_id"]) != expected_ids or len(split_df) != 48790:
        raise ValueError("Split row_ids are not all positions minus the dropped duplicates")

    train_ids = np.sort(split_df.loc[split_df["split"] == "train", "row_id"].to_numpy())

    folds = pd.read_csv(FOLDS_PATH)
    if fold_col not in folds.columns:
        raise ValueError(f"{fold_col} not in cv_folds.csv (columns: {list(folds.columns)})")
    if not np.array_equal(folds["row_id"].to_numpy(), train_ids):
        raise ValueError("cv_folds.csv row_ids differ from the train row_ids")

    frame = df.iloc[train_ids].reset_index(drop=True)
    del df  # the full table (with the test rows) is not kept

    income = frame["income"].astype(str).str.strip()
    if set(income.unique()) != {"<=50K", POSITIVE_LABEL}:
        raise ValueError(f"Unexpected income values: {sorted(income.unique())}")
    y = (income == POSITIVE_LABEL).astype(int).to_numpy()

    sex = frame["sex"].astype(str).str.strip().to_numpy()
    if set(sex) != set(SEX_GROUPS):
        raise ValueError(f"Unexpected sex values: {sorted(set(sex))}")

    return TrainData(frame=frame, y=y, sex=sex, fold=folds[fold_col].to_numpy(),
                     row_id=train_ids, fold_col=fold_col)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metric_block(y, score, pred) -> dict:
    """Ranking metrics use `score`; label metrics use the model's own `pred`."""
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _score(model, X) -> np.ndarray:
    """Probability of >50K if the model has one, otherwise its decision score."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


# --------------------------------------------------------------------------
# One model, cross-validated
# --------------------------------------------------------------------------
def cross_validate(spec, data: TrainData) -> tuple[dict, pd.DataFrame]:
    n = len(data.y)
    oof_score = np.full(n, np.nan)
    oof_pred = np.full(n, -1, dtype=int)
    fold_pr_auc, fit_seconds, warning_messages = [], 0.0, []
    n_convergence = 0

    for k in sorted(np.unique(data.fold)):
        is_val = data.fold == k
        fit_part, val_part = data.frame[~is_val], data.frame[is_val]
        y_fit, y_val = data.y[~is_val], data.y[is_val]

        model = clone(spec.build())        # a fresh, unfitted Pipeline every fold
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(fit_part, y_fit)     # scaler, encoder and weights learn from fit_part only
            score = _score(model, val_part)
            pred = np.asarray(model.predict(val_part)).astype(int)
        seconds = time.perf_counter() - t0
        fit_seconds += seconds

        if k == sorted(np.unique(data.fold))[0]:   # one-off guard on the first fold
            names = list(model.named_steps["prep"].get_feature_names_out())
            if any("income" in name for name in names):
                raise AssertionError("The target leaked into the feature columns")

        for w in caught:
            if "Convergence" in w.category.__name__:
                n_convergence += 1
            else:
                warning_messages.append(f"{w.category.__name__}: {str(w.message)[:120]}")

        oof_score[is_val], oof_pred[is_val] = score, pred
        fold_pr_auc.append(average_precision_score(y_val, score))
        print(f"    fold {k}: {seconds:6.1f}s   PR-AUC {fold_pr_auc[-1]:.4f}", flush=True)

    if np.isnan(oof_score).any() or (oof_pred < 0).any():
        raise AssertionError("Some rows never got an out-of-fold prediction")

    row = {
        "pr_auc_mean": float(np.mean(fold_pr_auc)),
        "pr_auc_std": float(np.std(fold_pr_auc)),
        "fold_pr_auc": json.dumps([round(float(v), 6) for v in fold_pr_auc]),
        "fit_seconds_total": round(fit_seconds, 2),
        "convergence_warnings": n_convergence,
        "other_warnings": " | ".join(sorted(set(warning_messages)))[:300],
    }
    pooled = metric_block(data.y, oof_score, oof_pred)
    row["pr_auc_pooled"] = pooled.pop("pr_auc")
    row.update(pooled)

    row["floor_all"] = float(data.y.mean())
    for group in SEX_GROUPS:
        mask = data.sex == group
        block = metric_block(data.y[mask], oof_score[mask], oof_pred[mask])
        row[f"floor_{group.lower()}"] = float(data.y[mask].mean())
        for name in ["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]:
            row[f"{name}_{group.lower()}"] = block[name]

    oof = pd.DataFrame({"row_id": data.row_id, "y_true": data.y,
                        "score": oof_score, "pred": oof_pred})
    return row, oof


# --------------------------------------------------------------------------
# Results file (append-only, never overwrites)
# --------------------------------------------------------------------------
RESULT_COLUMNS = (
    ["experiment_id", "model", "data_version", "fold_col", "n_rows", "timestamp",
     "sklearn_version", "params",
     "pr_auc_mean", "pr_auc_std", "fold_pr_auc", "pr_auc_pooled", "roc_auc",
     "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp",
     "floor_all"]
    + [f"{name}_{g}" for g in ["female", "male"]
       for name in ["floor", "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"]]
    + ["fit_seconds_total", "convergence_warnings", "other_warnings"]
)


def existing_ids() -> set:
    if not RESULTS_PATH.exists():
        return set()
    return set(pd.read_csv(RESULTS_PATH, usecols=["experiment_id"])["experiment_id"])


def append_result(row: dict) -> None:
    missing = [c for c in RESULT_COLUMNS if c not in row]
    extra = [c for c in row if c not in RESULT_COLUMNS]
    if missing or extra:
        raise ValueError(f"Result row does not match the schema. missing={missing} extra={extra}")
    new = pd.DataFrame([row])[RESULT_COLUMNS]
    if RESULTS_PATH.exists():
        header = list(pd.read_csv(RESULTS_PATH, nrows=0).columns)
        if header != RESULT_COLUMNS:
            raise ValueError("model_results.csv has a different header than this code expects; "
                             "move the old file aside instead of mixing layouts")
        new.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        new.to_csv(RESULTS_PATH, index=False)


def run_one(spec, data: TrainData) -> bool:
    """Run one model. Returns True if it ran, False if it was already logged."""
    suffix = data.fold_col.replace("fold_", "")
    experiment_id = f"{spec.key}_default_{suffix}"
    if experiment_id in existing_ids():
        print(f"[skip] {experiment_id} is already in results/model_results.csv", flush=True)
        return False

    print(f"[run ] {experiment_id}  ({spec.title})", flush=True)
    row, oof = cross_validate(spec, data)

    clf = spec.build().named_steps["clf"]
    row.update({
        "experiment_id": experiment_id,
        "model": spec.title,
        "data_version": DATA_VERSION,
        "fold_col": data.fold_col,
        "n_rows": int(len(data.y)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "params": json.dumps(clf.get_params(), default=str, sort_keys=True),
    })

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = OOF_DIR / f"{experiment_id}.csv"
    if oof_path.exists():
        raise FileExistsError(f"{oof_path} exists but the id is not in the results file; "
                              "resolve this by hand instead of overwriting")
    oof.to_csv(oof_path, index=False)
    append_result(row)
    print(f"[done] {experiment_id}: PR-AUC pooled {row['pr_auc_pooled']:.4f}, "
          f"F1 {row['f1']:.4f}, {row['fit_seconds_total']:.0f}s", flush=True)
    return True


def comparison_table() -> pd.DataFrame:
    cols = ["experiment_id", "pr_auc_mean", "pr_auc_std", "pr_auc_pooled", "roc_auc",
            "accuracy", "precision", "recall", "f1", "pr_auc_female", "pr_auc_male",
            "f1_female", "f1_male", "fit_seconds_total"]
    table = pd.read_csv(RESULTS_PATH)[cols]
    return table.sort_values("pr_auc_pooled", ascending=False).set_index("experiment_id")
