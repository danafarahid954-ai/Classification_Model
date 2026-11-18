"""
tune_finalists.py - light hyperparameter tuning for the finalists, using the same
harness, folds and preprocessor as run_models.py.

What it does, per model (default: hist_gb and random_forest)
  1. Tries every combination in a small fixed grid (see GRIDS below) with 5-fold CV
     on fold_r0, inside the TRAIN rows only. Every combination is written to
     results/tuning_grid.csv as soon as it finishes, so an interrupted run resumes
     where it stopped.
  2. Picks the combination with the highest per-fold mean PR-AUC.
  3. Logs that winner to results/model_results.csv as "<model>_tuned_r0" (with its
     out-of-fold predictions in results/oof/).
  4. Re-runs the winner on the other saved fold sets (fold_r1, fold_r2) as a stability
     check, logged as "<model>_tuned_r1" and "<model>_tuned_r2".

Honest limits, written down on purpose
  * The tuning is plain CV, not nested CV: the score used to PICK the winner is the same
    score reported for it, so the winner's CV score is slightly optimistic.
  * fold_r1 and fold_r2 are different shuffles of the SAME training rows, so they check
    stability but are not fresh data. The test set is the only honest final number, and it
    is not touched here.
  * class_weight="balanced" shifts the predicted probabilities, so precision, recall and F1
    at the 0.5 cut-off change a lot; PR-AUC (the ranking metric) is not affected in the same way.
    Thresholds are tuned separately, later.

Run from the project root:
    C:\\Users\\User\\venvs\\classification\\Scripts\\python.exe scripts\\tune_finalists.py
Options:
    --models hist_gb            tune only this one
    --confirm-folds ""          skip the stability re-runs
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

import model_zoo
import modeling
import preprocessing as prep

GRID_PATH = modeling.RESULTS_DIR / "tuning_grid.csv"
RANDOM_STATE = 42


def _hgb(**params) -> Pipeline:
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE, early_stopping=True,
                                         max_iter=500, **params)
    return Pipeline([("prep", prep.build_preprocessor()), ("clf", clf)])


def _rf(**params) -> Pipeline:
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, **params)
    return Pipeline([("prep", prep.build_preprocessor()), ("clf", clf)])


GRIDS = {
    "hist_gb": {
        "title": "Histogram gradient boosting (tuned)",
        "factory": _hgb,
        "grid": {
            "learning_rate": [0.05, 0.1],
            "max_leaf_nodes": [15, 31],
            "l2_regularization": [0.0, 1.0],
            "min_samples_leaf": [20, 50],
            "class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "title": "Random forest (tuned)",
        "factory": _rf,
        "grid": {
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", 0.3],
            "class_weight": [None, "balanced_subsample"],
        },
    },
}

GRID_COLUMNS = ["model", "params", "fold_col", "pr_auc_mean", "pr_auc_std", "fold_pr_auc",
                "roc_auc", "precision", "recall", "f1", "pr_auc_female", "pr_auc_male",
                "fit_seconds_total", "convergence_warnings", "timestamp"]


def _key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def _spec(key: str, params: dict) -> model_zoo.ModelSpec:
    cfg = GRIDS[key]
    return model_zoo.ModelSpec(key, cfg["title"], build=lambda p=dict(params): cfg["factory"](**p))


def read_grid(key: str, fold_col: str) -> pd.DataFrame:
    if not GRID_PATH.exists():
        return pd.DataFrame(columns=GRID_COLUMNS)
    grid = pd.read_csv(GRID_PATH)
    return grid[(grid["model"] == key) & (grid["fold_col"] == fold_col)]


def append_grid(row: dict) -> None:
    new = pd.DataFrame([row])[GRID_COLUMNS]
    if GRID_PATH.exists():
        if list(pd.read_csv(GRID_PATH, nrows=0).columns) != GRID_COLUMNS:
            raise ValueError("tuning_grid.csv has a different header; move the old file aside")
        new.to_csv(GRID_PATH, mode="a", header=False, index=False)
    else:
        new.to_csv(GRID_PATH, index=False)


def run_grid(key: str, data: modeling.TrainData) -> None:
    cfg = GRIDS[key]
    names = list(cfg["grid"])
    combos = [dict(zip(names, values)) for values in itertools.product(*cfg["grid"].values())]
    done = set(read_grid(key, data.fold_col)["params"])
    print(f"\n=== {key}: {len(combos)} combinations, {len(done)} already done ===", flush=True)
    for i, params in enumerate(combos, start=1):
        if _key(params) in done:
            continue
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):     # hide the per-fold lines
                row, _ = modeling.cross_validate(_spec(key, params), data)
        except Exception:
            print(f"[FAIL] {params}\n{traceback.format_exc()}", flush=True)
            continue
        append_grid({
            "model": key, "params": _key(params), "fold_col": data.fold_col,
            "pr_auc_mean": row["pr_auc_mean"], "pr_auc_std": row["pr_auc_std"],
            "fold_pr_auc": row["fold_pr_auc"], "roc_auc": row["roc_auc"],
            "precision": row["precision"], "recall": row["recall"], "f1": row["f1"],
            "pr_auc_female": row["pr_auc_female"], "pr_auc_male": row["pr_auc_male"],
            "fit_seconds_total": row["fit_seconds_total"],
            "convergence_warnings": row["convergence_warnings"],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"[{i:>2}/{len(combos)}] PR-AUC {row['pr_auc_mean']:.4f} (std {row['pr_auc_std']:.4f})  "
              f"{time.perf_counter() - t0:5.0f}s  {params}", flush=True)


def best_params(key: str, fold_col: str) -> tuple[dict, pd.DataFrame]:
    grid = read_grid(key, fold_col).sort_values("pr_auc_mean", ascending=False)
    if grid.empty:
        raise RuntimeError(f"No grid results for {key} on {fold_col}")
    return json.loads(grid.iloc[0]["params"]), grid


def save_winner(key: str, params: dict, data: modeling.TrainData) -> None:
    """Cross-validate the chosen settings on data.fold_col and log them (never overwrites)."""
    suffix = data.fold_col.replace("fold_", "")
    experiment_id = f"{key}_tuned_{suffix}"
    if experiment_id in modeling.existing_ids():
        print(f"[skip] {experiment_id} is already in results/model_results.csv", flush=True)
        return
    print(f"[run ] {experiment_id}  {params}", flush=True)
    spec = _spec(key, params)
    row, oof = modeling.cross_validate(spec, data)
    clf = spec.build().named_steps["clf"]
    row.update({
        "experiment_id": experiment_id,
        "model": GRIDS[key]["title"],
        "data_version": modeling.DATA_VERSION,
        "fold_col": data.fold_col,
        "n_rows": int(len(data.y)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "params": json.dumps(clf.get_params(), default=str, sort_keys=True),
    })
    modeling.OOF_DIR.mkdir(parents=True, exist_ok=True)
    oof_path = modeling.OOF_DIR / f"{experiment_id}.csv"
    if oof_path.exists():
        raise FileExistsError(f"{oof_path} exists but the id is not logged; resolve by hand")
    oof.to_csv(oof_path, index=False)
    modeling.append_result(row)
    print(f"[done] {experiment_id}: PR-AUC mean {row['pr_auc_mean']:.4f} (std {row['pr_auc_std']:.4f}), "
          f"F1 {row['f1']:.4f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", default="hist_gb,random_forest")
    parser.add_argument("--fold-col", default="fold_r0")
    parser.add_argument("--confirm-folds", default="fold_r1,fold_r2")
    args = parser.parse_args()

    keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [k for k in keys if k not in GRIDS]
    if unknown:
        raise SystemExit(f"No grid defined for {unknown}. Valid: {list(GRIDS)}")
    confirm = [c.strip() for c in args.confirm_folds.split(",") if c.strip()]

    print("Python executable:", sys.executable)
    print("pandas       ", pd.__version__)
    print("scikit-learn ", sklearn.__version__)
    data = modeling.load_train(args.fold_col)
    print(f"Train rows: {len(data.y):,}   tuning fold column: {args.fold_col}", flush=True)

    started = time.perf_counter()
    failed = []
    for key in keys:
        try:
            run_grid(key, data)
            params, grid = best_params(key, args.fold_col)
            print(f"\nTop 5 for {key} (by per-fold mean PR-AUC):")
            show = grid.head(5)[["pr_auc_mean", "pr_auc_std", "f1", "recall", "params"]]
            with pd.option_context("display.width", 250, "display.max_colwidth", 120):
                print(show.round(4).to_string(index=False))
            print(f"Winner: {params}", flush=True)
            save_winner(key, params, data)
            for fold_col in confirm:
                save_winner(key, params, modeling.load_train(fold_col))
        except Exception:
            failed.append(key)
            print(f"[FAIL] {key}\n{traceback.format_exc()}", flush=True)

    print("\n" + "=" * 70)
    print(f"Finished in {(time.perf_counter() - started) / 60:.1f} min.  failed: {failed}")
    table = modeling.comparison_table()
    print("\nDefault and tuned rows, best PR-AUC (per-fold mean) first:\n")
    table = table[table.index.str.contains("hist_gb|random_forest|logreg")]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.sort_values("pr_auc_mean", ascending=False).round(4).to_string())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
