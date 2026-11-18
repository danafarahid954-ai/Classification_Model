"""
final_evaluation.py - the ONE-TIME evaluation of the chosen model on the held-out test set.

Decisions made before this script was written (15 November, on training rows only)
  * model     : histogram gradient boosting, default settings, no class weight
  * threshold : 0.37 (best F1 on the out-of-fold predictions, averaged over 3 fold sets)
Neither is changed after seeing test results. The test rows are used here, once.

What it does
  1. Fits the model on ALL training rows (39,032) - the test rows play no part in fitting.
  2. Scores the 9,758 test rows once.
  3. Reports PR-AUC, ROC-AUC, precision, recall, F1, accuracy and the confusion matrix at the
     chosen threshold (and at 0.5 for reference), overall and by sex, with each group's floor
     (its own >50K share) and a bootstrap 95% interval for the overall numbers.
  4. Saves results/final_evaluation.csv, results/final_test_predictions.csv and the fitted model
     in models/. If results/final_evaluation.csv already exists the script refuses to run:
     the test set is used once.

Run from the project root. The flag is a deliberate guard against running it by accident:
    C:\\Users\\User\\venvs\\classification\\Scripts\\python.exe scripts\\final_evaluation.py --confirm-final
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

import model_zoo
import modeling
import preprocessing as prep

MODEL_KEY = "hist_gb"
THRESHOLD = 0.37
REFERENCE_THRESHOLD = 0.5

# What cross-validation on the training rows led us to expect (3 fold sets, default boosting).
CV_PR_AUC = 0.8307
CV_AT_THRESHOLD = {"precision": 0.6926, "recall": 0.7676, "f1": 0.7282}

OUT_CSV = modeling.RESULTS_DIR / "final_evaluation.csv"
OUT_PRED = modeling.RESULTS_DIR / "final_test_predictions.csv"
MODEL_DIR = modeling.ROOT / "models"
MODEL_PATH = MODEL_DIR / "final_hist_gb.joblib"
META_PATH = MODEL_DIR / "final_hist_gb_meta.json"


def load_test_rows(train_ids: np.ndarray):
    """Raw feature rows, labels and sex of the test rows, after checking they are disjoint from train."""
    df = pd.read_csv(modeling.DATA_PATH, header=None, names=prep.RAW_COLUMNS, keep_default_na=False)
    split_df = pd.read_csv(modeling.SPLIT_PATH)
    test_ids = np.sort(split_df.loc[split_df["split"] == "test", "row_id"].to_numpy())
    if len(test_ids) != 9758:
        raise ValueError(f"Expected 9,758 test rows, found {len(test_ids)}")
    if set(test_ids) & set(train_ids):
        raise AssertionError("Train and test row ids overlap")
    frame = df.iloc[test_ids].reset_index(drop=True)
    y = (frame["income"].astype(str).str.strip() == modeling.POSITIVE_LABEL).astype(int).to_numpy()
    sex = frame["sex"].astype(str).str.strip().to_numpy()
    return test_ids, frame, y, sex


def quick_prf(y, pred):
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def bootstrap_interval(y, score, threshold, n_boot=1000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {k: [] for k in ["pr_auc", "roc_auc", "precision", "recall", "f1"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], score[idx]
        draws["pr_auc"].append(average_precision_score(yb, sb))
        draws["roc_auc"].append(roc_auc_score(yb, sb))
        p, r, f = quick_prf(yb, (sb >= threshold).astype(int))
        draws["precision"].append(p); draws["recall"].append(r); draws["f1"].append(f)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


def print_report(result: pd.DataFrame, y_test, score, n_boot: int) -> None:
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        cols = ["group", "threshold", "n", "floor_pr_auc", "pr_auc", "roc_auc", "accuracy",
                "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
        print("\nTest-set results:\n")
        print(result[cols].round(4).to_string(index=False))

    overall = result[(result["group"] == "all") & (result["threshold"] == THRESHOLD)].iloc[0]
    ci = bootstrap_interval(y_test, score, THRESHOLD, n_boot=n_boot)
    print(f"\nBootstrap 95% intervals ({n_boot} resamples of the test rows), threshold {THRESHOLD}:")
    for name in ["pr_auc", "roc_auc", "precision", "recall", "f1"]:
        print(f"  {name:<10} {overall[name]:.4f}   [{ci[name][0]:.4f}, {ci[name][1]:.4f}]")

    print("\nCompared with what cross-validation on the training rows led us to expect:")
    print(f"  PR-AUC     expected {CV_PR_AUC:.4f}   test {overall['pr_auc']:.4f}   "
          f"({'inside' if ci['pr_auc'][0] <= CV_PR_AUC <= ci['pr_auc'][1] else 'OUTSIDE'} the test interval)")
    for name, expected in CV_AT_THRESHOLD.items():
        print(f"  {name:<10} expected {expected:.4f}   test {overall[name]:.4f}   "
              f"({'inside' if ci[name][0] <= expected <= ci[name][1] else 'OUTSIDE'} the test interval)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--confirm-final", action="store_true",
                        help="required: confirms you want to use the test set now, once")
    parser.add_argument("--report-only", action="store_true",
                        help="reprint the report from the saved files; does not refit or rescore")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    if args.report_only:                          # reads files that already exist; no model, no test rows
        if not OUT_CSV.exists() or not OUT_PRED.exists():
            raise SystemExit("No saved final results to report on.")
        saved = pd.read_csv(OUT_PRED)
        print_report(pd.read_csv(OUT_CSV), saved["y_true"].to_numpy(), saved["score"].to_numpy(), args.bootstrap)
        return 0
    if not args.confirm_final:
        raise SystemExit("Refusing to run without --confirm-final: this uses the test set, once.")
    if OUT_CSV.exists():
        raise SystemExit(f"{OUT_CSV.name} already exists: the test set has already been used. Nothing was "
                         "run. Use --report-only to see the saved results again.")

    print("Python executable:", sys.executable)
    print("scikit-learn     ", sklearn.__version__)

    train = modeling.load_train("fold_r0")                     # train rows only
    test_ids, test_frame, y_test, sex_test = load_test_rows(train.row_id)
    print(f"Train rows: {len(train.y):,}   Test rows: {len(y_test):,}   "
          f">50K share train {train.y.mean():.4%}  test {y_test.mean():.4%}")

    spec = model_zoo.SPEC_BY_KEY[MODEL_KEY]
    model = spec.build()
    print(f"\nFitting {spec.title} on all {len(train.y):,} training rows ...", flush=True)
    model.fit(train.frame, train.y)
    score = model.predict_proba(test_frame)[:, 1]              # the single use of the test rows

    rows = []
    for threshold in [THRESHOLD, REFERENCE_THRESHOLD]:
        for group, mask in [("all", np.ones(len(y_test), bool))] + \
                           [(g, sex_test == g) for g in modeling.SEX_GROUPS]:
            block = modeling.metric_block(y_test[mask], score[mask], (score[mask] >= threshold).astype(int))
            rows.append({"group": group, "threshold": threshold, "n": int(mask.sum()),
                         "floor_pr_auc": float(y_test[mask].mean()), **block})
    result = pd.DataFrame(rows)

    modeling.RESULTS_DIR.mkdir(exist_ok=True)
    result.assign(model=spec.title, data_version=modeling.DATA_VERSION,
                  timestamp=datetime.now().isoformat(timespec="seconds"),
                  sklearn_version=sklearn.__version__).to_csv(OUT_CSV, index=False)
    pd.DataFrame({"row_id": test_ids, "y_true": y_test, "sex": sex_test, "score": score,
                  f"pred_at_{THRESHOLD}": (score >= THRESHOLD).astype(int)}).to_csv(OUT_PRED, index=False)
    print(f"\nSaved the test results: {OUT_CSV.name} and {OUT_PRED.name} in results/.")

    # The report comes before the model is saved, so a problem saving the model cannot hide it.
    print_report(result, y_test, score, args.bootstrap)

    try:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        META_PATH.write_text(json.dumps({
            "model": spec.title, "threshold": THRESHOLD, "sklearn_version": sklearn.__version__,
            "train_rows": int(len(train.y)), "data_version": modeling.DATA_VERSION,
            "trained": datetime.now().isoformat(timespec="seconds"),
            "note": "Input: raw feature rows as in data/adult.csv (the pipeline includes preprocessing). "
                    "Predict the >50K probability with predict_proba(...)[:, 1] and compare with the threshold."},
            indent=2))
        print(f"\nSaved the fitted model to models/{MODEL_PATH.name} (threshold {THRESHOLD} in {META_PATH.name}).")
    except Exception as error:                    # e.g. a lambda inside the preprocessor cannot be pickled
        print(f"\n[WARNING] The fitted model could not be saved: {type(error).__name__}: {str(error)[:200]}")
        print("The test results above are saved and valid. The model can be refitted from the training "
              "rows with the same code (it is deterministic); this does not need the test set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
