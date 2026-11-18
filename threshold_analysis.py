"""
threshold_analysis.py - how precision, recall and F1 change with the decision threshold.

Reads the saved out-of-fold (OOF) predictions of one model (results/oof/<experiment_id>.csv),
so nothing is retrained and the TEST set is never touched. With several experiment ids
(the same model on fold_r0, fold_r1, fold_r2) it averages them, which shows how stable the
chosen threshold is.

Why this exists: the ranking quality of a model (PR-AUC) does not depend on the threshold, but
precision and recall do. The 0.5 cut-off is only a convention; for a 24% positive class the
best cut-off is usually lower. The threshold is a choice about what matters (catching more
>50K earners versus being right when we say >50K), so this script shows the trade-off and
the numbers by sex; it does not decide for you.

Run from the project root, for example for the default boosting model:
    C:\\Users\\User\\venvs\\classification\\Scripts\\python.exe scripts\\threshold_analysis.py ^
        --experiments hist_gb_default_r0,hist_gb_default_r1,hist_gb_default_r2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import modeling

GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
RECALL_TARGETS = [0.70, 0.75, 0.80]


def confusion_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / len(y), "flagged_share": (tp + fp) / len(y)}


def curve(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([{"threshold": t, **confusion_metrics(y, (score >= t).astype(int))} for t in GRID])


def at_threshold(y, score, sex, t) -> dict:
    out = {"threshold": t}
    out.update(confusion_metrics(y, (score >= t).astype(int)))
    for group in modeling.SEX_GROUPS:
        m = sex == group
        block = confusion_metrics(y[m], (score[m] >= t).astype(int))
        for name in ["precision", "recall", "f1"]:
            out[f"{name}_{group.lower()}"] = block[name]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--experiments", required=True,
                        help="comma list of experiment ids that have a file in results/oof/")
    args = parser.parse_args()
    ids = [e.strip() for e in args.experiments.split(",") if e.strip()]

    data = modeling.load_train("fold_r0")          # only used for row order, y and sex
    runs = []
    for experiment_id in ids:
        path = modeling.OOF_DIR / f"{experiment_id}.csv"
        if not path.exists():
            raise SystemExit(f"Not found: {path}")
        oof = pd.read_csv(path)
        if not np.array_equal(oof["row_id"].to_numpy(), data.row_id) or not np.array_equal(oof["y_true"].to_numpy(), data.y):
            raise SystemExit(f"{experiment_id}: OOF rows do not line up with the training rows")
        runs.append(oof["score"].to_numpy())
        print(f"loaded {experiment_id}: {len(oof):,} rows")

    # F1 versus threshold, per run and averaged.
    curves = [curve(data.y, s) for s in runs]
    mean_curve = pd.concat(curves).groupby("threshold", as_index=False).mean()
    best_each = [float(c.loc[c["f1"].idxmax(), "threshold"]) for c in curves]
    t_f1 = float(mean_curve.loc[mean_curve["f1"].idxmax(), "threshold"])
    print(f"\nF1-best threshold per run: {best_each}   averaged curve: {t_f1}")

    candidates = [("default 0.5", 0.50), ("best F1", t_f1)]
    for target in RECALL_TARGETS:
        ok = mean_curve[mean_curve["recall"] >= target]
        if not ok.empty:
            candidates.append((f"recall >= {target:.2f}", float(ok["threshold"].max())))

    rows = []
    for label, t in candidates:
        per_run = pd.DataFrame([at_threshold(data.y, s, data.sex, t) for s in runs])
        rows.append({"choice": label, **per_run.mean().to_dict()})
    table = pd.DataFrame(rows).set_index("choice")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nAverage over the runs (overall, then by sex):\n")
        print(table.round(4).to_string())
        print("\nF1 and recall along the averaged curve (every 0.05):\n")
        print(mean_curve[np.isclose(mean_curve["threshold"] * 20, np.round(mean_curve["threshold"] * 20))]
              [["threshold", "precision", "recall", "f1", "flagged_share"]].round(4).to_string(index=False))

    out = modeling.RESULTS_DIR / f"threshold_curve_{ids[0]}.csv"
    if out.exists():
        print(f"\n[note] {out.name} already exists and was not overwritten")
    else:
        mean_curve.to_csv(out, index=False)
        print(f"\nSaved the averaged curve to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
