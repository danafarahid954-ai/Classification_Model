"""
run_models.py - train and cross-validate every model in model_zoo.py, one after the other.

Run from the project root, with the project's virtual environment:

    C:\\Users\\User\\venvs\\classification\\Scripts\\python.exe scripts\\run_models.py

Useful options:
    --list                      show the models and exit
    --models knn,random_forest  run only these
    --skip rbf_svm              leave one out (the RBF SVM is the slowest)
    --fold-col fold_r1          use another saved fold assignment (default fold_r0)

Each model is independent: if one fails, the error is printed and the rest still run.
Models already in results/model_results.csv are skipped, so the script can be re-run
safely after an interruption.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # lets "import preprocessing" work

import numpy as np
import pandas as pd
import sklearn

import model_zoo
import modeling


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", default="all")
    parser.add_argument("--skip", default="")
    parser.add_argument("--fold-col", default="fold_r0")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    specs = model_zoo.select(args.models, args.skip)
    if args.list:
        for spec in model_zoo.MODEL_SPECS:
            print(f"{spec.key:<14} {spec.title}{'   (slow)' if spec.slow else ''}")
        return 0

    print("Python executable:", sys.executable)
    print("pandas       ", pd.__version__)
    print("numpy        ", np.__version__)
    print("scikit-learn ", sklearn.__version__)

    data = modeling.load_train(args.fold_col)
    print(f"\nTrain rows: {len(data.y):,}   >50K share: {data.y.mean():.4%}   fold column: {args.fold_col}")
    print(f"Models to run: {[s.key for s in specs]}\n")

    started = time.perf_counter()
    ran, skipped, failed = [], [], []
    for spec in specs:
        try:
            (ran if modeling.run_one(spec, data) else skipped).append(spec.key)
        except Exception:                        # one broken model must not stop the others
            failed.append(spec.key)
            print(f"[FAIL] {spec.key}\n{traceback.format_exc()}", flush=True)

    print("\n" + "=" * 70)
    print(f"Finished in {(time.perf_counter() - started) / 60:.1f} min.  ran: {ran}  skipped: {skipped}  failed: {failed}")
    if modeling.RESULTS_PATH.exists():
        print("\nAll logged models, best PR-AUC first (floor = 0.2394 overall):\n")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(modeling.comparison_table().round(4).to_string())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
