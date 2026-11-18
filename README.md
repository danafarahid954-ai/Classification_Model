# Adult Census Income: income classification

A classification project that predicts whether a person earns more than 50K a year from census features, built step by step with a strict no-leakage method (everything learned from the data is fitted on training rows only, inside scikit-learn pipelines). It is a learning project, not a tool for decisions about real people.

## Result
Final model: histogram gradient boosting (scikit-learn, default settings, no class weight) with decision threshold 0.37. On the 9,758 held-out test rows, used once: PR-AUC 0.817 (a model with no skill scores 0.239), ROC-AUC 0.924, precision 0.683, recall 0.770, F1 0.724. By sex at the same threshold: recall 0.706 for women and 0.781 for men, precision 0.704 and 0.679. Cross-validation on the training rows had predicted PR-AUC 0.831; the 0.0135 gap is a little larger than sampling noise alone explains (the test 95% bootstrap interval, 0.804 to 0.830, ends just below the CV figure), so it is most likely noise plus mild optimism from choosing the model on the CV scores; I found no sign of leakage.

Model comparison (per-fold mean PR-AUC, 5-fold CV on the training rows): histogram gradient boosting 0.830, random forest 0.809, linear SVM 0.777, neural network (MLP) 0.771, RBF SVM 0.770, logistic regression 0.768, decision tree 0.756, k-nearest neighbors 0.732, Gaussian naive Bayes 0.637. Tuning did not improve boosting; class weights only moved the threshold.

## Method
- 52 exact duplicate rows are dropped first, then a stratified 80/20 split (on income and sex, seed 42): 39,032 training and 9,758 test rows.
- 5-fold stratified cross-validation, saved in results/cv_folds.csv with three repeats (seeds 42, 43, 44), reused by every model.
- Preprocessing is one pipeline (scripts/preprocessing.py): "?" becomes "Unknown", native_country grouped in three, capital gain/loss as log1p plus a flag, scaling and one-hot encoding fitted per fold.
- PR-AUC for the >50K class is the ranking metric; precision, recall, F1, ROC-AUC, the confusion matrix and everything by sex are reported against each group's own floor.

## Project layout
- data/ (raw adult.csv, never modified), notebooks/ (01 to 05), scripts/ (preprocessing and the model scripts), results/ (splits, folds, logs, out-of-fold predictions, final test results), models/ (the final fitted model), docs/ (full project documentation), images/.

## Setup
Python 3.11 with numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, matplotlib 3.11.2, scikit-learn 1.9.1 and ipykernel 7.4.0 (see requirements.txt). Tested on 15 Nov 2026 in a fresh Python 3.11 venv: `pip install -r requirements.txt` installed without errors, the packages imported at the pinned versions, `scripts\run_models.py --list` listed all nine models, and models/final_hist_gb.joblib loaded. Only the top-level packages are pinned, so their sub-dependencies may resolve to different versions over time. Notebooks use a Jupyter kernel named "Python (classification)". If pip reports a certificate error, point it at your antivirus certificate with --cert; never disable verification.

## Reproduce (from the project root)
- Run notebooks/01_split.ipynb, 02_cv_folds.ipynb, 03_baseline.ipynb, 04_preprocessing_checks.ipynb and 05_logreg_plain.ipynb on that kernel.
- `python scripts\run_models.py` runs all nine models; repeat with `--fold-col fold_r1` and `--fold-col fold_r2` and `--models logreg,hist_gb,random_forest`.
- `python scripts\tune_finalists.py` tunes the finalists.
- `python scripts\threshold_analysis.py --experiments hist_gb_default_r0,hist_gb_default_r1,hist_gb_default_r2` shows the threshold trade-off.
- `python scripts\final_evaluation.py --confirm-final` is the one-time test evaluation; it refuses to run twice.

## Limits
Sex is a model input and the single threshold gives women a lower recall; the census sampling weights (fnlwgt) are ignored; the data are old, from one country, and income thresholds are not inflation-adjusted; probability calibration was not checked.

## Documentation
The full record (decisions, results of every model, model card, problems and fixes) is in docs/Classification_Project_Learning_Log.md.
