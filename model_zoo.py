"""
model_zoo.py - the classifiers we compare, each a self-contained spec.

Every model is built as Pipeline([("prep", preprocessor), ("clf", classifier)]) so
that the scaler and one-hot categories are refitted on each fold's training part.
The settings below are sensible, fixed, UNTUNED defaults: this is a screening
round. Tuning is done later, only for the finalists.

The same preprocessor (scaled numbers, one-hot categories) is used for all models
so the only thing that changes between runs is the model. Scaling is not needed by
the tree-based models, but it is harmless for them (it does not change any split).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import preprocessing as prep

RANDOM_STATE = 42


def _pipeline(clf) -> Pipeline:
    return Pipeline([("prep", prep.build_preprocessor()), ("clf", clf)])


@dataclass(frozen=True)
class ModelSpec:
    key: str                       # short id used in experiment_id, e.g. "knn"
    title: str                     # readable name
    build: Callable[[], Pipeline]  # returns a fresh, unfitted pipeline
    slow: bool = False


# Ordered from fastest to slowest so early results appear quickly.
MODEL_SPECS = [
    ModelSpec("logreg", "Logistic regression",
              lambda: _pipeline(LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000))),
    ModelSpec("naive_bayes", "Gaussian naive Bayes",
              lambda: _pipeline(GaussianNB())),
    ModelSpec("decision_tree", "Decision tree",
              lambda: _pipeline(DecisionTreeClassifier(
                  max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE))),
    ModelSpec("linear_svm", "Linear SVM",
              lambda: _pipeline(LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE))),
    ModelSpec("hist_gb", "Histogram gradient boosting",
              lambda: _pipeline(HistGradientBoostingClassifier(random_state=RANDOM_STATE))),
    ModelSpec("random_forest", "Random forest",
              lambda: _pipeline(RandomForestClassifier(
                  n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))),
    ModelSpec("knn", "k-nearest neighbors",
              lambda: _pipeline(KNeighborsClassifier(n_neighbors=25, n_jobs=-1))),
    ModelSpec("mlp", "Neural network (MLP)",
              lambda: _pipeline(MLPClassifier(
                  hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300,
                  random_state=RANDOM_STATE)), slow=True),
    ModelSpec("rbf_svm", "SVM with RBF kernel",
              lambda: _pipeline(SVC(kernel="rbf", C=1.0, gamma="scale")), slow=True),
]

SPEC_BY_KEY = {spec.key: spec for spec in MODEL_SPECS}


def select(models: str = "all", skip: str = "") -> list[ModelSpec]:
    """models: 'all' or a comma list of keys. skip: a comma list of keys to leave out."""
    wanted = [m.strip() for m in models.split(",") if m.strip()]
    skipped = {m.strip() for m in skip.split(",") if m.strip()}
    unknown = [m for m in wanted + sorted(skipped) if m != "all" and m not in SPEC_BY_KEY]
    if unknown:
        raise SystemExit(f"Unknown model key(s): {unknown}. Valid: {list(SPEC_BY_KEY)}")
    chosen = MODEL_SPECS if (not wanted or "all" in wanted) else [SPEC_BY_KEY[m] for m in wanted]
    return [spec for spec in chosen if spec.key not in skipped]
