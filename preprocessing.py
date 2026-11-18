"""Preprocessing for the Adult Census Income project.

Implements the agreed decisions in CLAUDE.md as one sklearn object:

    build_preprocessor() -> Pipeline([
        ("fixed",   FunctionTransformer(apply_fixed_rules)),  # stateless rules, learns nothing
        ("learned", ColumnTransformer([...])),                # fitted on training rows only
    ])

Input: the RAW DataFrame (columns as loaded from data/adult.csv with
header=None, names=RAW_COLUMNS, keep_default_na=False). Extra columns such as
fnlwgt, education and income may be present; they are dropped.

Fixed rules (no fitting, so no leakage):
- "?" -> "Unknown" in every categorical column.
- native_country -> native_country_group: "United-States" / "Unknown" / "Other".
- capital_gain, capital_loss -> log1p(value) and a 0/1 flag (value > 0).
- fnlwgt (sample weight only), education (education_num is used) and income
  (the target) are not features.

Learned steps (fit on the training part only):
- Numeric columns: mean fill (safety net; the data has no missing numeric
  values today), then StandardScaler.
- Flags: passed through unscaled.
- Categorical columns: OneHotEncoder(handle_unknown="ignore"); an unseen
  category becomes an all-zero block.

Nothing runs on import.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Raw file layout (data/adult.csv has no header row)
RAW_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income',
]

# Raw columns that are not features
DROPPED_COLS = ['fnlwgt', 'education', 'income']

# Raw columns the fixed rules read
RAW_NUMERIC_COLS = ['age', 'education_num', 'hours_per_week', 'capital_gain', 'capital_loss']
RAW_CATEGORICAL_COLS = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex']
REQUIRED_RAW_COLS = RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS + ['native_country']

# Feature columns produced by the fixed rules, by treatment
NUMERIC_COLS = ['age', 'education_num', 'hours_per_week', 'capital_gain_log1p', 'capital_loss_log1p']
FLAG_COLS = ['capital_gain_pos', 'capital_loss_pos']
CATEGORICAL_COLS = RAW_CATEGORICAL_COLS + ['native_country_group']
FEATURE_COLS = NUMERIC_COLS + FLAG_COLS + CATEGORICAL_COLS

# Fixed labels
MISSING = '?'
UNKNOWN = 'Unknown'
US = 'United-States'
OTHER = 'Other'


def apply_fixed_rules(X):
    """Apply the fixed (stateless) rules to a raw DataFrame.

    Returns a new DataFrame with exactly FEATURE_COLS, in that order, and the
    same index as X. X itself is not changed. A NaN in a capital column gives
    NaN in its log1p column (filled later by the mean imputer) and flag 0,
    because NaN > 0 is False.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(f'apply_fixed_rules expects a pandas DataFrame, got {type(X).__name__}')

    missing_cols = [c for c in REQUIRED_RAW_COLS if c not in X.columns]
    if missing_cols:
        raise KeyError(f'Raw columns missing from input: {missing_cols}')

    for c in RAW_NUMERIC_COLS:
        if not pd.api.types.is_numeric_dtype(X[c]):
            raise TypeError(f'Column {c!r} should be numeric but has dtype {X[c].dtype}')

    out = pd.DataFrame(index=X.index)

    # Numeric columns used as they are
    for c in ['age', 'education_num', 'hours_per_week']:
        out[c] = X[c].astype(float)

    # Capital columns: log1p plus a 0/1 flag
    for c in ['capital_gain', 'capital_loss']:
        values = X[c].astype(float)
        out[f'{c}_log1p'] = np.log1p(values)
        out[f'{c}_pos'] = (values > 0).astype(int)

    # Categorical columns: "?" -> "Unknown"
    for c in RAW_CATEGORICAL_COLS:
        out[c] = X[c].replace(MISSING, UNKNOWN)

    # native_country -> three groups by a fixed mapping
    country = X['native_country']
    out['native_country_group'] = np.where(
        country == US, US, np.where(country == MISSING, UNKNOWN, OTHER)
    )

    return out[FEATURE_COLS]


def _fixed_feature_names(transformer, input_features):
    """Output column names of the fixed-rules step (for get_feature_names_out)."""
    return np.asarray(FEATURE_COLS, dtype=object)


def build_preprocessor():
    """Return a new, unfitted preprocessor: fixed rules, then learned steps.

    Fit it on the training part only (e.g. the training folds in CV); then
    transform both the training part and the validation part with it.
    """
    fixed = FunctionTransformer(
        apply_fixed_rules,
        feature_names_out=_fixed_feature_names,
    )

    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
    ])

    learned = ColumnTransformer(
        [
            ('num', numeric, NUMERIC_COLS),
            ('flag', 'passthrough', FLAG_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder='drop',
    )

    return Pipeline([
        ('fixed', fixed),
        ('learned', learned),
    ])
