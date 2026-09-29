"""Tests for src/churn_model.py (no database required)."""

import numpy as np
import pandas as pd
import pytest

from src import churn_model as cm


def _synthetic_rows(n=40, seed=0):
    """Small raw-feature frame using the real category labels."""
    rng = np.random.default_rng(seed)
    yes_no = ["Yes", "No"]
    addon = ["Yes", "No", "No internet service"]
    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "senior_citizen": rng.choice(yes_no, n),
        "partner": rng.choice(yes_no, n),
        "dependents": rng.choice(yes_no, n),
        "phone_service": rng.choice(yes_no, n),
        "multiple_lines": rng.choice(["Yes", "No", "No phone service"], n),
        "internet_service": rng.choice(["DSL", "Fiber optic", "No"], n),
        **{c: rng.choice(addon, n) for c in ["online_security", "online_backup", "device_protection",
                                             "tech_support", "streaming_tv", "streaming_movies"]},
        "contract_type": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "payment_method": rng.choice(["Electronic check", "Mailed check"], n),
        "paperless_billing": rng.choice(yes_no, n),
        "tenure_months": rng.integers(0, 73, n),
        "monthly_charges": rng.uniform(18, 120, n),
        "total_charges": rng.uniform(0, 8000, n),
        "num_services": rng.integers(1, 10, n),
        "rfm_recency_score": rng.integers(1, 6, n),
        "rfm_frequency_score": rng.integers(1, 6, n),
        "rfm_monetary_score": rng.integers(1, 6, n),
    })


def test_feature_groups_cover_every_feature_once():
    covered = [c for cols in cm.FEATURE_GROUPS.values() for c in cols]
    assert sorted(covered) == sorted(cm.FEATURES)


def test_pipeline_fits_and_predicts_probabilities():
    X = _synthetic_rows()
    y = pd.Series(np.arange(len(X)) % 2)
    model = cm.build_pipeline(cm.LogisticRegression(max_iter=1000)).fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")
def test_preprocessor_handles_unseen_category():
    X = _synthetic_rows()
    prep = cm.build_preprocessor().fit(X)
    unseen = X.head(1).copy()
    unseen["payment_method"] = "Credit card (automatic)"   # not in the synthetic training data
    assert prep.transform(unseen).shape[1] == prep.transform(X.head(1)).shape[1]


def test_threshold_table_counts_are_consistent():
    y = pd.Series([0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.7, 0.9])
    table = cm.threshold_table(y, proba, thresholds=[0.5])
    row = table.loc[0.5]
    assert (row.tp, row.fp, row.fn, row.tn) == (2, 1, 1, 1)
    assert row.recall == pytest.approx(2 / 3)


@pytest.mark.skipif(not cm.MODEL_PATH.exists(), reason="model not trained yet")
def test_saved_artifact_predicts_from_raw_columns():
    artifact = cm.load_model()
    assert artifact["features"] == cm.FEATURES
    assert artifact["calibration"] == "isotonic"
    rows = _synthetic_rows(5)[artifact["features"]]
    for key in ("pipeline", "base_pipeline"):
        proba = artifact[key].predict_proba(rows)[:, 1]
        assert ((proba >= 0) & (proba <= 1)).all()
    assert list(artifact["base_pipeline"].named_steps) == ["prep", "model"]
