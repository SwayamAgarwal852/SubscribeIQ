"""Tests for src/explainability.py (saved model; the last test needs PostgreSQL)."""

import numpy as np
import pytest
from sqlalchemy import text

from src import churn_model as cm
from src import explainability as xai
from tests.test_churn_model import _synthetic_rows


@pytest.fixture(scope="module")
def explainer():
    X = _synthetic_rows(n=300, seed=1)
    return xai.build_explainer(cm.load_model(), X, background_size=100), X


def test_every_encoded_column_maps_to_a_feature(explainer):
    ce, _ = explainer
    assert sorted(set(ce.encoded_to_feature)) == sorted(cm.FEATURES)


def test_shap_values_add_up_to_the_margin(explainer):
    ce, X = explainer
    values = xai.shap_values(ce, X)
    assert list(values.columns) == cm.FEATURES
    np.testing.assert_allclose(ce.base_value + values.sum(axis=1), xai.margins(ce, X), atol=1e-6)


def test_margin_maps_to_the_calibrated_probability(explainer):
    ce, X = explainer
    np.testing.assert_allclose(xai.to_probability(ce, xai.margins(ce, X)),
                               cm.predict_churn(ce.calibrated, X), atol=1e-12)


def test_global_importance_shares_sum_to_one(explainer):
    ce, X = explainer
    values = xai.shap_values(ce, X)
    for groups in (None, cm.FEATURE_GROUPS):
        imp = xai.global_importance(values, groups)
        assert imp.share.sum() == pytest.approx(1.0)
        assert imp.mean_abs_shap.is_monotonic_decreasing
    assert len(xai.global_importance(values, cm.FEATURE_GROUPS)) == len(cm.FEATURE_GROUPS)


def test_customer_explanation_keeps_the_total(explainer):
    ce, X = explainer
    values = xai.shap_values(ce, X)
    expl = xai.customer_explanation(values.iloc[0], X.iloc[0], top_n=5)
    assert len(expl) == 6 and expl.iloc[-1].label == f"Other {len(cm.FEATURES) - 5} features"
    assert expl.shap.sum() == pytest.approx(values.iloc[0].sum())
    top = expl.shap.iloc[:5].abs()
    assert top.is_monotonic_decreasing


def test_format_value():
    assert xai.format_value("tenure_months", 1) == "1 month"
    assert xai.format_value("tenure_months", 24) == "24 months"
    assert xai.format_value("monthly_charges", 70.7) == "$70.70"
    assert xai.format_value("rfm_recency_score", 3.0) == "3"
    assert xai.format_value("contract_type", "Two year") == "Two year"


def _db_available() -> bool:
    try:
        from src.db_connection import get_engine
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable")
def test_explains_every_real_customer_exactly():
    from src.db_connection import get_engine

    data = cm.load_modeling_data(get_engine())
    X = data[cm.FEATURES]
    ce = xai.build_explainer(cm.load_model(), X)
    values = xai.shap_values(ce, X)
    margin = xai.margins(ce, X)
    np.testing.assert_allclose(ce.base_value + values.sum(axis=1), margin, atol=1e-6)
    np.testing.assert_allclose(xai.to_probability(ce, margin),
                               cm.predict_churn(cm.load_model()["pipeline"], X), atol=1e-12)
    # Month-to-month contracts push risk up on average, two-year contracts push it down
    by_contract = values.contract_type.groupby(X.contract_type.to_numpy()).mean()
    assert by_contract["Month-to-month"] > 0 > by_contract["Two year"]
