"""Tests for src/retention_matrix.py (no database required)."""

import pandas as pd
import pytest

from src import retention_matrix as rm


@pytest.fixture
def customers():
    # One customer per quadrant, plus one sitting exactly on both medians
    return pd.DataFrame({
        "customer_id": ["hv_hr", "hv_lr", "lv_hr", "lv_lr", "median"],
        "segment_name": ["A", "A", "B", "B", "B"],
        "churn_probability": [0.9, 0.1, 0.8, 0.05, 0.3],
        "estimated_ltv": [1200.0, 1100.0, 300.0, 250.0, 600.0],
        "monthly_charges": [100.0, 91.67, 25.0, 20.83, 50.0],
        "tenure_months": [2, 60, 3, 40, 10],
        "churn": [True, False, True, False, False],
        "contract_type": ["Month-to-month"] * 5,
    })


def test_quadrant_assignment(customers):
    out, thresholds = rm.assign_actions(customers)
    assert thresholds == {"ltv_threshold": 600.0, "risk_threshold": 0.3}
    actions = dict(zip(out.customer_id, out.retention_action))
    assert actions["hv_hr"] == rm.RETENTION_OFFER
    assert actions["hv_lr"] == rm.EARLY_ACCESS
    assert actions["lv_hr"] == rm.LET_GO
    assert actions["lv_lr"] == rm.NURTURE
    # Exactly on the median is "not high" on both axes
    assert actions["median"] == rm.NURTURE


def test_revenue_at_risk_is_probability_times_ltv(customers):
    out, _ = rm.assign_actions(customers)
    assert out.revenue_at_risk.tolist() == pytest.approx([1080.0, 110.0, 240.0, 12.5, 180.0])


def test_summary_totals_add_up(customers):
    out, _ = rm.assign_actions(customers)
    summary = rm.matrix_summary(out)
    assert summary.loc[rm.ACTIONS, "customers"].sum() == summary.loc["Total", "customers"]
    assert summary.loc[rm.ACTIONS, "revenue_at_risk"].sum() == pytest.approx(
        summary.loc["Total", "revenue_at_risk"])


def test_projected_savings_scales_with_success_rate(customers):
    out, _ = rm.assign_actions(customers)
    savings = rm.projected_savings(out, success_rates=(0.1, 0.3))
    assert savings.loc[10.0, "projected_revenue_saved"] == pytest.approx(108.0)
    assert savings.loc[30.0, "projected_revenue_saved"] == pytest.approx(324.0)
    assert savings.loc[30.0, "expected_customers_saved"] == pytest.approx(0.27)


def test_cost_based_flags_threshold(customers):
    out, _ = rm.assign_actions(customers)
    flags = rm.cost_based_flags(out, cost_ratio=4)          # threshold 0.2
    assert flags["threshold"] == pytest.approx(0.2)
    assert flags["customers_flagged"] == 3                   # 0.9, 0.8, 0.3
    with pytest.raises(ValueError):
        rm.cost_based_flags(out, cost_ratio=0)
