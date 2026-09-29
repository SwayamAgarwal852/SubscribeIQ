"""Tests for the ETL transform logic in database/seed.py (no database required)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))

from seed import RAW_CSV, clean_total_charges, count_services, transform, yes_no_to_bool  # noqa: E402


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW_CSV, dtype={"TotalCharges": str})


def test_blank_total_charges_become_zero_only_for_tenure_zero(raw):
    cleaned = clean_total_charges(raw)
    assert cleaned["TotalCharges"].notna().all()
    blank = raw["TotalCharges"].str.strip() == ""
    assert blank.sum() == 11
    assert (cleaned.loc[blank, "TotalCharges"] == 0).all()
    assert (raw.loc[blank, "tenure"] == 0).all()


def test_blank_total_charges_with_positive_tenure_raises():
    bad = pd.DataFrame({"TotalCharges": [" "], "tenure": [5]})
    with pytest.raises(ValueError):
        clean_total_charges(bad)


def test_count_services_known_customer(raw):
    # 7590-VHVEG: no phone, DSL internet, OnlineBackup only -> 2 services
    row = raw[raw["customerID"] == "7590-VHVEG"]
    assert count_services(row).iloc[0] == 2


def test_count_services_range(raw):
    counts = count_services(raw)
    assert counts.between(0, 9).all()


def test_yes_no_to_bool_rejects_unexpected_values():
    with pytest.raises(ValueError):
        yes_no_to_bool(pd.Series(["Yes", "Maybe"], name="x"))


def test_transform_shapes_and_ltv(raw):
    frames = transform(raw)
    assert set(frames) == {"dim_customer", "dim_service", "dim_contract", "fact_subscription"}
    for frame in frames.values():
        assert len(frame) == 7043
        assert frame["customer_id"].is_unique
    fact = frames["fact_subscription"]
    expected = (fact["monthly_charges"] * fact["tenure_months"]).round(2)
    assert (fact["estimated_ltv"] == expected).all()
    assert fact["churn"].sum() == 1869
