"""End-to-end audit: raw CSV -> warehouse -> segments -> model -> retention matrix.

Every stored number is recomputed from the raw CSV or from the src/ logic and compared with
PostgreSQL. The committed headline figures are pinned in EXPECTED: a deliberate re-run of a
pipeline phase that changes them should update EXPECTED in the same commit.

Skipped automatically if PostgreSQL is unreachable.
"""

import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sqlalchemy import text

from src import churn_model as cm
from src import retention_matrix as rm
from src import segmentation as seg
from src.db_connection import get_engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from seed import RAW_CSV  # noqa: E402

TABLES = ["dim_customer", "dim_service", "dim_contract", "fact_subscription", "customer_segments"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

EXPECTED = {
    "customers": 7043,
    "churn": {True: 1869, False: 5174},
    "sum_monthly_charges": Decimal("456116.60"),
    "sum_total_charges": Decimal("16056168.70"),
    "blank_total_charges": 11,
    "segment_sizes": {"New & Uncommitted": 2420, "Flexible Fiber Users": 1674,
                      "Established Power Users": 1800, "Loyal Basics": 1149},
    "avg_churn_probability": 0.2657,
    "actual_churn_rate": 0.2654,
    "action_counts": {rm.RETENTION_OFFER: 2259, rm.EARLY_ACCESS: 1256, rm.MONITOR_ONLY: 1260, rm.NURTURE: 2268},
    "ltv_threshold": 844.20,
    "risk_threshold": 0.1833,
    "active_revenue_at_risk": 807576,
    "active_retention_offer_customers": 1123,
    "active_savings_10_20_30": (50493, 100986, 151480),
}


def _db_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable")


@pytest.fixture(scope="module")
def engine():
    return get_engine()


def _query(engine, sql: str) -> pd.DataFrame:
    return pd.read_sql(text(sql), engine)


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW_CSV, dtype=str).set_index("customerID")


@pytest.fixture(scope="module")
def warehouse(engine):
    """All four warehouse tables joined, one row per customer, with exact Decimal money columns."""
    sql = """SELECT * FROM dim_customer JOIN dim_service USING (customer_id)
             JOIN dim_contract USING (customer_id) JOIN fact_subscription USING (customer_id)"""
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    return df.set_index("customer_id")


@pytest.fixture(scope="module")
def segments(engine):
    return _query(engine, "SELECT * FROM customer_segments").set_index("customer_id")


@pytest.fixture(scope="module")
def matrix(engine):
    df, thresholds = rm.assign_actions(rm.load_matrix_inputs(engine))
    return df.set_index("customer_id"), thresholds


# --------------------------------------------------------------------------- data integrity
@pytest.mark.parametrize("table", TABLES)
def test_every_table_has_one_row_per_customer(engine, table):
    n, distinct = _query(engine, f"SELECT COUNT(*) n, COUNT(DISTINCT customer_id) d FROM {table}").iloc[0]
    assert n == distinct == EXPECTED["customers"]


@pytest.mark.parametrize("table", TABLES[1:])
def test_no_orphaned_or_missing_customers(engine, table):
    orphans = _query(engine, f"""SELECT COUNT(*) n FROM {table} x
                                 LEFT JOIN dim_customer c USING (customer_id) WHERE c.customer_id IS NULL""").n[0]
    missing = _query(engine, f"""SELECT COUNT(*) n FROM dim_customer c
                                 LEFT JOIN {table} x USING (customer_id) WHERE x.customer_id IS NULL""").n[0]
    assert orphans == missing == 0


def test_churn_counts_match_csv(raw, warehouse):
    assert raw["Churn"].eq("Yes").sum() == EXPECTED["churn"][True]
    assert warehouse["churn"].value_counts().to_dict() == EXPECTED["churn"]


def test_charge_totals_match_csv_to_the_cent(raw, warehouse):
    exact = lambda col: sum((Decimal(v.strip()) for v in col if v.strip()), Decimal("0"))
    assert exact(raw["MonthlyCharges"]) == sum(warehouse["monthly_charges"]) == EXPECTED["sum_monthly_charges"]
    assert exact(raw["TotalCharges"]) == sum(warehouse["total_charges"]) == EXPECTED["sum_total_charges"]


def test_every_raw_column_matches_row_for_row(raw, warehouse):
    r = raw.loc[warehouse.index]
    yes_no = lambda b: np.where(b, "Yes", "No")
    pairs = {
        "gender": warehouse.gender, "SeniorCitizen": np.where(warehouse.senior_citizen, "1", "0"),
        "Partner": yes_no(warehouse.partner), "Dependents": yes_no(warehouse.dependents),
        "PhoneService": yes_no(warehouse.phone_service), "MultipleLines": warehouse.multiple_lines,
        "InternetService": warehouse.internet_service, "OnlineSecurity": warehouse.online_security,
        "OnlineBackup": warehouse.online_backup, "DeviceProtection": warehouse.device_protection,
        "TechSupport": warehouse.tech_support, "StreamingTV": warehouse.streaming_tv,
        "StreamingMovies": warehouse.streaming_movies, "Contract": warehouse.contract_type,
        "PaymentMethod": warehouse.payment_method, "PaperlessBilling": yes_no(warehouse.paperless_billing),
        "Churn": yes_no(warehouse.churn), "tenure": warehouse.tenure_months.astype(str),
        "MonthlyCharges": warehouse.monthly_charges.map(str),
    }
    r = r.assign(MonthlyCharges=r["MonthlyCharges"].map(lambda v: str(Decimal(v).quantize(Decimal("0.01")))))
    mismatches = {col: int((np.asarray(r[col]) != np.asarray(db_col)).sum()) for col, db_col in pairs.items()}
    assert not any(mismatches.values()), mismatches


def test_blank_total_charges_are_zero_and_no_other_row_affected(raw, warehouse):
    r = raw.loc[warehouse.index]
    blank = r["TotalCharges"].str.strip() == ""
    assert blank.sum() == EXPECTED["blank_total_charges"]
    assert (r.loc[blank, "tenure"] == "0").all()
    assert (warehouse.loc[blank, "total_charges"] == 0).all()
    assert (r.loc[~blank, "TotalCharges"].map(lambda v: Decimal(v.strip()))
            == warehouse.loc[~blank, "total_charges"]).all()
    assert (warehouse["tenure_months"] == 0).sum() == blank.sum()


def test_derived_columns_recompute_from_raw(raw, warehouse):
    r = raw.loc[warehouse.index]
    services = ((r.PhoneService == "Yes").astype(int) + (r.MultipleLines == "Yes")
                + r.InternetService.isin(["DSL", "Fiber optic"]) + sum((r[c] == "Yes").astype(int) for c in ADDONS))
    assert (services.to_numpy() == warehouse["num_services"].to_numpy()).all()
    assert (r["MonthlyCharges"].map(Decimal).to_numpy() * 12 == warehouse["estimated_ltv"].to_numpy()).all()


# --------------------------------------------------------------------------- segmentation
def test_every_customer_has_one_complete_segment_row(segments):
    assert segments.isna().sum().sum() == 0
    assert segments["segment_name"].value_counts().to_dict() == EXPECTED["segment_sizes"]
    pairs = segments.groupby("cluster_label")["segment_name"].nunique()
    assert (pairs == 1).all() and segments["segment_name"].nunique() == len(pairs)


def test_stored_rfm_and_segments_reproduce(engine, segments):
    fresh = seg.segment_customers(seg.load_rfm_inputs(engine)).set_index("customer_id")
    stored = segments.loc[fresh.index]
    for col in seg.SCORE_COLUMNS + ["rfm_combined", "segment_name"]:
        assert (fresh[col] == stored[col]).all(), col
    concat = stored[seg.SCORE_COLUMNS].astype(str).agg("".join, axis=1)
    assert (concat == stored["rfm_combined"]).all()


def test_hand_traced_customers_match_quintiles_and_nearest_centroid(engine, segments):
    """Re-derive RFM scores from explicit quintile edges and the segment from the nearest centroid."""
    inputs = seg.load_rfm_inputs(engine).set_index("customer_id")
    edges = {col: pd.qcut(inputs[col], 5, retbins=True, duplicates="drop")[1]
             for col in ["tenure_months", "num_services", "total_charges"]}
    X = segments.loc[inputs.index, seg.SCORE_COLUMNS].to_numpy(float)
    mean, std = X.mean(0), X.std(0)
    centroids = segments.groupby("cluster_label")[seg.SCORE_COLUMNS].mean()
    names = segments.groupby("cluster_label")["segment_name"].first()

    sample = (segments.groupby("segment_name").sample(2, random_state=7).index.tolist()
              + [inputs.index[inputs.tenure_months == 0][0], inputs.index[inputs.tenure_months == 6][0]])
    for cid in sample:
        scores = [int(np.searchsorted(edges[col][1:], inputs.loc[cid, col], side="left")) + 1 for col in edges]
        assert "".join(map(str, scores)) == segments.loc[cid, "rfm_combined"], cid
        z = (np.array(scores) - mean) / std
        nearest = centroids.index[np.argmin((((centroids.to_numpy() - mean) / std - z) ** 2).sum(1))]
        assert names[nearest] == segments.loc[cid, "segment_name"], cid


# --------------------------------------------------------------------------- model
def test_calibrated_average_matches_actual_churn(engine):
    row = _query(engine, """SELECT AVG(churn_probability)::float p, AVG(churn::int)::float a,
                                   MIN(churn_probability)::float lo, MAX(churn_probability)::float hi
                            FROM customer_segments JOIN fact_subscription USING (customer_id)""").iloc[0]
    assert round(row.p, 4) == EXPECTED["avg_churn_probability"]
    assert round(row.a, 4) == EXPECTED["actual_churn_rate"]
    assert abs(row.p - row.a) < 0.005
    assert cm.PROBA_FLOOR <= row.lo and row.hi <= cm.PROBA_CEILING   # no hard 0% / 100%


def test_decile_ranking(engine):
    df = _query(engine, """SELECT churn_probability::float p, churn::int churn
                           FROM customer_segments JOIN fact_subscription USING (customer_id)""")
    deciles = pd.qcut(df.p.rank(method="first"), 10, labels=False)
    actual = df.groupby(deciles)["churn"].mean()
    assert actual.iloc[0] < 0.05 and actual.iloc[-1] > 0.6
    assert actual.iloc[2:].is_monotonic_increasing   # the bottom two deciles are both ~1% (noise)
    assert roc_auc_score(df.churn, df.p) > 0.83


@pytest.mark.skipif(not cm.MODEL_PATH.exists(), reason="model not trained yet")
def test_saved_model_agrees_with_stored_out_of_fold_probabilities(engine, segments):
    """The refit model and the stored OOF probabilities differ, but only within the known gap."""
    artifact = cm.load_model()
    assert artifact["features"] == cm.FEATURES
    data = cm.load_modeling_data(engine).set_index("customer_id")
    refit = cm.predict_churn(artifact["pipeline"], data[cm.FEATURES])
    stored = segments.loc[data.index, "churn_probability"].astype(float).to_numpy()
    gap = np.abs(refit - stored)
    assert gap.mean() < 0.04
    assert np.quantile(gap, 0.95) < 0.12
    assert pd.Series(refit).corr(pd.Series(stored), method="spearman") > 0.97
    assert ((refit >= cm.PROBA_FLOOR) & (refit <= cm.PROBA_CEILING)).all()


# --------------------------------------------------------------------------- retention matrix
def test_every_customer_has_one_action_with_pinned_counts(segments):
    assert segments["retention_action"].isin(rm.ACTIONS).all()
    assert segments["retention_action"].value_counts().to_dict() == EXPECTED["action_counts"]


def test_median_thresholds_and_stored_actions_reproduce(matrix, segments):
    df, thresholds = matrix
    assert round(thresholds["ltv_threshold"], 2) == EXPECTED["ltv_threshold"]
    assert round(thresholds["risk_threshold"], 4) == EXPECTED["risk_threshold"]
    assert (df["retention_action"] == segments.loc[df.index, "retention_action"]).all()


def test_revenue_at_risk_by_hand(engine, matrix):
    df, _ = matrix
    with engine.connect() as conn:
        rows = conn.execute(text("""SELECT customer_id, cs.churn_probability, f.estimated_ltv,
                                           cs.churn_probability * f.estimated_ltv AS rar
                                    FROM customer_segments cs JOIN fact_subscription f USING (customer_id)""")).all()
    for cid, p, ltv, rar in rows:
        assert p * ltv == rar                                   # exact Decimal arithmetic in SQL
        assert df.loc[cid, "revenue_at_risk"] == pytest.approx(float(rar), abs=1e-6)


def test_active_customer_headlines_and_savings(matrix):
    df, _ = matrix
    active = df[~df["churn"]]
    assert round(rm.matrix_summary(active.reset_index()).loc["Total", "revenue_at_risk"]) == EXPECTED["active_revenue_at_risk"]
    offer = active[active["retention_action"] == rm.RETENTION_OFFER]
    assert len(offer) == EXPECTED["active_retention_offer_customers"]
    savings = rm.projected_savings(active)
    by_hand = [rate * (offer.churn_probability * offer.estimated_ltv).sum() for rate in (0.1, 0.2, 0.3)]
    assert np.allclose(savings["projected_revenue_saved"], by_hand)
    assert tuple(round(v) for v in by_hand) == EXPECTED["active_savings_10_20_30"]
