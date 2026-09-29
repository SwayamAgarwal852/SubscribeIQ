"""Retention strategy matrix: combine churn risk with customer value into four actions.

Inputs (all from PostgreSQL):
    churn_probability  calibrated out-of-fold churn probability (Phase 4, customer_segments)
    estimated_ltv      forward-looking 12-month value = monthly_charges x 12 (fact_subscription)

Quadrants use median splits on both axes (a customer is "high" when strictly above the median):

                     Low churn risk            High churn risk
    High LTV         Early Access / Upsell     Retention Offer
    Low LTV          Nurture                   Let Go

Revenue at risk for a customer = churn_probability x estimated_ltv, i.e. the expected
12-month revenue lost to churn. Because the probabilities are calibrated, these sums are
expected values in dollars.

Run as a script to assign actions, write them to customer_segments and print the report:
    python -m src.retention_matrix
"""

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

RETENTION_OFFER = "Retention Offer"
EARLY_ACCESS = "Early Access / Upsell"
LET_GO = "Let Go"
NURTURE = "Nurture"
ACTIONS = [RETENTION_OFFER, EARLY_ACCESS, LET_GO, NURTURE]

ACTION_DESCRIPTIONS = {
    RETENTION_OFFER: "High value, high risk: proactive retention offer (e.g. contract upgrade "
                     "incentive, Security/Tech Support bundle).",
    EARLY_ACCESS: "High value, low risk: reward loyalty with early access and upsell "
                  "opportunities; no discount needed.",
    LET_GO: "Low value, high risk: do not spend retention budget; low-cost automated "
            "touchpoints only.",
    NURTURE: "Low value, low risk: grow value over time with low-cost engagement and "
             "add-on recommendations.",
}

DEFAULT_SUCCESS_RATES = (0.10, 0.20, 0.30)


def load_matrix_inputs(engine: Engine) -> pd.DataFrame:
    """Load churn probability, LTV and context columns for every customer.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.

    Returns:
        One row per customer.

    Raises:
        ValueError: If any customer is missing a churn probability (run src.churn_model first).
    """
    query = """
        SELECT cs.customer_id,
               cs.segment_name,
               cs.churn_probability::float AS churn_probability,
               f.estimated_ltv::float      AS estimated_ltv,
               f.monthly_charges::float    AS monthly_charges,
               f.tenure_months,
               f.churn,
               k.contract_type
        FROM customer_segments cs
        JOIN fact_subscription f USING (customer_id)
        JOIN dim_contract k      USING (customer_id)
        ORDER BY cs.customer_id
    """
    df = pd.read_sql(text(query), engine)
    if df["churn_probability"].isna().any():
        raise ValueError("Missing churn_probability; run `python -m src.churn_model` first")
    return df


def assign_actions(df: pd.DataFrame, ltv_threshold: float | None = None,
                   risk_threshold: float | None = None) -> tuple[pd.DataFrame, dict]:
    """Place each customer in a quadrant and assign its retention action.

    Args:
        df: Output of load_matrix_inputs.
        ltv_threshold: LTV cut-off; defaults to the median estimated_ltv.
        risk_threshold: Churn-probability cut-off; defaults to the median churn_probability.

    Returns:
        Copy of df with high_ltv, high_risk, revenue_at_risk and retention_action columns,
        plus a dict of the thresholds used.
    """
    ltv_threshold = df["estimated_ltv"].median() if ltv_threshold is None else ltv_threshold
    risk_threshold = df["churn_probability"].median() if risk_threshold is None else risk_threshold

    out = df.copy()
    out["high_ltv"] = out["estimated_ltv"] > ltv_threshold
    out["high_risk"] = out["churn_probability"] > risk_threshold
    out["revenue_at_risk"] = out["churn_probability"] * out["estimated_ltv"]
    out["retention_action"] = np.select(
        [out.high_ltv & out.high_risk, out.high_ltv & ~out.high_risk,
         ~out.high_ltv & out.high_risk],
        [RETENTION_OFFER, EARLY_ACCESS, LET_GO],
        default=NURTURE,
    )
    return out, {"ltv_threshold": ltv_threshold, "risk_threshold": risk_threshold}


def matrix_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each action group: size, risk, value, revenue at risk and actual churn.

    The actual churn rate is included as a sanity check that high-risk quadrants really
    contain more churners.

    Args:
        df: Output of assign_actions.

    Returns:
        One row per action (in ACTIONS order) plus a Total row.
    """
    grouped = df.groupby("retention_action").agg(
        customers=("customer_id", "size"),
        avg_churn_probability=("churn_probability", "mean"),
        actual_churn_rate=("churn", "mean"),
        avg_monthly_charges=("monthly_charges", "mean"),
        total_ltv=("estimated_ltv", "sum"),
        revenue_at_risk=("revenue_at_risk", "sum"),
    ).reindex(ACTIONS)
    total = pd.DataFrame({
        "customers": [len(df)],
        "avg_churn_probability": [df["churn_probability"].mean()],
        "actual_churn_rate": [df["churn"].mean()],
        "avg_monthly_charges": [df["monthly_charges"].mean()],
        "total_ltv": [df["estimated_ltv"].sum()],
        "revenue_at_risk": [df["revenue_at_risk"].sum()],
    }, index=["Total"])
    summary = pd.concat([grouped, total])
    summary["share_of_customers_pct"] = 100 * summary["customers"] / len(df)
    summary["share_of_revenue_at_risk_pct"] = 100 * summary["revenue_at_risk"] / df["revenue_at_risk"].sum()
    return summary


def projected_savings(df: pd.DataFrame, success_rates=DEFAULT_SUCCESS_RATES,
                      action: str = RETENTION_OFFER) -> pd.DataFrame:
    """Projected 12-month revenue saved if a share of an action group's expected churn is prevented.

    savings = success_rate x sum(churn_probability x estimated_ltv) over the action group,
    i.e. the rate applies to the churn that would otherwise happen, not to every customer.

    Args:
        df: Output of assign_actions.
        success_rates: Fractions of expected churn prevented, one scenario each.
        action: Action group the offer targets.

    Returns:
        One row per scenario with customers saved (expected) and revenue saved.
    """
    group = df[df["retention_action"] == action]
    expected_churners = group["churn_probability"].sum()
    at_risk = group["revenue_at_risk"].sum()
    return pd.DataFrame([{
        "success_rate_pct": 100 * rate,
        "customers_targeted": len(group),
        "expected_churners": expected_churners,
        "expected_customers_saved": rate * expected_churners,
        "revenue_at_risk": at_risk,
        "projected_revenue_saved": rate * at_risk,
    } for rate in success_rates]).set_index("success_rate_pct")


def segment_action_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Count customers by Phase 3 segment and retention action.

    Args:
        df: Output of assign_actions.

    Returns:
        Crosstab with segments as rows and actions as columns (plus totals).
    """
    return pd.crosstab(df["segment_name"], df["retention_action"], margins=True,
                       margins_name="Total").reindex(columns=ACTIONS + ["Total"])


def cost_based_flags(df: pd.DataFrame, cost_ratio: float) -> dict:
    """Customers worth contacting under a cost-based rule, for the dashboard's adjustable view.

    With calibrated probabilities, contacting a customer is worthwhile when
    churn_probability > 1 / (1 + r), where r = cost of a missed churner / cost of an offer.
    This is kept separate from the median-split matrix.

    Args:
        df: Output of assign_actions.
        cost_ratio: r, how many times more a missed churner costs than a wasted offer.

    Returns:
        Dict with the threshold, number flagged, and their revenue at risk.

    Raises:
        ValueError: If cost_ratio is not positive.
    """
    if cost_ratio <= 0:
        raise ValueError("cost_ratio must be positive")
    threshold = 1 / (1 + cost_ratio)
    flagged = df[df["churn_probability"] > threshold]
    return {"threshold": threshold, "customers_flagged": len(flagged),
            "share_flagged_pct": 100 * len(flagged) / len(df),
            "revenue_at_risk_flagged": flagged["revenue_at_risk"].sum()}


def write_actions(engine: Engine, df: pd.DataFrame) -> int:
    """Write retention_action to customer_segments.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.
        df: Output of assign_actions.

    Returns:
        Number of rows with a non-null retention_action after the update.
    """
    stmt = text("UPDATE customer_segments SET retention_action = :a WHERE customer_id = :cid")
    stmt = stmt.bindparams(bindparam("a"), bindparam("cid"))
    records = [{"cid": c, "a": a} for c, a in zip(df["customer_id"], df["retention_action"])]
    with engine.begin() as conn:
        conn.execute(stmt, records)
        return conn.execute(text(
            "SELECT COUNT(retention_action) FROM customer_segments")).scalar_one()


def main() -> None:
    """Assign actions, persist them and print the matrix and business-impact report."""
    from src.db_connection import get_engine

    engine = get_engine()
    df, thresholds = assign_actions(load_matrix_inputs(engine))
    rows = write_actions(engine, df)

    pd.set_option("display.width", 200)
    print(f"Thresholds: LTV > ${thresholds['ltv_threshold']:,.2f} (median), "
          f"churn probability > {thresholds['risk_threshold']:.4f} (median)")
    print(f"retention_action written for {rows} customers\n")
    print(matrix_summary(df).round(3).to_string(), "\n")
    print(projected_savings(df).round(2).to_string(), "\n")
    print(segment_action_crosstab(df).to_string())


if __name__ == "__main__":
    main()
