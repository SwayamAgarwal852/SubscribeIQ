"""ETL: load the raw IBM Telco churn CSV into the SubscribeIQ star schema.

Usage (from the project root):
    python database/seed.py

Steps:
    1. Apply database/schema.sql (CREATE TABLE IF NOT EXISTS, so safe to re-run).
    2. Read and clean the raw CSV.
    3. Derive num_services and estimated_ltv.
    4. Upsert into dim_customer, dim_service, dim_contract, fact_subscription.

Idempotency: every table is upserted on customer_id (INSERT ... ON CONFLICT DO UPDATE)
inside a single transaction. Re-running never duplicates rows, and it does not touch
customer_segments, so analytics results from later phases survive a re-seed.
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db_connection import get_engine  # noqa: E402

RAW_CSV = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SCHEMA_SQL = PROJECT_ROOT / "database" / "schema.sql"

ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
TABLES = ["dim_customer", "dim_service", "dim_contract", "fact_subscription", "customer_segments"]


def yes_no_to_bool(series: pd.Series) -> pd.Series:
    """Map a strict 'Yes'/'No' column to booleans, failing loudly on any other value.

    Args:
        series: Column containing only 'Yes' or 'No'.

    Returns:
        Boolean Series.

    Raises:
        ValueError: If the column contains a value other than 'Yes' or 'No'.
    """
    unexpected = set(series.unique()) - {"Yes", "No"}
    if unexpected:
        raise ValueError(f"{series.name}: unexpected values {unexpected}")
    return series.eq("Yes")


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric, setting blanks for tenure-0 customers to 0.

    Blank TotalCharges rows are brand-new customers (tenure = 0) who have not been
    billed yet, so 0 is their true value rather than an imputed estimate. Any blank
    row with tenure > 0 would mean a different problem, so that raises instead.

    Args:
        df: Raw dataframe with TotalCharges as strings.

    Returns:
        Copy of df with TotalCharges as float.

    Raises:
        ValueError: If a non-numeric TotalCharges appears on a row with tenure > 0.
    """
    df = df.copy()
    numeric = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    blank = numeric.isna()
    if (df.loc[blank, "tenure"] != 0).any():
        raise ValueError("Non-numeric TotalCharges found on rows with tenure > 0")
    df["TotalCharges"] = numeric.fillna(0.0)
    print(f"[clean] TotalCharges: {blank.sum()} blank value(s) on tenure-0 rows set to 0.00")
    return df


def count_services(df: pd.DataFrame) -> pd.Series:
    """Count active services per customer (range 0-9).

    Counted: PhoneService, MultipleLines, InternetService (DSL or Fiber optic),
    and the six internet add-ons, each when active ('Yes').

    Args:
        df: Dataframe with the raw service columns.

    Returns:
        Integer Series of active-service counts.
    """
    count = (df["PhoneService"] == "Yes").astype(int)
    count += (df["MultipleLines"] == "Yes").astype(int)
    count += df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
    for col in ADDON_COLUMNS:
        count += (df[col] == "Yes").astype(int)
    return count


def transform(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Clean the raw data and split it into one dataframe per warehouse table.

    Args:
        raw: Dataframe read directly from the raw CSV.

    Returns:
        Mapping of table name to a dataframe whose columns match that table.
    """
    df = clean_total_charges(raw)
    df["num_services"] = count_services(df)
    df["estimated_ltv"] = (df["MonthlyCharges"] * df["tenure"]).round(2)

    dim_customer = pd.DataFrame({
        "customer_id": df["customerID"],
        "gender": df["gender"],
        "senior_citizen": df["SeniorCitizen"].astype(bool),
        "partner": yes_no_to_bool(df["Partner"]),
        "dependents": yes_no_to_bool(df["Dependents"]),
    })
    dim_service = pd.DataFrame({
        "customer_id": df["customerID"],
        "phone_service": yes_no_to_bool(df["PhoneService"]),
        "multiple_lines": df["MultipleLines"],
        "internet_service": df["InternetService"],
        "online_security": df["OnlineSecurity"],
        "online_backup": df["OnlineBackup"],
        "device_protection": df["DeviceProtection"],
        "tech_support": df["TechSupport"],
        "streaming_tv": df["StreamingTV"],
        "streaming_movies": df["StreamingMovies"],
    })
    dim_contract = pd.DataFrame({
        "customer_id": df["customerID"],
        "contract_type": df["Contract"],
        "payment_method": df["PaymentMethod"],
        "paperless_billing": yes_no_to_bool(df["PaperlessBilling"]),
    })
    fact_subscription = pd.DataFrame({
        "customer_id": df["customerID"],
        "tenure_months": df["tenure"],
        "monthly_charges": df["MonthlyCharges"],
        "total_charges": df["TotalCharges"],
        "churn": yes_no_to_bool(df["Churn"]),
        "num_services": df["num_services"],
        "estimated_ltv": df["estimated_ltv"],
    })
    return {
        "dim_customer": dim_customer,
        "dim_service": dim_service,
        "dim_contract": dim_contract,
        "fact_subscription": fact_subscription,
    }


def upsert(conn, table: Table, frame: pd.DataFrame) -> None:
    """Insert rows, updating existing ones that share the same customer_id.

    Args:
        conn: Open SQLAlchemy connection (inside a transaction).
        table: Reflected target table.
        frame: Rows to load; columns must match the table's columns.
    """
    records = frame.to_dict(orient="records")
    stmt = insert(table).values(records)
    update_cols = {c: stmt.excluded[c] for c in frame.columns if c != "customer_id"}
    conn.execute(stmt.on_conflict_do_update(index_elements=["customer_id"], set_=update_cols))


def table_counts(conn) -> dict[str, int]:
    """Return the current row count of every SubscribeIQ table."""
    return {t: conn.execute(text(f"SELECT COUNT(*) FROM {t}")).scalar_one() for t in TABLES}


def main() -> None:
    """Run the full ETL and print a per-table row summary."""
    engine = get_engine()

    raw = pd.read_csv(RAW_CSV, dtype={"TotalCharges": str})
    print(f"[extract] {len(raw)} rows, {raw.shape[1]} columns from {RAW_CSV.name}")
    if raw["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values in raw CSV")

    frames = transform(raw)

    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL.read_text())
        before = table_counts(conn)

        metadata = MetaData()
        for name, frame in frames.items():  # dim_customer first: the others reference it
            upsert(conn, Table(name, metadata, autoload_with=conn), frame)

        after = table_counts(conn)

    print("\n[load] Summary")
    print(f"  {'table':<20}{'rows processed':>16}{'rows before':>14}{'rows after':>13}")
    for t in TABLES:
        processed = len(frames[t]) if t in frames else 0
        print(f"  {t:<20}{processed:>16}{before[t]:>14}{after[t]:>13}")


if __name__ == "__main__":
    main()
