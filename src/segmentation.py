"""RFM scoring and KMeans behavioural segmentation.

RFM proxies for a subscription business (the data has no purchase timestamps):
    Recency   -> tenure_months   (reversed from classic RFM: 5 = most established customer)
    Frequency -> num_services    (number of active services, 1-9)
    Monetary  -> total_charges   (actual billed revenue to date)

Known limitation: total_charges is roughly monthly_charges x tenure, so the M score is
strongly correlated with R (Spearman ~0.89). Clusters therefore lean on tenure.

Run as a script to score, cluster and write results to customer_segments:
    python -m src.segmentation
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import MetaData, Table, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

SCORE_COLUMNS = ["rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score"]
N_CLUSTERS = 4
RANDOM_STATE = 42

SEGMENT_DESCRIPTIONS = {
    "New & Uncommitted": (
        "Customers in their first months (median tenure ~4 mo) with few services and low "
        "billed revenue so far; almost all on month-to-month contracts. Highest churn risk."
    ),
    "Flexible Fiber Users": (
        "Mid-tenure, high-spend customers (median ~29 mo, 5 services, ~$84/mo) who have not "
        "committed to a long contract: ~68% month-to-month and ~60% on fiber optic. Fiber alone "
        "is not the differentiator (Established Power Users are ~61% fiber); fiber without a "
        "contract is. Second-highest churn."
    ),
    "Loyal Basics": (
        "Long-tenure customers (median ~45 mo) on a single low-cost service; most have no "
        "internet (phone-only). Very low churn but limited revenue."
    ),
    "Established Power Users": (
        "The longest-tenured customers (median ~64 mo) with the most services and the highest "
        "lifetime revenue; mostly on one- or two-year contracts. Low churn, highest value."
    ),
}

# Recommended strategy per segment, grounded in the EDA (notebook 01) and churn-driver
# findings (notebook 03): contract commitment, first-year risk, fiber + month-to-month risk,
# and the protective effect of Online Security / Tech Support.
SEGMENT_STRATEGIES = {
    "New & Uncommitted": (
        "Focus on onboarding and the first 12 months, where most churn happens: welcome "
        "check-ins, early service reviews, and an incentive to move from month-to-month to an "
        "annual contract once the customer is settled."
    ),
    "Flexible Fiber Users": (
        "Offer contract upgrades to high-spend fiber customers still on month-to-month, and "
        "bundle Online Security / Tech Support (the add-ons most associated with lower churn). "
        "Review fiber price-to-value, since discounts alone may not address it."
    ),
    "Established Power Users": (
        "Protect rather than discount: loyalty recognition, early access to new services, and "
        "priority support. Watch the minority on month-to-month contracts."
    ),
    "Loyal Basics": (
        "Stable and low-cost to keep. Grow value with careful upsell (e.g. a first internet "
        "service or add-on) without disrupting a relationship that already works."
    ),
}


def load_rfm_inputs(engine: Engine) -> pd.DataFrame:
    """Load the RFM source columns (plus churn, for profiling only) from the warehouse.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.

    Returns:
        One row per customer with customer_id, tenure_months, num_services,
        total_charges, monthly_charges and churn.
    """
    query = """
        SELECT customer_id, tenure_months, num_services,
               total_charges::float   AS total_charges,
               monthly_charges::float AS monthly_charges,
               churn
        FROM fact_subscription
        ORDER BY customer_id
    """
    return pd.read_sql(text(query), engine)


def quintile_score(series: pd.Series) -> pd.Series:
    """Score a series 1-5 by quintile using plain pd.qcut (higher value -> higher score).

    Ties always receive the same score. For a discrete column such as num_services this
    means bins are not equally sized; duplicate edges are dropped rather than splitting ties.

    Args:
        series: Numeric values to score.

    Returns:
        Integer scores starting at 1.
    """
    return pd.qcut(series, 5, labels=False, duplicates="drop").astype(int) + 1


def score_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Add R, F and M quintile scores and the concatenated rfm_combined string (e.g. "534").

    Args:
        df: Output of load_rfm_inputs.

    Returns:
        Copy of df with rfm_recency_score, rfm_frequency_score, rfm_monetary_score
        and rfm_combined columns.
    """
    df = df.copy()
    df["rfm_recency_score"] = quintile_score(df["tenure_months"])
    df["rfm_frequency_score"] = quintile_score(df["num_services"])
    df["rfm_monetary_score"] = quintile_score(df["total_charges"])
    df["rfm_combined"] = df[SCORE_COLUMNS].astype(str).agg("".join, axis=1)
    return df


def standardize_scores(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Standardise the three RFM scores to zero mean and unit variance for KMeans.

    Args:
        df: Dataframe containing the RFM score columns.

    Returns:
        The standardised feature matrix and the fitted scaler.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df[SCORE_COLUMNS]), scaler


def evaluate_k(X: np.ndarray, k_values=range(2, 11), seeds=range(5)) -> pd.DataFrame:
    """Compute elbow, silhouette and seed-stability diagnostics for each candidate K.

    Stability is the mean adjusted Rand index between the labels from the first seed and
    each other seed: 1.0 means KMeans finds identical clusters regardless of initialisation.

    Args:
        X: Standardised feature matrix.
        k_values: Candidate numbers of clusters.
        seeds: Random seeds used for the stability check.

    Returns:
        Dataframe indexed by k with inertia, silhouette and stability_ari columns.
    """
    rows = []
    for k in k_values:
        runs = [KMeans(n_clusters=k, random_state=s, n_init=10).fit(X) for s in seeds]
        base = runs[0]
        ari = np.mean([adjusted_rand_score(base.labels_, r.labels_) for r in runs[1:]])
        rows.append({
            "k": k,
            "inertia": base.inertia_,
            "silhouette": silhouette_score(X, base.labels_),
            "stability_ari": ari,
        })
    return pd.DataFrame(rows).set_index("k")


def fit_kmeans(X: np.ndarray, n_clusters: int = N_CLUSTERS) -> KMeans:
    """Fit the final KMeans model.

    Args:
        X: Standardised feature matrix.
        n_clusters: Number of clusters (K=4 chosen in notebook 02).

    Returns:
        The fitted KMeans model.
    """
    return KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit(X)


def name_clusters(df: pd.DataFrame) -> dict[int, str]:
    """Map cluster labels to business names using each cluster's average RFM profile.

    Rules (applied in order, each to the clusters not yet named):
        1. Lowest average recency score        -> "New & Uncommitted"
        2. Lowest average frequency score      -> "Loyal Basics"
        3. Highest average recency score       -> "Established Power Users"
        4. Remaining cluster                   -> "Flexible Fiber Users"

    Naming by profile, not by label number, keeps names correct if KMeans relabels clusters.

    Args:
        df: Dataframe with cluster_label and the RFM score columns (exactly 4 clusters).

    Returns:
        Mapping of cluster_label to segment name.

    Raises:
        ValueError: If there are not exactly 4 clusters.
    """
    profile = df.groupby("cluster_label")[SCORE_COLUMNS].mean()
    if len(profile) != 4:
        raise ValueError(f"Naming rules expect 4 clusters, got {len(profile)}")

    names = {}
    remaining = profile.copy()

    def take(label, name):
        names[int(label)] = name
        return remaining.drop(index=label)

    remaining = take(remaining["rfm_recency_score"].idxmin(), "New & Uncommitted")
    remaining = take(remaining["rfm_frequency_score"].idxmin(), "Loyal Basics")
    remaining = take(remaining["rfm_recency_score"].idxmax(), "Established Power Users")
    take(remaining.index[0], "Flexible Fiber Users")
    return names


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full segmentation: score RFM, standardise, cluster with K=4 and name clusters.

    Args:
        df: Output of load_rfm_inputs.

    Returns:
        Scored dataframe with cluster_label and segment_name columns added.
    """
    scored = score_rfm(df)
    X, _ = standardize_scores(scored)
    scored["cluster_label"] = fit_kmeans(X).labels_
    scored["segment_name"] = scored["cluster_label"].map(name_clusters(scored))
    return scored


def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each segment: size, average RFM scores, typical values and churn rate.

    Args:
        df: Output of segment_customers.

    Returns:
        One row per segment, ordered by churn rate (highest first).
    """
    profile = df.groupby("segment_name").agg(
        customers=("customer_id", "size"),
        avg_recency_score=("rfm_recency_score", "mean"),
        avg_frequency_score=("rfm_frequency_score", "mean"),
        avg_monetary_score=("rfm_monetary_score", "mean"),
        median_tenure_months=("tenure_months", "median"),
        median_services=("num_services", "median"),
        median_monthly_charges=("monthly_charges", "median"),
        median_total_charges=("total_charges", "median"),
        churn_rate_pct=("churn", "mean"),
    )
    profile["share_pct"] = 100 * profile["customers"] / profile["customers"].sum()
    profile["churn_rate_pct"] *= 100
    return profile.sort_values("churn_rate_pct", ascending=False).round(2)


def write_segments(engine: Engine, df: pd.DataFrame) -> int:
    """Upsert RFM scores, cluster labels and segment names into customer_segments.

    Only the Phase 3 columns are written; churn_probability and retention_action
    (Phases 4-5) are left untouched on existing rows.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.
        df: Output of segment_customers.

    Returns:
        Number of rows in customer_segments after the write.
    """
    cols = ["customer_id", *SCORE_COLUMNS, "rfm_combined", "cluster_label", "segment_name"]
    records = df[cols].to_dict(orient="records")
    with engine.begin() as conn:
        table = Table("customer_segments", MetaData(), autoload_with=conn)
        stmt = insert(table).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["customer_id"],
            set_={c: stmt.excluded[c] for c in cols if c != "customer_id"},
        )
        conn.execute(stmt)
        return conn.execute(text("SELECT COUNT(*) FROM customer_segments")).scalar_one()


def main() -> None:
    """Score, segment and persist all customers, then print the segment profile."""
    from src.db_connection import get_engine

    engine = get_engine()
    segmented = segment_customers(load_rfm_inputs(engine))
    rows = write_segments(engine, segmented)
    print(f"customer_segments rows: {rows}")
    print(profile_segments(segmented).to_string())


if __name__ == "__main__":
    main()
