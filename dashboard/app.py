"""SubscribeIQ Streamlit dashboard.

Every number on every page is queried live from PostgreSQL (cached for a few minutes, with a
refresh button). The only file read from disk is the trained model (models/churn_model.pkl),
which the Churn Drivers page uses for live what-if predictions.

Run from the project root:
    streamlit run dashboard/app.py

Adding a page: write a function that renders it and register it in PAGES at the bottom.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import churn_model as cm  # noqa: E402
from src import retention_matrix as rm  # noqa: E402
from src import segmentation as seg  # noqa: E402
from src.db_connection import get_engine  # noqa: E402

st.set_page_config(page_title="SubscribeIQ", page_icon="📉", layout="wide")

CACHE_TTL_SECONDS = 300

# Fixed categorical colour order (never cycled); segments ordered by churn rate
SEGMENT_ORDER = ["New & Uncommitted", "Flexible Fiber Users", "Established Power Users", "Loyal Basics"]
SEGMENT_COLORS = dict(zip(SEGMENT_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
SEGMENT_SYMBOLS = dict(zip(SEGMENT_ORDER, ["circle", "diamond", "square", "triangle-up"]))
ACTION_COLORS = dict(zip(rm.ACTIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
PRIMARY = "#2a78d6"
MUTED = "#9a9993"

ADDON_COLUMNS = ["online_security", "online_backup", "device_protection",
                 "tech_support", "streaming_tv", "streaming_movies"]


# --------------------------------------------------------------------------- data access
@st.cache_resource
def engine():
    """Shared SQLAlchemy engine for the session."""
    return get_engine()


@st.cache_resource
def model_artifact():
    """The saved churn model artifact (calibrated pipeline + metadata)."""
    return cm.load_model()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only SQL query against PostgreSQL.

    Args:
        sql: SQL text with :named parameters.
        params: Tuple of (name, value) pairs (a tuple so the result can be cached).

    Returns:
        Query result as a DataFrame.
    """
    return pd.read_sql(text(sql), engine(), params=dict(params))


CUSTOMER_SQL = """
    SELECT c.customer_id, c.gender, c.senior_citizen, c.partner, c.dependents,
           s.phone_service, s.multiple_lines, s.internet_service, s.online_security,
           s.online_backup, s.device_protection, s.tech_support, s.streaming_tv,
           s.streaming_movies, k.contract_type, k.payment_method, k.paperless_billing,
           f.tenure_months, f.monthly_charges::float AS monthly_charges,
           f.total_charges::float AS total_charges, f.num_services,
           f.estimated_ltv::float AS estimated_ltv, f.churn,
           cs.rfm_recency_score, cs.rfm_frequency_score, cs.rfm_monetary_score,
           cs.rfm_combined, cs.segment_name, cs.churn_probability::float AS churn_probability,
           cs.retention_action,
           (cs.churn_probability * f.estimated_ltv)::float AS revenue_at_risk
    FROM dim_customer c
    JOIN dim_service s        USING (customer_id)
    JOIN dim_contract k       USING (customer_id)
    JOIN fact_subscription f  USING (customer_id)
    JOIN customer_segments cs USING (customer_id)
"""


def all_customers() -> pd.DataFrame:
    """Every customer with dimensions, facts, segment, probability and action."""
    return query(CUSTOMER_SQL + " ORDER BY c.customer_id")


def one_customer(customer_id: str) -> pd.DataFrame:
    """A single customer's row (empty if the ID does not exist)."""
    return query(CUSTOMER_SQL + " WHERE c.customer_id = :cid", (("cid", customer_id),))


def to_model_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Convert warehouse rows to the raw feature format the model was trained on.

    Booleans become "Yes"/"No", matching src.churn_model.load_modeling_data.
    """
    out = rows.copy()
    for col in ["senior_citizen", "partner", "dependents", "phone_service", "paperless_billing"]:
        out[col] = out[col].map({True: "Yes", False: "No"})
    return out[cm.FEATURES]


def money(x: float, decimals: int = 0) -> str:
    """Format a number as dollars."""
    return f"${x:,.{decimals}f}"


# --------------------------------------------------------------------------- charts
def risk_gauge(probability: float, title: str) -> go.Figure:
    """Semicircular gauge for a churn probability, banded at the matrix and cost thresholds."""
    median = query("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY churn_probability)::float AS m "
                   "FROM customer_segments")["m"].iloc[0]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": "#0d366b", "thickness": 0.3},
            "steps": [
                {"range": [0, median * 100], "color": "#cde2fb"},
                {"range": [median * 100, 50], "color": "#86b6ef"},
                {"range": [50, 100], "color": "#3987e5"},
            ],
            "threshold": {"line": {"color": "#d03b3b", "width": 3}, "value": median * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def style(fig: go.Figure, height: int = 380, legend_top: bool = False) -> go.Figure:
    """Shared, recessive chart styling.

    legend_top: horizontal legend between the title and the plot, with extra top margin so the
    two never collide.
    """
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend_title_text="", hoverlabel=dict(font_size=12))
    if legend_top:
        fig.update_layout(margin_t=95, title=dict(y=0.98, yref="container", yanchor="top"),
                          legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    return fig


# --------------------------------------------------------------------------- pages
def page_overview():
    st.title("Overview")
    st.caption("Portfolio health at a glance: all 7,043 customers in the warehouse.")

    kpi = query("""
        SELECT COUNT(*) AS customers,
               AVG(churn::int)::float AS churn_rate,
               AVG(estimated_ltv)::float AS avg_ltv,
               SUM(monthly_charges)::float AS mrr
        FROM fact_subscription
    """).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{int(kpi.customers):,}")
    c2.metric("Churn rate", f"{kpi.churn_rate:.1%}")
    c3.metric("Avg 12-month LTV", money(kpi.avg_ltv))
    c4.metric("Monthly recurring revenue", money(kpi.mrr))

    segments = query("""
        SELECT cs.segment_name,
               COUNT(*)                          AS customers,
               AVG(f.churn::int)::float          AS churn_rate,
               AVG(cs.churn_probability)::float  AS avg_churn_probability,
               AVG(f.monthly_charges)::float     AS avg_monthly_charges
        FROM customer_segments cs JOIN fact_subscription f USING (customer_id)
        GROUP BY cs.segment_name
    """).set_index("segment_name").reindex(SEGMENT_ORDER).reset_index()

    left, right = st.columns(2)
    with left:
        fig = px.pie(segments, names="segment_name", values="customers", hole=0.55,
                     color="segment_name", color_discrete_map=SEGMENT_COLORS,
                     category_orders={"segment_name": SEGMENT_ORDER},
                     title="Customers by segment")
        fig.update_traces(textinfo="percent", sort=False,
                          marker=dict(line=dict(color="white", width=2)),
                          hovertemplate="%{label}<br>%{value:,} customers (%{percent})<extra></extra>")
        st.plotly_chart(style(fig), width="stretch")
    with right:
        fig = px.bar(segments.sort_values("churn_rate"), x="churn_rate", y="segment_name",
                     orientation="h", title="Churn rate by segment",
                     text=segments.sort_values("churn_rate").churn_rate.map("{:.1%}".format))
        fig.update_traces(marker_color=PRIMARY, textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>Churn rate %{x:.1%}<extra></extra>")
        fig.add_vline(x=kpi.churn_rate, line_dash="dash", line_color=MUTED,
                      annotation_text=f"overall {kpi.churn_rate:.1%}", annotation_position="top")
        fig.update_layout(xaxis_tickformat=".0%", xaxis_title="", yaxis_title="",
                          xaxis_range=[0, max(0.55, segments.churn_rate.max() * 1.25)])
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("Segment summary")
    table = segments.rename(columns={
        "segment_name": "Segment", "customers": "Customers", "churn_rate": "Churn rate",
        "avg_churn_probability": "Avg predicted churn", "avg_monthly_charges": "Avg monthly charges"})
    st.dataframe(table.style.format({"Customers": "{:,}", "Churn rate": "{:.1%}",
                                     "Avg predicted churn": "{:.1%}", "Avg monthly charges": "${:,.2f}"}),
                 hide_index=True, width="stretch")


def page_segment_explorer():
    st.title("Segment Explorer")
    segment = st.selectbox("Segment", SEGMENT_ORDER)

    profile = query("""
        SELECT cs.segment_name,
               COUNT(*)                                AS customers,
               AVG(cs.rfm_recency_score)::float        AS recency,
               AVG(cs.rfm_frequency_score)::float      AS frequency,
               AVG(cs.rfm_monetary_score)::float       AS monetary,
               AVG(f.churn::int)::float                AS churn_rate,
               AVG(f.monthly_charges)::float           AS avg_monthly,
               AVG(f.tenure_months)::float             AS avg_tenure,
               AVG(f.estimated_ltv)::float             AS avg_ltv
        FROM customer_segments cs JOIN fact_subscription f USING (customer_id)
        GROUP BY cs.segment_name
    """).set_index("segment_name")
    row = profile.loc[segment]
    overall = query("""
        SELECT AVG(rfm_recency_score)::float AS recency, AVG(rfm_frequency_score)::float AS frequency,
               AVG(rfm_monetary_score)::float AS monetary
        FROM customer_segments
    """).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{int(row.customers):,}", f"{row.customers / profile.customers.sum():.1%} of base",
              delta_color="off")
    c2.metric("Churn rate", f"{row.churn_rate:.1%}")
    c3.metric("Avg monthly charges", money(row.avg_monthly, 2))
    c4.metric("Avg tenure", f"{row.avg_tenure:.0f} months")

    st.markdown(f"**Profile.** {seg.SEGMENT_DESCRIPTIONS[segment]}")
    st.info(f"**Recommended strategy.** {seg.SEGMENT_STRATEGIES[segment]}")

    left, right = st.columns(2)
    with left:
        rfm = pd.DataFrame({
            "dimension": ["Recency (tenure)", "Frequency (services)", "Monetary (total billed)"] * 2,
            "score": [row.recency, row.frequency, row.monetary,
                      overall.recency, overall.frequency, overall.monetary],
            "group": [segment] * 3 + ["All customers"] * 3,
        })
        fig = px.bar(rfm, x="dimension", y="score", color="group", barmode="group",
                     title="Average RFM scores (1–5)",
                     color_discrete_map={segment: SEGMENT_COLORS[segment], "All customers": MUTED})
        fig.update_traces(hovertemplate="%{x}<br>%{y:.2f}<extra></extra>")
        fig.update_layout(yaxis_range=[0, 5.2], xaxis_title="", yaxis_title="Score")
        st.plotly_chart(style(fig, legend_top=True), width="stretch")
    with right:
        actions = query("""
            SELECT retention_action, COUNT(*) AS customers
            FROM customer_segments WHERE segment_name = :seg GROUP BY retention_action
        """, (("seg", segment),)).set_index("retention_action").reindex(rm.ACTIONS).fillna(0).reset_index()
        fig = px.bar(actions, x="customers", y="retention_action", orientation="h",
                     title="Retention actions within this segment",
                     text=actions.customers.astype(int).map("{:,}".format))
        fig.update_traces(marker_color=SEGMENT_COLORS[segment], textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}: %{x:,} customers<extra></extra>")
        fig.update_layout(xaxis_title="Customers", yaxis_title="",
                          yaxis=dict(categoryorder="array", categoryarray=rm.ACTIONS[::-1]))
        st.plotly_chart(style(fig), width="stretch")

    customers = query("""
        SELECT cs.customer_id, cs.segment_name, f.tenure_months,
               f.monthly_charges::float AS monthly_charges, f.churn
        FROM customer_segments cs JOIN fact_subscription f USING (customer_id)
    """)
    customers["Status"] = customers.churn.map({True: "Churned", False: "Active"})
    fig = go.Figure()
    for name in SEGMENT_ORDER:
        sub = customers[customers.segment_name == name]
        selected = name == segment
        fig.add_trace(go.Scattergl(
            x=sub.tenure_months, y=sub.monthly_charges, mode="markers", name=name,
            marker=dict(color=SEGMENT_COLORS[name], symbol=SEGMENT_SYMBOLS[name],
                        size=6 if selected else 4, opacity=0.75 if selected else 0.12,
                        line=dict(width=0)),
            customdata=sub[["customer_id", "Status"]],
            hovertemplate=(f"<b>{name}</b><br>%{{customdata[0]}} (%{{customdata[1]}})<br>"
                           "Tenure %{x} mo · $%{y:.2f}/mo<extra></extra>"),
        ))
    fig.update_layout(title=f"Monthly charges vs tenure ({segment} highlighted)",
                      xaxis_title="Tenure (months)", yaxis_title="Monthly charges ($)")
    st.plotly_chart(style(fig, 460, legend_top=True), width="stretch")


def page_customer_lookup():
    st.title("Customer Lookup")
    customer_id = st.text_input("Customer ID", value="7590-VHVEG",
                                help="Format like 7590-VHVEG. Every customer in the warehouse can be looked up.")
    customer_id = customer_id.strip().upper()
    if not customer_id:
        st.info("Enter a customer ID to begin.")
        return
    rows = one_customer(customer_id)
    if rows.empty:
        st.error(f"No customer with ID **{customer_id}** was found in the warehouse.")
        return
    c = rows.iloc[0]

    status = "Churned" if c.churn else "Active"
    st.subheader(f"{c.customer_id} · {c.segment_name}")
    st.caption(f"Status in the data snapshot: **{status}**"
               + (" (actions are still shown for reference)." if c.churn else "."))

    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(risk_gauge(c.churn_probability, "Churn probability"), width="stretch")
        st.caption("Calibrated out-of-fold estimate stored in the database (the model never saw this "
                   "customer's outcome). The red marker is the median risk used by the retention matrix.")
    with right:
        st.markdown(f"### Recommended action: {c.retention_action}")
        st.write(rm.ACTION_DESCRIPTIONS[c.retention_action])
        m1, m2, m3 = st.columns(3)
        m1.metric("12-month LTV", money(c.estimated_ltv))
        m2.metric("Revenue at risk", money(c.revenue_at_risk))
        m3.metric("RFM", c.rfm_combined, help="Recency (tenure), Frequency (services), Monetary (total billed), each 1–5")
        r1, r2, r3 = st.columns(3)
        r1.metric("Recency score", int(c.rfm_recency_score))
        r2.metric("Frequency score", int(c.rfm_frequency_score))
        r3.metric("Monetary score", int(c.rfm_monetary_score))

    st.markdown("#### Account details")
    services = [label for col, label in [
        ("online_security", "Online Security"), ("online_backup", "Online Backup"),
        ("device_protection", "Device Protection"), ("tech_support", "Tech Support"),
        ("streaming_tv", "Streaming TV"), ("streaming_movies", "Streaming Movies")] if c[col] == "Yes"]
    details = pd.DataFrame({
        "Field": ["Tenure", "Contract", "Payment method", "Paperless billing", "Internet",
                  "Phone", "Add-ons", "Monthly charges", "Total billed"],
        "Value": [f"{c.tenure_months} month{'' if c.tenure_months == 1 else 's'}", c.contract_type, c.payment_method,
                  "Yes" if c.paperless_billing else "No", c.internet_service,
                  ("Yes" + (", multiple lines" if c.multiple_lines == "Yes" else "")) if c.phone_service else "No",
                  ", ".join(services) or "None", money(c.monthly_charges, 2), money(c.total_charges, 2)],
    })
    st.dataframe(details, hide_index=True, width="stretch")
    st.session_state["lookup_customer"] = c   # available to features added in later phases


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Measuring feature importance…")
def grouped_importance() -> pd.DataFrame:
    """Grouped permutation importance of the deployed model over all customers (live data)."""
    data = cm.load_modeling_data(engine())
    return cm.grouped_permutation_importances(model_artifact()["pipeline"], data[cm.FEATURES],
                                              data[cm.TARGET], n_repeats=5)


def rfm_bin_edges() -> dict:
    """Upper bound of each RFM score's source value, read from the warehouse."""
    edges = query("""
        SELECT 'R' AS dim, cs.rfm_recency_score AS score, MAX(f.tenure_months)::float AS upper
        FROM customer_segments cs JOIN fact_subscription f USING (customer_id) GROUP BY 1, 2
        UNION ALL
        SELECT 'F', cs.rfm_frequency_score, MAX(f.num_services)::float
        FROM customer_segments cs JOIN fact_subscription f USING (customer_id) GROUP BY 1, 2
        UNION ALL
        SELECT 'M', cs.rfm_monetary_score, MAX(f.total_charges)::float
        FROM customer_segments cs JOIN fact_subscription f USING (customer_id) GROUP BY 1, 2
    """)
    return {d: g.sort_values("score")[["score", "upper"]].to_numpy().tolist() for d, g in edges.groupby("dim")}


def score_from_edges(value: float, edges: list) -> int:
    """First score whose upper bound covers the value (top score if above all bounds)."""
    for score, upper in edges:
        if value <= upper:
            return int(score)
    return int(edges[-1][0])


def page_churn_drivers():
    st.title("Churn Drivers")
    artifact = model_artifact()
    st.caption(f"Model: calibrated {artifact['model_name']} (trained {artifact['trained_at'][:10]}).")

    imp = grouped_importance().reset_index(names="group").sort_values("importance_mean")
    fig = px.bar(imp, x="importance_mean", y="group", orientation="h", error_x="importance_std",
                 title="What drives churn: drop in ROC-AUC when each feature group is shuffled")
    fig.update_traces(marker_color=PRIMARY, hovertemplate="%{y}<br>AUC drop %{x:.4f}<extra></extra>")
    fig.update_layout(xaxis_title="Drop in ROC-AUC", yaxis_title="")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.caption("Measured live on the deployed model over all customers (in-sample). The held-out "
               "analysis in notebook 03 gives the same ranking.")
    st.markdown(
        "- **Tenure and contract type dominate.** Risk is highest in the first months and on "
        "month-to-month contracts.\n"
        "- **Protective add-ons and internet type come next.** Customers without Online Security / "
        "Tech Support and fiber customers churn more.\n"
        "- **Payment & billing:** electronic-check payers churn more than auto-pay customers.\n"
        "- **Demographics and the number of services barely matter** once the above are known.")

    st.divider()
    st.subheader("What-if simulator")
    st.caption("Start from a real customer, change their contract or services, and see the model's "
               "calibrated churn probability update.")
    base_id = st.text_input("Baseline customer ID", value="7590-VHVEG", key="whatif_id").strip().upper()
    base_rows = one_customer(base_id)
    if base_rows.empty:
        st.error(f"No customer with ID **{base_id}** was found.")
        return
    base = to_model_features(base_rows).iloc[0]
    pipeline = artifact["pipeline"]
    base_prob = cm.predict_churn(pipeline, pd.DataFrame([base.to_dict()]))[0]

    yes_no_addon = lambda cur, internet: (["No internet service"] if internet == "No" else ["Yes", "No"])
    c1, c2, c3 = st.columns(3)
    contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"],
                            index=["Month-to-month", "One year", "Two year"].index(base.contract_type))
    payment_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    payment = c2.selectbox("Payment method", payment_options, index=payment_options.index(base.payment_method))
    internet = c3.selectbox("Internet service", ["DSL", "Fiber optic", "No"],
                            index=["DSL", "Fiber optic", "No"].index(base.internet_service))
    c4, c5, c6 = st.columns(3)
    tenure = c4.slider("Tenure (months)", 0, 72, int(base.tenure_months))
    monthly = c5.slider("Monthly charges ($)", 18.0, 120.0, float(base.monthly_charges), step=0.5)
    paperless = c6.selectbox("Paperless billing", ["Yes", "No"], index=["Yes", "No"].index(base.paperless_billing))

    addon_values = {}
    cols = st.columns(6)
    for col, name in zip(cols, ADDON_COLUMNS):
        options = yes_no_addon(base[name], internet)
        current = base[name] if base[name] in options else options[-1]
        addon_values[name] = col.selectbox(name.replace("_", " ").title(), options,
                                           index=options.index(current), key=f"whatif_{name}")

    scenario = base.copy()
    scenario.update({"contract_type": contract, "payment_method": payment, "internet_service": internet,
                     "tenure_months": tenure, "monthly_charges": monthly, "paperless_billing": paperless,
                     **addon_values})
    # Keep derived features consistent with the edited inputs
    scenario["num_services"] = int(
        (scenario.phone_service == "Yes") + (scenario.multiple_lines == "Yes")
        + (internet != "No") + sum(v == "Yes" for v in addon_values.values()))
    if tenure != base.tenure_months or monthly != base.monthly_charges:
        scenario["total_charges"] = monthly * tenure
    edges = rfm_bin_edges()
    scenario["rfm_recency_score"] = score_from_edges(tenure, edges["R"])
    scenario["rfm_frequency_score"] = score_from_edges(scenario["num_services"], edges["F"])
    scenario["rfm_monetary_score"] = score_from_edges(scenario["total_charges"], edges["M"])

    new_prob = cm.predict_churn(pipeline, pd.DataFrame([scenario.to_dict()])[cm.FEATURES])[0]
    g1, g2 = st.columns(2)
    g1.plotly_chart(risk_gauge(base_prob, f"Baseline ({base_id})"), width="stretch")
    g2.plotly_chart(risk_gauge(new_prob, "What-if scenario"), width="stretch")
    delta = new_prob - base_prob
    st.metric("Change in churn probability", f"{new_prob:.1%}", f"{delta * 100:+.1f} pts", delta_color="inverse")
    st.caption("Both gauges use the deployed model, so the baseline can differ slightly from the database "
               "value shown on Customer Lookup (see notebook 03, section 5). Monthly charges do not change "
               "automatically when services change; adjust the slider to reflect a new price. If tenure or "
               "price is edited, total billed is approximated as monthly charges × tenure.")


def page_business_impact():
    st.title("Business Impact")
    df = all_customers()
    active = df[~df.churn]

    summary = rm.matrix_summary(active)
    total_risk = summary.loc["Total", "revenue_at_risk"]
    offer = summary.loc[rm.RETENTION_OFFER]
    c1, c2, c3 = st.columns(3)
    c1.metric("12-month revenue at risk (active customers)", money(total_risk))
    c2.metric("Active customers", f"{len(active):,}")
    c3.metric("In 'Retention Offer'", f"{int(offer.customers):,}",
              f"{offer.revenue_at_risk / total_risk:.0%} of revenue at risk", delta_color="off")
    st.caption("Revenue at risk = calibrated churn probability × 12-month LTV (monthly charges × 12), "
               "summed over active customers.")

    left, right = st.columns([1.1, 1])
    with left:
        by_action = summary.loc[rm.ACTIONS].reset_index(names="action")
        fig = px.bar(by_action, x="revenue_at_risk", y="action", orientation="h", color="action",
                     color_discrete_map=ACTION_COLORS, title="Revenue at risk by retention action",
                     text=by_action.revenue_at_risk.map(lambda v: money(v)))
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>%{x:$,.0f} at risk<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="",
                          yaxis=dict(categoryorder="array", categoryarray=rm.ACTIONS[::-1]))
        st.plotly_chart(style(fig), width="stretch")
    with right:
        st.markdown("#### Revenue at risk by action")
        table = summary.reset_index(names="Action")[[
            "Action", "customers", "avg_churn_probability", "avg_monthly_charges", "revenue_at_risk",
            "share_of_revenue_at_risk_pct"]]
        st.dataframe(table.rename(columns={
            "customers": "Customers", "avg_churn_probability": "Avg churn prob.",
            "avg_monthly_charges": "Avg monthly", "revenue_at_risk": "Revenue at risk",
            "share_of_revenue_at_risk_pct": "Share %"}).style.format({
                "Customers": "{:,.0f}", "Avg churn prob.": "{:.1%}", "Avg monthly": "${:,.2f}",
                "Revenue at risk": "${:,.0f}", "Share %": "{:.1f}%"}),
            hide_index=True, width="stretch")

    st.subheader("Projected savings from the Retention Offer group")
    savings = rm.projected_savings(active).reset_index()
    s_cols = st.columns(len(savings))
    for col, (_, s) in zip(s_cols, savings.iterrows()):
        col.metric(f"{s.success_rate_pct:.0f}% success", money(s.projected_revenue_saved),
                   f"≈{s.expected_customers_saved:.0f} customers kept", delta_color="off")
    custom = st.slider("Custom success rate (%)", 1, 60, 20)
    custom_row = rm.projected_savings(active, success_rates=(custom / 100,)).iloc[0]
    st.write(f"At **{custom}%**, preventing that share of the expected churn among "
             f"{int(custom_row.customers_targeted):,} Retention Offer customers saves about "
             f"**{money(custom_row.projected_revenue_saved)}** over 12 months "
             f"(≈{custom_row.expected_customers_saved:.0f} customers).")

    st.subheader("Cost-based targeting (adjustable)")
    st.caption("Separate from the median-split matrix. With calibrated probabilities it is worth contacting "
               "a customer when churn probability > 1 / (1 + r), where r = cost of a missed churner ÷ "
               "cost of a retention offer.")
    r = st.slider("r: missed churner cost ÷ offer cost", 1.0, 20.0, 5.0, step=0.5)
    flags = rm.cost_based_flags(active, r)
    f1, f2, f3 = st.columns(3)
    f1.metric("Contact threshold", f"{flags['threshold']:.1%}")
    f2.metric("Active customers to contact", f"{flags['customers_flagged']:,}",
              f"{flags['share_flagged_pct']:.0f}% of active", delta_color="off")
    f3.metric("Revenue at risk covered", money(flags["revenue_at_risk_flagged"]),
              f"{flags['revenue_at_risk_flagged'] / total_risk:.0%} of total", delta_color="off")

    st.subheader("Segment-level action summary (active customers)")
    pivot = active.pivot_table(index="segment_name", columns="retention_action", values="revenue_at_risk",
                               aggfunc=["count", "sum"], fill_value=0)
    counts = pivot["count"].reindex(index=SEGMENT_ORDER, columns=rm.ACTIONS, fill_value=0)
    risk = pivot["sum"].reindex(index=SEGMENT_ORDER, columns=rm.ACTIONS, fill_value=0)
    combined = counts.astype(int).map("{:,}".format) + " · " + risk.map(lambda v: money(v))
    combined.index.name = "Segment"
    st.dataframe(combined, width="stretch")
    st.caption("Each cell: active customers · 12-month revenue at risk.")

    everyone = rm.matrix_summary(df)
    everyone_savings = rm.projected_savings(df)
    st.caption(
        f"Including recently churned customers (all {len(df):,} in the snapshot): revenue at risk "
        f"{money(everyone.loc['Total', 'revenue_at_risk'])}, Retention Offer group "
        f"{int(everyone.loc[rm.RETENTION_OFFER, 'customers']):,} customers, projected savings "
        + " / ".join(money(v) for v in everyone_savings.projected_revenue_saved)
        + " at 10 / 20 / 30%. These overstate the opportunity: churned customers' revenue is already lost.")


# --------------------------------------------------------------------------- navigation
# Register pages here; later phases add "SHAP Explanations" and "Ask SubscribeIQ".
PAGES = {
    "Overview": page_overview,
    "Segment Explorer": page_segment_explorer,
    "Customer Lookup": page_customer_lookup,
    "Churn Drivers": page_churn_drivers,
    "Business Impact": page_business_impact,
}


def main():
    st.sidebar.title("SubscribeIQ")
    st.sidebar.caption("Subscription Intelligence & Retention Engine")
    choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    if st.sidebar.button("Refresh data from PostgreSQL"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(f"Data is queried live from PostgreSQL and cached for {CACHE_TTL_SECONDS // 60} minutes.")
    try:
        PAGES[choice]()
    except Exception as exc:  # keep the app alive and show a readable error
        st.error(f"Something went wrong while loading this page: {exc}")


main()
