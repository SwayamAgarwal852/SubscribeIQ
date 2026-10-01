"""Every number the dashboard displays must equal an independent calculation from PostgreSQL.

Pages are rendered headlessly with AppTest (fresh cache, live database); metrics, tables and
chart data are read back from the rendered elements and compared with SQL computed here.

Skipped automatically if PostgreSQL is unreachable.
"""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src import churn_model as cm
from src import retention_matrix as rm
from src.db_connection import get_engine

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
SEGMENTS = ["New & Uncommitted", "Flexible Fiber Users", "Established Power Users", "Loyal Basics"]


def _db_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable")


def _query(sql: str, **params) -> pd.DataFrame:
    return pd.read_sql(text(sql), get_engine(), params=params)


def _render(page: str, setup=None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    if setup:
        setup(at)
        at.run()
    assert not at.exception and not at.error
    return at


def _metrics(at) -> dict:
    return {m.label: (m.value, m.delta) for m in at.metric}


def _values(v) -> list:
    """Plotly figure arrays, decoding the binary typed-array form."""
    if isinstance(v, dict) and "bdata" in v:
        return np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"]).tolist()
    return list(v)


def _chart(at, title_prefix: str) -> dict:
    for el in at.get("plotly_chart"):
        spec = json.loads(el.proto.spec)
        if ((spec["layout"].get("title") or {}).get("text") or "").startswith(title_prefix):
            return spec
    raise AssertionError(f"no chart titled {title_prefix!r}")


money = lambda x, d=0: f"${x:,.{d}f}"
pct = lambda x: f"{x:.1%}"


def test_overview():
    at = _render("Overview")
    k = _query("""SELECT COUNT(*) n, AVG(churn::int)::float cr, AVG(estimated_ltv)::float ltv,
                         SUM(monthly_charges)::float mrr FROM fact_subscription""").iloc[0]
    shown = {label: v for label, (v, _) in _metrics(at).items()}
    assert shown == {"Customers": f"{int(k.n):,}", "Churn rate": pct(k.cr),
                     "Avg 12-month LTV": money(k.ltv), "Monthly recurring revenue": money(k.mrr)}

    seg = _query("""SELECT segment_name, COUNT(*) customers, AVG(churn::int)::float churn_rate,
                           AVG(churn_probability)::float avg_p, AVG(monthly_charges)::float avg_m
                    FROM customer_segments JOIN fact_subscription USING (customer_id) GROUP BY 1""").set_index("segment_name")
    table = at.dataframe[0].value.set_index("Segment")
    assert (table["Customers"] == seg.customers.loc[table.index]).all()
    for col, src in [("Churn rate", "churn_rate"), ("Avg predicted churn", "avg_p"), ("Avg monthly charges", "avg_m")]:
        assert np.allclose(table[col], seg[src].loc[table.index]), col

    pie = _chart(at, "Customers by segment")["data"][0]
    assert dict(zip(pie["labels"], _values(pie["values"]))) == seg.customers.to_dict()
    bar = _chart(at, "Churn rate by segment")
    b = bar["data"][0]
    for x, y, label in zip(_values(b["x"]), b["y"], b["text"]):
        assert x == pytest.approx(seg.churn_rate[y]) and label == pct(seg.churn_rate[y])
    assert f"overall {pct(k.cr)}" in [a["text"] for a in bar["layout"]["annotations"]]


@pytest.mark.parametrize("segment", SEGMENTS)
def test_segment_explorer(segment):
    at = _render("Segment Explorer", lambda a: a.selectbox[0].set_value(segment))
    p = _query("""SELECT COUNT(*) n, AVG(rfm_recency_score)::float r, AVG(rfm_frequency_score)::float f,
                         AVG(rfm_monetary_score)::float m, AVG(churn::int)::float cr,
                         AVG(monthly_charges)::float am, AVG(tenure_months)::float t
                  FROM customer_segments JOIN fact_subscription USING (customer_id) WHERE segment_name = :s""",
               s=segment).iloc[0]
    shown = {label: v for label, (v, _) in _metrics(at).items()}
    assert shown["Customers"] == f"{int(p.n):,}"
    assert shown["Churn rate"] == pct(p.cr)
    assert shown["Avg monthly charges"] == money(p.am, 2)
    assert shown["Avg tenure"] == f"{p.t:.0f} months"

    overall = _query("""SELECT AVG(rfm_recency_score)::float r, AVG(rfm_frequency_score)::float f,
                               AVG(rfm_monetary_score)::float m FROM customer_segments""").iloc[0]
    bars = {t["name"]: _values(t["y"]) for t in _chart(at, "Average RFM")["data"]}
    assert np.allclose(bars[segment], [p.r, p.f, p.m])
    assert np.allclose(bars["All customers"], [overall.r, overall.f, overall.m])

    actions = _query("SELECT retention_action, COUNT(*) n FROM customer_segments WHERE segment_name = :s GROUP BY 1",
                     s=segment).set_index("retention_action").n
    a = _chart(at, "Retention actions")["data"][0]
    assert dict(zip(a["y"], _values(a["x"]))) == {act: actions.get(act, 0) for act in rm.ACTIONS}

    sizes = _query("SELECT segment_name, COUNT(*) n FROM customer_segments GROUP BY 1").set_index("segment_name").n
    points = {t["name"]: len(_values(t["x"])) for t in _chart(at, "Monthly charges vs tenure")["data"]}
    assert points == sizes.to_dict()


@pytest.mark.parametrize("customer_id", ["7590-VHVEG", "0637-KVDLV", "4472-LVYGI"])  # active, churned, tenure 0
def test_customer_lookup(customer_id):
    at = _render("Customer Lookup", lambda a: a.text_input[0].set_value(customer_id))
    c = _query("""SELECT cs.*, f.*, (cs.churn_probability * f.estimated_ltv)::float rar, k.contract_type
                  FROM customer_segments cs JOIN fact_subscription f USING (customer_id)
                  JOIN dim_contract k USING (customer_id) WHERE customer_id = :c""", c=customer_id).iloc[0]
    shown = {label: v for label, (v, _) in _metrics(at).items()}
    assert shown == {"12-month LTV": money(float(c.estimated_ltv)), "Revenue at risk": money(c.rar),
                     "RFM": c.rfm_combined, "Recency score": str(c.rfm_recency_score),
                     "Frequency score": str(c.rfm_frequency_score), "Monetary score": str(c.rfm_monetary_score)}
    gauge = json.loads(at.get("plotly_chart")[0].proto.spec)["data"][0]
    assert gauge["value"] == pytest.approx(float(c.churn_probability) * 100)
    text_shown = " ".join(e.value for e in [*at.markdown, *at.subheader])
    assert c.retention_action in text_shown and c.segment_name in text_shown

    details = at.dataframe[0].value.set_index("Field")["Value"]
    unit = "month" if c.tenure_months == 1 else "months"
    assert details["Tenure"] == f"{c.tenure_months} {unit}"
    assert details["Contract"] == c.contract_type
    assert details["Monthly charges"] == money(float(c.monthly_charges), 2)
    assert details["Total billed"] == money(float(c.total_charges), 2)


def test_customer_lookup_singular_month():
    cid = _query("SELECT customer_id FROM fact_subscription WHERE tenure_months = 1 LIMIT 1").customer_id[0]
    at = _render("Customer Lookup", lambda a: a.text_input[0].set_value(cid))
    assert at.dataframe[0].value.set_index("Field")["Value"]["Tenure"] == "1 month"


def test_churn_drivers():
    at = _render("Churn Drivers")
    artifact = cm.load_model()
    captions = " ".join(c.value for c in at.caption)
    assert f"calibrated {artifact['model_name']} (trained {artifact['trained_at'][:10]})" in captions

    imp = _chart(at, "What drives churn")["data"][0]
    importance = dict(zip(imp["y"], _values(imp["x"])))
    assert set(importance) == set(cm.FEATURE_GROUPS)
    # The narrative under the chart says tenure and contract dominate
    assert set(sorted(importance, key=importance.get)[-2:]) == {"Tenure & lifetime spend", "Contract type"}

    data = cm.load_modeling_data(get_engine()).set_index("customer_id")
    baseline = cm.predict_churn(artifact["pipeline"], data.loc[["7590-VHVEG"], cm.FEATURES])[0]
    value, delta = _metrics(at)["Change in churn probability"]
    assert value == pct(baseline) and delta == "+0.0 pts"
    gauges = [json.loads(el.proto.spec)["data"][0]["value"] for el in at.get("plotly_chart")
              if json.loads(el.proto.spec)["data"][0].get("type") == "indicator"]
    assert np.allclose(gauges, baseline * 100)


def test_business_impact():
    at = _render("Business Impact")
    d = _query("""SELECT cs.retention_action a, cs.segment_name s, cs.churn_probability::float p,
                         f.monthly_charges::float mc, (cs.churn_probability * f.estimated_ltv)::float rar, f.churn
                  FROM customer_segments cs JOIN fact_subscription f USING (customer_id)""")
    active = d[~d.churn]
    total = active.rar.sum()
    offer = active[active.a == rm.RETENTION_OFFER]
    flagged = active[active.p > 1 / (1 + 5.0)]                     # slider default r = 5
    expected = {
        "12-month revenue at risk (active customers)": (money(total), ""),
        "Active customers": (f"{len(active):,}", ""),
        "In 'Retention Offer'": (f"{len(offer):,}", f"{offer.rar.sum() / total:.0%} of revenue at risk"),
        **{f"{r * 100:.0f}% success": (money(r * offer.rar.sum()), f"≈{r * offer.p.sum():.0f} customers kept")
           for r in (0.1, 0.2, 0.3)},
        "Contact threshold": (pct(1 / 6), ""),
        "Active customers to contact": (f"{len(flagged):,}", f"{100 * len(flagged) / len(active):.0f}% of active"),
        "Revenue at risk covered": (money(flagged.rar.sum()), f"{flagged.rar.sum() / total:.0%} of total"),
    }
    shown = _metrics(at)
    for label, (value, delta) in expected.items():
        assert shown[label][0] == value, label
        assert (shown[label][1] or "") == delta, label

    by_action = active.groupby("a").agg(n=("p", "size"), ap=("p", "mean"), am=("mc", "mean"), r=("rar", "sum"))
    table = at.dataframe[0].value.set_index("Action")
    for action, row in by_action.iterrows():
        assert table.loc[action, "Customers"] == row.n
        assert table.loc[action, "Avg churn prob."] == pytest.approx(row.ap)
        assert table.loc[action, "Avg monthly"] == pytest.approx(row.am)
        assert table.loc[action, "Revenue at risk"] == pytest.approx(row.r)
    assert table.loc["Total", "Customers"] == len(active)
    assert table["Share %"].drop("Total").sum() == pytest.approx(100)

    for trace in _chart(at, "Revenue at risk by retention action")["data"]:
        action = _values(trace["y"])[0]
        assert _values(trace["x"])[0] == pytest.approx(by_action.r[action])
        assert trace["text"][0] == money(by_action.r[action])

    sentence = " ".join(m.value for m in at.markdown)               # custom slider default 20%
    assert money(0.2 * offer.rar.sum()) in sentence and f"{len(offer):,} Retention Offer" in sentence

    cells = active.groupby(["s", "a"]).agg(n=("rar", "size"), r=("rar", "sum"))
    grid = at.dataframe[1].value
    for (segment, action), row in cells.iterrows():
        assert grid.loc[segment, action] == f"{int(row.n):,} · {money(row.r)}"

    everyone_offer = d[d.a == rm.RETENTION_OFFER]
    caption = (f"revenue at risk {money(d.rar.sum())}, Retention Offer group {len(everyone_offer):,} customers, "
               "projected savings " + " / ".join(money(r * everyone_offer.rar.sum()) for r in (0.1, 0.2, 0.3)))
    assert caption in " ".join(c.value for c in at.caption)
