"""Smoke tests: every dashboard page renders against the live database without errors.

Skipped automatically if PostgreSQL is unreachable.
"""

from pathlib import Path

import pytest
from sqlalchemy import text

from src.db_connection import get_engine

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
PAGES = ["Overview", "Segment Explorer", "Customer Lookup", "Churn Drivers", "SHAP Explanations",
         "Business Impact", "Ask SubscribeIQ"]


def _db_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable")


def _run_page(page: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    at = _run_page(page)
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == page


def test_customer_lookup_unknown_id_shows_error():
    at = _run_page("Customer Lookup")
    at.text_input[0].set_value("0000-NOPE").run()
    assert any("No customer" in e.value for e in at.error)


def test_customer_lookup_shows_action():
    at = _run_page("Customer Lookup")
    at.text_input[0].set_value("5575-GNVDE").run()
    assert not at.exception
    assert any("Recommended action" in m.value for m in at.markdown)


def test_what_if_contract_change_moves_probability():
    at = _run_page("Churn Drivers")
    before = next(m for m in at.metric if m.label == "Change in churn probability")
    assert before.delta.startswith("+0.0")
    contract = next(s for s in at.selectbox if s.label == "Contract")
    contract.set_value("Two year").run()
    after = next(m for m in at.metric if m.label == "Change in churn probability")
    assert after.delta.startswith("-")   # a two-year contract lowers risk for this customer


def test_shap_page_matches_the_deployed_model():
    from src import churn_model as cm

    at = _run_page("SHAP Explanations")
    at.text_input(key="shap_id").set_value("5575-GNVDE").run()
    assert not at.exception and not at.error
    shown = next(m for m in at.metric if m.label == "This customer (deployed model)").value
    data = cm.load_modeling_data(get_engine())
    rows = data[data.customer_id == "5575-GNVDE"]
    expected = cm.predict_churn(cm.load_model()["pipeline"], rows[cm.FEATURES])[0]
    assert shown == f"{expected:.1%}"


def test_shap_page_unknown_id_shows_error():
    at = _run_page("SHAP Explanations")
    at.text_input(key="shap_id").set_value("0000-NOPE").run()
    assert any("No customer" in e.value for e in at.error)
