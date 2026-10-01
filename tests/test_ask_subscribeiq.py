"""Tests for src/ask_subscribeiq.py.

The SQL guard tests need nothing. The ask() tests use a scripted fake LLM against the live
database (skipped if PostgreSQL is unreachable). One live Gemini test runs only when
RUN_GEMINI_TESTS=1, so the suite never spends API quota by default.
"""

import json
import os

import pytest
from sqlalchemy import text

from src import ask_subscribeiq as ask
from src.db_connection import get_engine


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM fact_subscription",
    "select segment_name from customer_segments;",
    "WITH t AS (SELECT * FROM fact_subscription) SELECT COUNT(*) FROM t",
    "SELECT 'drop; update' AS s FROM dim_customer",
    "SELECT c.gender FROM dim_customer c JOIN dim_contract k USING (customer_id) "
    "WHERE k.payment_method LIKE '%check%'",
    # shapes the model commonly writes, and aliases the old keyword list rejected
    "SELECT ROUND(SUM(cs.churn_probability * f.estimated_ltv)::numeric, 2) AS risk "
    "FROM customer_segments cs JOIN fact_subscription f ON cs.customer_id = f.customer_id",
    "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monthly_charges) FROM fact_subscription",
    "SELECT COUNT(*) FILTER (WHERE churn) AS churned FROM fact_subscription",
    "SELECT customer_id, RANK() OVER (PARTITION BY churn ORDER BY monthly_charges DESC) "
    "FROM fact_subscription",
    "SELECT segment_name AS cluster, COUNT(*) AS \"Delete count\" FROM customer_segments "
    "GROUP BY segment_name",
    "SELECT EXTRACT(YEAR FROM DATE '2020-01-01') AS y FROM dim_customer",
    "SELECT COUNT(*) FROM dim_service WHERE online_security IS DISTINCT FROM tech_support",
])
def test_check_sql_accepts_read_queries(sql):
    assert ask.check_sql(sql) == sql.strip().rstrip(";")


@pytest.mark.parametrize("sql", [
    "UPDATE fact_subscription SET churn = TRUE",
    "DELETE FROM dim_customer",
    "SELECT 1; DROP TABLE dim_customer",
    "WITH x AS (DELETE FROM dim_customer RETURNING *) SELECT * FROM x",
    "SELECT * INTO copy_table FROM dim_customer",
    "SELECT * FROM pg_user",
    "SELECT pg_sleep(30)",
    "SELECT * FROM information_schema.tables",
    "SELECT set_config('transaction_read_only', 'off', true)",
    "SELECT nextval('dim_service_service_id_seq')",
    "SELECT NEXTVAL ('dim_service_service_id_seq')",
    "SELECT setval('dim_service_service_id_seq', 1)",
    "SELECT 1 -- comment",
    "SELECT /* c */ 1",
    "SELECT * FROM users",
    "",
    # function escape hatches: arbitrary SQL in a string, whole-database dumps, server info
    "SELECT query_to_xml('select rolname, rolpassword from pg_authid', true, false, '')",
    "SELECT \"query_to_xml\"('select 1', true, false, '')",
    "SELECT database_to_xml(true, false, '')",
    "SELECT table_to_xml('dim_customer', true, false, '')",
    "SELECT current_setting('data_directory')",
    "SELECT version()",
    "SELECT inet_server_addr()",
    "SELECT public.some_function(1)",
    "SELECT * FROM \"pg_authid\"",
    "SELECT * FROM \"users\"",
])
def test_check_sql_rejects_everything_else(sql):
    with pytest.raises(ask.AskError):
        ask.check_sql(sql)


class FakeLLM:
    """Returns scripted responses in order and records the prompts it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, system, prompt, json_mode=False):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _plan(sql, population="all 7,043 customers", reason=None):
    return json.dumps({"sql": sql, "population": population, "reason": reason})


def _db_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


db = pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable")


@db
@pytest.mark.parametrize("sql", [
    "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY monthly_charges) FROM fact_subscription",
    "SELECT segment_name AS cluster, COUNT(*) AS \"Delete count\" FROM customer_segments "
    "GROUP BY segment_name",
    "SELECT COUNT(*) FROM dim_service WHERE online_security IS DISTINCT FROM tech_support",
])
def test_accepted_queries_actually_run(sql):
    rows, _ = ask.run_sql(get_engine(), sql)
    assert len(rows) > 0


@db
def test_ask_runs_query_and_summarises():
    llm = FakeLLM(_plan("SELECT COUNT(*) AS customers FROM fact_subscription"), "There are 7,043.")
    answer = ask.ask("How many customers?", get_engine(), llm)
    assert answer.answered and answer.text == "There are 7,043."
    assert answer.rows.customers.iloc[0] == 7043
    assert "7043" in llm.prompts[1]   # the summary is grounded in the actual result


@db
def test_ask_sends_a_failing_query_back_once():
    llm = FakeLLM(_plan("SELECT no_such_column FROM fact_subscription"),
                  _plan("SELECT COUNT(*) AS n FROM fact_subscription WHERE churn"), "1,869 churned.")
    answer = ask.ask("How many churned?", get_engine(), llm)
    assert answer.answered and answer.rows.n.iloc[0] == 1869
    assert len(answer.attempts) == 1 and "no_such_column" in answer.attempts[0]["error"]
    assert "previous query failed" in llm.prompts[1]


@db
def test_ask_gives_up_after_the_fix_attempt():
    bad = _plan("DELETE FROM dim_customer")
    answer = ask.ask("Delete everyone", get_engine(), FakeLLM(bad, bad))
    assert not answer.answered and len(answer.attempts) == 2
    with get_engine().connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM dim_customer")).scalar_one() == 7043


@db
def test_ask_reports_a_declined_question():
    llm = FakeLLM(_plan(None, None, "Weather data is not in the warehouse."))
    answer = ask.ask("Weather in Paris?", get_engine(), llm)
    assert not answer.answered and answer.sql is None
    assert answer.text == "Weather data is not in the warehouse."


@db
def test_run_sql_caps_rows_and_converts_numerics():
    rows, truncated = ask.run_sql(get_engine(), "SELECT customer_id, monthly_charges FROM fact_subscription",
                                  max_rows=10)
    assert len(rows) == 10 and truncated
    assert rows.monthly_charges.dtype.kind == "f"


@db
def test_run_sql_transaction_is_read_only():
    # Bypass check_sql to prove the database itself refuses writes inside run_sql's transaction
    original = ask.check_sql
    ask.check_sql = lambda sql: sql
    try:
        with pytest.raises(ask.AskError, match="read-only"):   # nextval would advance a sequence
            ask.run_sql(get_engine(), "SELECT nextval('dim_service_service_id_seq')")
    finally:
        ask.check_sql = original


@db
@pytest.mark.skipif(os.getenv("RUN_GEMINI_TESTS") != "1", reason="set RUN_GEMINI_TESTS=1 to call Gemini")
def test_live_gemini_answers_a_simple_question():
    answer = ask.ask("How many customers are there in total?", get_engine())
    assert answer.answered, answer.text
    assert 7043 in answer.rows.to_numpy().ravel()
    assert "7,043" in answer.text or "7043" in answer.text
