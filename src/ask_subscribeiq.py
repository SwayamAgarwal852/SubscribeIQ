"""Ask SubscribeIQ: natural-language questions answered from the warehouse with Gemini.

Two model calls per question:
    1. Gemini writes one PostgreSQL SELECT for the question (JSON: sql, population, or a reason
       it cannot be answered from the data).
    2. The query runs read-only against the warehouse; Gemini then writes a short answer using
       only the returned rows. If the query fails, the error is sent back once for a fix.

Every answer keeps the SQL and the rows it was based on, so the dashboard can show them.

Generated SQL is never trusted. check_sql accepts a single SELECT / WITH statement over the
SubscribeIQ tables only, with no comments, system catalogs or admin functions. run_sql wraps
it in a subquery with a row limit and runs it inside a READ ONLY transaction with a statement
timeout, so even a statement that slipped past the check cannot write. For deployment, also
connect with a database role that only has SELECT on these tables.

Requires GEMINI_API_KEY in .env. GEMINI_MODEL optionally overrides the default model.
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-flash-latest")
MAX_ROWS = 500
ROWS_SENT_TO_MODEL = 60
STATEMENT_TIMEOUT_MS = 5000

ALLOWED_TABLES = {"dim_customer", "dim_service", "dim_contract", "fact_subscription",
                  "customer_segments"}
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|upsert|drop|alter|create|truncate|grant|revoke|copy|vacuum|"
    r"analyze|reindex|cluster|lock|call|do|execute|prepare|deallocate|listen|notify|set|reset|"
    r"show|comment|security|refresh|import|load|discard|checkpoint|into|set_config|dblink)\b"
    r"|\bpg_|\blo_|information_schema|--|/\*",
    re.IGNORECASE)

SCHEMA_GUIDE = """
PostgreSQL warehouse for the IBM Telco customer churn snapshot: 7,043 customers, one row per
customer in every table, all joined on customer_id (VARCHAR, e.g. '7590-VHVEG').

dim_customer(customer_id, gender TEXT 'Male'|'Female', senior_citizen BOOLEAN, partner BOOLEAN,
             dependents BOOLEAN)
dim_service(customer_id, phone_service BOOLEAN,
            multiple_lines TEXT 'Yes'|'No'|'No phone service',
            internet_service TEXT 'DSL'|'Fiber optic'|'No',
            online_security, online_backup, device_protection, tech_support, streaming_tv,
            streaming_movies: each TEXT 'Yes'|'No'|'No internet service')
dim_contract(customer_id, contract_type TEXT 'Month-to-month'|'One year'|'Two year',
             payment_method TEXT 'Electronic check'|'Mailed check'|'Bank transfer (automatic)'|
             'Credit card (automatic)', paperless_billing BOOLEAN)
fact_subscription(customer_id, tenure_months INTEGER, monthly_charges NUMERIC,
                  total_charges NUMERIC (lifetime billed), churn BOOLEAN,
                  num_services SMALLINT (0-9 active services),
                  estimated_ltv NUMERIC (12-month LTV = monthly_charges * 12))
customer_segments(customer_id, rfm_recency_score, rfm_frequency_score, rfm_monetary_score
                  (SMALLINT 1-5; recency = tenure, frequency = services, monetary = total billed),
                  rfm_combined TEXT e.g. '5-4-3', cluster_label SMALLINT,
                  segment_name TEXT 'New & Uncommitted'|'Flexible Fiber Users'|
                  'Established Power Users'|'Loyal Basics',
                  churn_probability NUMERIC (calibrated model probability, 0.005-0.995),
                  retention_action TEXT 'Retention Offer'|'Early Access / Upsell'|
                  'Monitor Only'|'Nurture')

Business definitions:
- churn = TRUE means the customer has ALREADY churned in this snapshot (1,869 customers);
  active customers are churn = FALSE (5,174).
- Churn rate = AVG(churn::int).
- Revenue at risk (12-month) = churn_probability * estimated_ltv.
- Retention actions come from median splits: high LTV = estimated_ltv > 844.20,
  high risk = churn_probability > 0.1833. Retention Offer = high LTV + high risk,
  Early Access / Upsell = high LTV + low risk, Monitor Only = low LTV + high risk,
  Nurture = low LTV + low risk.
- Projected savings at success rate s = s * revenue at risk of the Retention Offer group.
- Monthly recurring revenue (MRR) = SUM(monthly_charges).
"""

SQL_INSTRUCTIONS = f"""You translate business questions about SubscribeIQ into one PostgreSQL query.
{SCHEMA_GUIDE}
Rules:
- Write exactly one SELECT statement (WITH ... SELECT is fine). Use only the five tables above.
- Population: money questions (revenue at risk, savings, retention targeting, value of a
  group) default to ACTIVE customers (churn = FALSE) because churned revenue is already lost.
  Churn rates, segment profiles, counts and model questions default to ALL 7,043 customers.
  Follow the user if they name a population.
- Cast NUMERIC results to float and round sensibly (ROUND(x::numeric, 2)). Give columns clear
  snake_case aliases. Prefer aggregates; for lists of customers ORDER BY something meaningful
  and LIMIT 50 or fewer.
- If the question cannot be answered from these tables (e.g. it needs data that does not exist,
  or asks you to change data), do not write SQL.

Respond with JSON only:
{{"sql": "<query or null>", "population": "<who the query covers, e.g. 'all 7,043 customers'
or 'active customers only (5,174)'>", "reason": "<if sql is null: why, in one sentence>"}}"""

ANSWER_INSTRUCTIONS = """You are SubscribeIQ's analyst. Answer the user's question in 2-5 plain
sentences using ONLY the query result provided. Rules:
- Every number you state must come from the result (you may round it). Do not invent figures,
  causes or recommendations the data does not support.
- Say which population the figures cover (all customers vs active customers only).
- Format money as $1,234 and rates as percentages with one decimal.
- estimated_ltv is a 12-month LTV (monthly charges x 12): call it "12-month LTV", never
  "lifetime value". churn_probability is the model's calibrated churn probability.
- If the result is empty, say that no matching customers were found.
- If the result was truncated, say the answer is based on the rows shown.
Plain text only, no markdown headings or tables."""


class AskError(Exception):
    """A question could not be answered (bad SQL, model refusal, or API failure)."""


@dataclass
class Answer:
    """Everything produced for one question.

    Attributes:
        question: The user's question.
        text: The final answer (or the reason none could be given).
        sql: The query that was run (None if no query was written).
        population: Who the query covers, as described by the model.
        rows: Query result (empty if no query ran).
        truncated: True if the result hit MAX_ROWS.
        answered: False if the question was declined or failed.
        attempts: SQL errors encountered before the final query.
    """
    question: str
    text: str
    sql: str | None = None
    population: str | None = None
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    truncated: bool = False
    answered: bool = True
    attempts: list = field(default_factory=list)


class GeminiLLM:
    """Minimal Gemini client: one system instruction + one prompt -> text.

    Tries each model in order and moves on when one is unavailable (overloaded, rate-limited
    or retired), so a busy model does not break the page.
    """

    def __init__(self, api_key: str | None = None, models: tuple | None = None):
        from google import genai

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise AskError("GEMINI_API_KEY is not set (add it to .env)")
        override = os.getenv("GEMINI_MODEL")
        self.models = models or ((override,) if override else DEFAULT_MODELS)
        self.client = genai.Client(api_key=api_key)
        self.last_model = None

    def generate(self, system: str, prompt: str, json_mode: bool = False) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system, temperature=0,
            response_mime_type="application/json" if json_mode else "text/plain",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        failures = []
        for model in self.models:
            try:
                response = self.client.models.generate_content(model=model, contents=prompt,
                                                               config=config)
            except errors.APIError as exc:
                if exc.code in (404, 429, 500, 503):
                    failures.append(f"{model}: {exc.code} {exc.status}")
                    continue
                raise AskError(f"Gemini request failed ({exc.code} {exc.status})") from exc
            self.last_model = model
            if not response.text:
                raise AskError("Gemini returned an empty response")
            return response.text
        raise AskError("No Gemini model is available right now (" + "; ".join(failures) + ")")


def _strip_literals(sql: str) -> str:
    """Remove string literals and quoted identifiers so keyword checks ignore their contents."""
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "''", sql)


def check_sql(sql: str) -> str:
    """Validate model-written SQL and return it without a trailing semicolon.

    Args:
        sql: Candidate query.

    Returns:
        The cleaned query.

    Raises:
        AskError: If it is not a single read-only SELECT over the SubscribeIQ tables.
    """
    cleaned = (sql or "").strip().rstrip(";").strip()
    bare = _strip_literals(cleaned)
    if not re.match(r"^(select|with)\b", bare, re.IGNORECASE):
        raise AskError("Only SELECT queries are allowed")
    if ";" in bare:
        raise AskError("Only a single statement is allowed")
    match = FORBIDDEN.search(bare)
    if match:
        raise AskError(f"Query uses a disallowed keyword or object: {match.group(0)!r}")
    ctes = {m.lower() for m in re.findall(r"(\w+)\s+as\s+(?:not\s+)?(?:materialized\s+)?\(",
                                          bare, re.IGNORECASE)}
    tables = {t.lower().split(".")[-1]
              for t in re.findall(r"\b(?:from|join)\s+([\w.]+)", bare, re.IGNORECASE)}
    unknown = tables - ALLOWED_TABLES - ctes
    if unknown:
        raise AskError(f"Query reads tables outside SubscribeIQ: {', '.join(sorted(unknown))}")
    return cleaned


def run_sql(engine: Engine, sql: str, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, bool]:
    """Run a checked query read-only, with a timeout and a row cap.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.
        sql: Query (passed through check_sql first).
        max_rows: Maximum rows to return.

    Returns:
        (result dataframe, truncated flag).

    Raises:
        AskError: If the query is rejected or fails in PostgreSQL.
    """
    sql = check_sql(sql)
    wrapped = f"SELECT * FROM (\n{sql}\n) AS answer LIMIT {max_rows + 1}"
    with engine.connect() as conn:
        raw = conn.connection.dbapi_connection
        raw.rollback()   # start clean: the driver opens a new transaction on the next execute
        try:
            with raw.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(f"SET LOCAL statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
                cur.execute(wrapped)   # no parameters: the driver does no % interpolation
                columns = [d[0] for d in cur.description]
                records = cur.fetchall()
        except Exception as exc:
            raise AskError(str(exc).strip().splitlines()[0]) from exc
        finally:
            raw.rollback()
    df = pd.DataFrame.from_records(records, columns=columns)
    for col in df.columns:   # NUMERIC arrives as Decimal
        if df[col].map(lambda v: v.__class__.__name__ == "Decimal").any():
            df[col] = pd.to_numeric(df[col])
    return df.head(max_rows), len(df) > max_rows


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AskError("Gemini did not return valid JSON for the query") from exc


def write_sql(llm, question: str, previous: str | None = None, error: str | None = None) -> dict:
    """Ask the model for a query (or a fix of a failed one).

    Returns:
        Dict with keys sql, population, reason.
    """
    prompt = f"Question: {question}"
    if previous:
        prompt += (f"\n\nYour previous query failed.\nQuery:\n{previous}\nError: {error}\n"
                   "Return a corrected query in the same JSON format.")
    plan = _parse_json(llm.generate(SQL_INSTRUCTIONS, prompt, json_mode=True))
    return {"sql": plan.get("sql") or None, "population": plan.get("population"),
            "reason": plan.get("reason")}


def summarise(llm, question: str, sql: str, population: str | None, rows: pd.DataFrame,
              truncated: bool) -> str:
    """Ask the model to answer the question from the query result."""
    shown = rows.head(ROWS_SENT_TO_MODEL)
    note = ""
    if truncated or len(rows) > len(shown):
        note = f"\n(Showing {len(shown)} of {'more than ' if truncated else ''}{len(rows)} rows.)"
    result = shown.to_csv(index=False) if len(shown) else "(no rows)"
    prompt = (f"Question: {question}\nPopulation covered: {population or 'not stated'}\n"
              f"SQL:\n{sql}\n\nResult ({len(rows)} rows):\n{result}{note}")
    return llm.generate(ANSWER_INSTRUCTIONS, prompt).strip()


def ask(question: str, engine: Engine, llm=None, max_fix_attempts: int = 1) -> Answer:
    """Answer a natural-language question from the warehouse.

    Args:
        question: The user's question.
        engine: SQLAlchemy engine for the SubscribeIQ database.
        llm: Object with generate(system, prompt, json_mode=False) -> str (default GeminiLLM).
        max_fix_attempts: How many times a failing query is sent back for correction.

    Returns:
        An Answer. Declined or failed questions return answered=False with the reason as text.
    """
    question = question.strip()
    if not question:
        return Answer(question, "Please type a question.", answered=False)
    try:
        llm = llm or GeminiLLM()
        plan = write_sql(llm, question)
    except AskError as exc:
        return Answer(question, str(exc), answered=False)
    if not plan["sql"]:
        return Answer(question, plan["reason"] or "That question can't be answered from the "
                      "SubscribeIQ data.", population=plan["population"], answered=False)

    attempts = []
    for attempt in range(max_fix_attempts + 1):
        try:
            rows, truncated = run_sql(engine, plan["sql"])
            break
        except AskError as exc:
            attempts.append({"sql": plan["sql"], "error": str(exc)})
            if attempt == max_fix_attempts:
                return Answer(question, f"The generated query could not be run: {exc}",
                              sql=plan["sql"], population=plan["population"], answered=False,
                              attempts=attempts)
            try:
                plan = write_sql(llm, question, plan["sql"], str(exc))
            except AskError as fix_exc:
                return Answer(question, str(fix_exc), answered=False, attempts=attempts)
            if not plan["sql"]:
                return Answer(question, plan["reason"] or "No valid query could be written.",
                              answered=False, attempts=attempts)

    try:
        text = summarise(llm, question, plan["sql"], plan["population"], rows, truncated)
    except AskError as exc:
        text = f"The query ran, but the summary failed ({exc}). The result is shown below."
    return Answer(question, text, sql=plan["sql"], population=plan["population"], rows=rows,
                  truncated=truncated, attempts=attempts)
