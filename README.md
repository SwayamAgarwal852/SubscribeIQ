# SubscribeIQ

**Subscription intelligence and retention engine.** SubscribeIQ loads the IBM Telco Customer
Churn dataset (7,043 customers) into a PostgreSQL star schema. It segments customers with RFM
scoring and KMeans, and predicts churn with a calibrated gradient-boosting model. It then combines
churn risk with customer value to assign each customer one retention action. A Streamlit
dashboard serves every result live from the database, explains individual predictions with
SHAP, and answers plain-English questions about the data with Gemini ("Ask SubscribeIQ").

---

## Headline results

> **Two figure sets — always check which one is being quoted.**
> The dataset is a single snapshot that includes 1,869 customers who **have already churned**.
> - **All customers (7,043)** is what the pipeline scripts print and what is stored in
>   `customer_segments`. It is the right base for model evaluation and segment profiles.
> - **Active customers only (5,174)** is the business headline on the dashboard's Business
>   Impact page. Revenue already lost to churned customers can't be saved, so the all-customer
>   money figures overstate the opportunity.

### Portfolio (all 7,043 customers)

| Metric | Value |
|---|---|
| Customers | 7,043 (1,869 churned / 5,174 active) |
| Churn rate | 26.5% |
| Monthly recurring revenue (sum of monthly charges) | $456,117 |
| Average 12-month LTV | $777 |

### Segments (all 7,043 customers)

| Segment | Customers | Share | Churn rate | Median tenure | Median services | Median monthly |
|---|---:|---:|---:|---:|---:|---:|
| New & Uncommitted | 2,420 | 34.4% | 42.6% | 4 mo | 3 | $50.55 |
| Flexible Fiber Users | 1,674 | 23.8% | 32.2% | 29 mo | 5 | $83.93 |
| Established Power Users | 1,800 | 25.6% | 12.5% | 64 mo | 7 | $94.35 |
| Loyal Basics | 1,149 | 16.3% | 6.5% | 45 mo | 1 | $23.40 |

### Churn model

Calibrated GradientBoosting (learning rate 0.05, depth 2, 200 trees), isotonic calibration.

| Metric | Value | Base |
|---|---|---|
| ROC-AUC, held-out test set (20%) | 0.848 | 1,409 test customers |
| Recall / precision at 0.5, uncalibrated balanced model | 0.81 / 0.54 | 1,409 test customers |
| Brier score, calibrated, test set | 0.135 | 1,409 test customers |
| ROC-AUC of stored out-of-fold probabilities | 0.847 | all 7,043 |
| Mean predicted churn vs actual churn rate | 0.2657 vs 0.2654 | all 7,043 |

The strongest churn drivers are **tenure and lifetime spend**, then **contract type**, and
well behind those, protective add-ons (Online Security / Tech Support) and internet service
type. All four groups lower ROC-AUC when shuffled.

SHAP over all 7,043 customers agrees on the top four but orders them differently: **contract
type** moves individual risk scores the most (26% of mean |SHAP|), then tenure & lifetime spend
(23%), protective add-ons (13%) and internet service type (13%). The two measures answer
different questions; see [Method notes](#method-notes-and-limitations).

### Retention matrix: action counts (all 7,043 customers)

Each customer is placed by median splits on 12-month LTV (**> $844.20**) and calibrated churn
probability (**> 0.1833**). A customer exactly at a median counts as "low".

| | Low churn risk | High churn risk |
|---|---|---|
| **High LTV** | Early Access / Upsell: **1,256** | Retention Offer: **2,259** |
| **Low LTV** | Nurture: **2,268** | Monitor Only: **1,260** |

### Revenue at risk and projected savings

Revenue at risk = calibrated churn probability × 12-month LTV (monthly charges × 12), i.e.
expected 12-month revenue lost to churn. Projected savings = success rate × the Retention Offer
group's revenue at risk.

| | **Active customers only (headline)** | All customers (incl. churned) |
|---|---:|---:|
| Customers | 5,174 | 7,043 |
| 12-month revenue at risk | **$807,576** | $1,666,382 |
| Retention Offer customers | **1,123** | 2,259 |
| Retention Offer share of revenue at risk | 62.5% | 72.0% |
| Savings at 10% success | **$50,493** (≈47 customers kept) | $120,039 |
| Savings at 20% success | **$100,986** (≈94 customers kept) | $240,078 |
| Savings at 30% success | **$151,480** (≈141 customers kept) | $360,118 |

Quote the active-only column for business impact. The all-customer column matches
`python -m src.retention_matrix` output and is kept for reconciliation.

---

## Architecture

```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
        │
        ▼  database/seed.py ── clean, derive num_services + estimated_ltv, idempotent upsert
┌──────────────────────── PostgreSQL (star schema) ────────────────────────┐
│ dim_customer   dim_service   dim_contract   fact_subscription            │
│                                                                          │
│ customer_segments  ◄── src/segmentation.py     RFM scores, segment       │
│                    ◄── src/churn_model.py      churn_probability         │
│                    ◄── src/retention_matrix.py retention_action          │
└──────────────────────────────────────────────────────────────────────────┘
        │                                   models/churn_model.pkl
        ▼                                          │
dashboard/app.py (Streamlit) ◄─────────────────────┤  what-if simulator, importance
        │                                          └── src/explainability.py (SHAP)
        └── src/ask_subscribeiq.py ──► Gemini: question → read-only SQL → answer
```

- **Every dashboard number is queried live** from PostgreSQL and cached for 5 minutes. The
  sidebar has a "Refresh data" button. The only file read from disk is the trained model.
- **Ask SubscribeIQ is the only page that calls an external service.** It sends the question,
  the schema description and up to 60 result rows to the Gemini API.
- **`customer_segments`** is the analytics output table. Each phase writes only its own
  columns, so re-running one phase never wipes another's results. Re-seeding never touches it.

### Schema

| Table | Grain | Contents |
|---|---|---|
| `dim_customer` | 1 row / customer | gender, senior citizen, partner, dependents |
| `dim_service` | 1 row / customer | phone, lines, internet type, six add-ons |
| `dim_contract` | 1 row / customer | contract type, payment method, paperless billing |
| `fact_subscription` | 1 row / customer | tenure, monthly / total charges, churn, `num_services`, `estimated_ltv` |
| `customer_segments` | 1 row / customer | R/F/M scores, `rfm_combined`, cluster, `segment_name`, `churn_probability`, `retention_action` |

The full DDL is in [`database/schema.sql`](database/schema.sql).

### Repository layout

```
database/   schema.sql, seed.py (ETL)
src/        db_connection.py, segmentation.py, churn_model.py, retention_matrix.py,
            explainability.py (SHAP), ask_subscribeiq.py (Gemini Q&A)
dashboard/  app.py (Streamlit)
notebooks/  01_eda, 02_rfm_segmentation, 03_churn_modeling
models/     churn_model.pkl (calibrated model + metadata, committed)
tests/      unit tests, dashboard smoke tests, end-to-end audit and dashboard reconciliation
```

---

## Setup

**Prerequisites:** Python 3.11 and PostgreSQL (developed on 17), plus an empty database
(default name `subscribeiq`).

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt

cp .env.example .env             # then fill in DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

`GEMINI_API_KEY` is only needed for the Ask SubscribeIQ page; every other page works without
it. The default model is `gemini-3.5-flash`, falling back to `gemini-flash-latest` when it is
busy. Set `GEMINI_MODEL` to pin a different one.

### Build the data (run in order, from the project root)

```bash
python database/seed.py          # 1. schema + load 7,043 customers (safe to re-run)
python -m src.segmentation       # 2. RFM scores + KMeans segments -> customer_segments
python -m src.churn_model        # 3. tune, calibrate, write probabilities, save models/churn_model.pkl
python -m src.retention_matrix   # 4. assign retention actions, print matrix and savings
```

Optionally, `python -m src.explainability` prints global SHAP importance and one example
customer's explanation (read-only).

The order matters because each step reads the previous step's output. The churn model uses the
RFM scores, and the retention matrix uses the probabilities. Step 3 grid-searches three model
families and **overwrites the committed `models/churn_model.pkl`**. Skip it unless you mean to
retrain.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

Then open <http://localhost:8501>. The pages are:

| Page | What it shows |
|---|---|
| Overview | Portfolio KPIs, segment mix, churn rate by segment |
| Segment Explorer | Per-segment RFM profile, action mix, tenure-vs-charges scatter, strategy |
| Customer Lookup | One customer's risk gauge, recommended action, revenue at risk, account details |
| Churn Drivers | Grouped feature importance and a what-if simulator on the saved model |
| SHAP Explanations | Waterfall of why one customer is scored as they are (defaults to the Customer Lookup customer); global mean \|SHAP\| by feature or group |
| Business Impact | Active-customer revenue at risk, savings scenarios, cost-based targeting, segment × action table |
| Ask SubscribeIQ | Plain-English questions answered from the warehouse by Gemini; every answered question shows its SQL and result rows |

## Run the tests

```bash
pytest -q
```

There are 103 tests. Tests that need PostgreSQL skip automatically when it is unreachable. One
test calls the real Gemini API and runs only when `RUN_GEMINI_TESTS=1`. Every other Ask
SubscribeIQ test uses a scripted stand-in for the model, so the default run makes no API calls.

| File | Covers |
|---|---|
| `test_seed.py`, `test_segmentation.py`, `test_churn_model.py`, `test_retention_matrix.py` | Unit tests for each pipeline module |
| `test_explainability.py` | SHAP values add up exactly to the model's log-odds score for every customer; margins map exactly to the calibrated probabilities; global and per-customer summaries |
| `test_ask_subscribeiq.py` | SQL guard accepts reads and rejects writes, multiple statements, catalogs and comments; read-only transaction enforced by PostgreSQL; query-fix retry, declined questions, row cap |
| `test_dashboard.py` | Every page renders without errors; lookup, what-if and SHAP page behaviour |
| `test_audit.py` | End-to-end audit: raw CSV equals the warehouse row for row and to the cent; stored RFM scores, segments and actions reproduce exactly; hand-traced customers; calibration and decile ranking; revenue-at-risk arithmetic; pinned headline figures |
| `test_dashboard_reconciliation.py` | Every metric, table cell and chart value on the five Phase 6 pages equals an independent SQL calculation |

The committed headline figures are pinned in the `EXPECTED` block of `tests/test_audit.py`. If a
deliberate re-run of a pipeline phase changes them, update that block and this README in the
same commit.

---

## Pipeline phases

| Phase | Deliverable |
|---|---|
| 1. Setup & ETL | Star schema; idempotent upsert ETL. The 11 blank `TotalCharges` rows are brand-new tenure-0 customers, so they are set to 0 rather than imputed |
| 2. EDA | `notebooks/01_eda.ipynb`: churn by contract, tenure, internet type, add-ons, payment method |
| 3. Segmentation | Quintile RFM scores, K diagnostics (elbow, silhouette, seed stability), K = 4 KMeans, segments named by profile rather than by label number |
| 4. Churn model | LogisticRegression, RandomForest and GradientBoosting compared with balanced class weights, then isotonic calibration. Stored probabilities are nested out-of-fold, so no customer is scored by a model that saw their outcome |
| 5. Retention matrix | Median-split risk × value quadrants, revenue at risk, 10/20/30% savings scenarios, cost-based 1/(1+r) targeting helper |
| 6. Dashboard | Five-page Streamlit app over live PostgreSQL |
| 7. Documentation | This README |
| 8. Explainability & Q&A | `src/explainability.py`: exact interventional TreeSHAP on the deployed model, one-hot columns summed back to business features; "Ask SubscribeIQ": Gemini writes one SQL query, it runs read-only, Gemini answers from the rows |

## Method notes and limitations

- **RFM proxies.** The data has no purchase timestamps. Recency is tenure (5 = most
  established), Frequency is the number of active services, and Monetary is total charges.
  Because total charges ≈ monthly charges × tenure, M is strongly correlated with R
  (Spearman ≈ 0.89), so the segments lean on tenure.
- **LTV is a 12-month window** (monthly charges × 12). It does not capture growth potential,
  which matters for the Monitor Only group, mostly new low-price customers.
- **Probabilities are calibrated and clipped to [0.5%, 99.5%].** Isotonic calibration can
  output exactly 0 or 1, so values are clipped and no customer is presented as certain to stay
  or leave. The mean predicted churn matches the actual rate (0.2657 vs 0.2654).
- **Stored vs live probabilities.** The database holds out-of-fold probabilities. The what-if
  simulator uses the saved model refit on all customers, so a customer's baseline there can
  differ slightly from Customer Lookup (mean absolute gap ≈ 0.03).
- **Median ties.** 82 customers sit exactly at the median churn probability. The rule is
  strictly "above the median", so they count as low risk, which is why the high-risk quadrants
  are slightly smaller than the low-risk ones.
- **What SHAP explains.** The deployed model's probability is isotonic(margin), where margin is
  the gradient-boosting log-odds score. SHAP splits that margin exactly (base value + contributions
  = margin), so both ends of a waterfall convert exactly to calibrated probabilities. The bars
  themselves are in log-odds. Calibration is monotonic, so a bar never points the wrong way,
  but its size in percentage points depends on the customer's starting point. The base value is
  the average of 200 randomly sampled real customers (15.0% at the average score), not the
  class-balanced training baseline. Like the what-if simulator, SHAP explains the deployed model,
  not the stored out-of-fold probability.
- **SHAP vs permutation importance.** SHAP measures how far a feature moves individual scores.
  Permutation importance measures how much ranking accuracy is lost when it is shuffled. Tenure,
  total billed and the R/M scores carry overlapping information, so SHAP splits the credit
  between them and they rank lower individually than as a group.
- **Ask SubscribeIQ guardrails.** Generated SQL must be a single SELECT over the five SubscribeIQ
  tables, with no comments, system catalogs or sequence changes (`nextval`, `setval`), and it
  may only call allowlisted aggregate, math, string and window functions. Everything else is
  rejected, including `query_to_xml` (which would run SQL hidden in a string),
  `database_to_xml`, `current_setting` and `version`. It runs inside a `READ ONLY`
  transaction with a 5-second timeout and a 500-row cap, so PostgreSQL itself refuses writes
  even if the check is bypassed. Money questions default to active customers and everything
  else to all customers, and each answer states its population. Answers come from a language
  model, so check the SQL shown with each one before quoting a figure. For a shared deployment,
  also connect with a database role that has only SELECT on these tables.
- **Snapshot data.** Churned customers are still in the snapshot. They are scored (for model
  evaluation) and assigned actions (for reference), but they are excluded from the business
  headline. See [Headline results](#headline-results).

## Data

IBM Telco Customer Churn sample dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`, 7,043 rows,
21 columns), stored in `data/raw/`.
