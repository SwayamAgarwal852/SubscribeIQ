-- SubscribeIQ star schema (PostgreSQL)
-- Safe to re-run: every object uses IF NOT EXISTS.
--
-- Type conventions:
--   * Pure Yes/No source columns        -> BOOLEAN
--   * Three-state service columns        -> TEXT with CHECK, keeping the source labels
--     ("No phone service" / "No internet service" carry meaning beyond a plain "No")

-- ---------------------------------------------------------------
-- Dimension: customer demographics
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_id     VARCHAR(20) PRIMARY KEY,
    gender          TEXT    NOT NULL CHECK (gender IN ('Male', 'Female')),
    senior_citizen  BOOLEAN NOT NULL,
    partner         BOOLEAN NOT NULL,
    dependents      BOOLEAN NOT NULL
);

-- ---------------------------------------------------------------
-- Dimension: subscribed services (one row per customer)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_service (
    service_id        SERIAL PRIMARY KEY,
    customer_id       VARCHAR(20) NOT NULL UNIQUE
                      REFERENCES dim_customer (customer_id) ON DELETE CASCADE,
    phone_service     BOOLEAN NOT NULL,
    multiple_lines    TEXT NOT NULL CHECK (multiple_lines    IN ('Yes', 'No', 'No phone service')),
    internet_service  TEXT NOT NULL CHECK (internet_service  IN ('DSL', 'Fiber optic', 'No')),
    online_security   TEXT NOT NULL CHECK (online_security   IN ('Yes', 'No', 'No internet service')),
    online_backup     TEXT NOT NULL CHECK (online_backup     IN ('Yes', 'No', 'No internet service')),
    device_protection TEXT NOT NULL CHECK (device_protection IN ('Yes', 'No', 'No internet service')),
    tech_support      TEXT NOT NULL CHECK (tech_support      IN ('Yes', 'No', 'No internet service')),
    streaming_tv      TEXT NOT NULL CHECK (streaming_tv      IN ('Yes', 'No', 'No internet service')),
    streaming_movies  TEXT NOT NULL CHECK (streaming_movies  IN ('Yes', 'No', 'No internet service'))
);

-- ---------------------------------------------------------------
-- Dimension: contract & billing (one row per customer)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_contract (
    contract_id        SERIAL PRIMARY KEY,
    customer_id        VARCHAR(20) NOT NULL UNIQUE
                       REFERENCES dim_customer (customer_id) ON DELETE CASCADE,
    contract_type      TEXT NOT NULL CHECK (contract_type IN ('Month-to-month', 'One year', 'Two year')),
    payment_method     TEXT NOT NULL,
    paperless_billing  BOOLEAN NOT NULL
);

-- ---------------------------------------------------------------
-- Fact: subscription metrics (one row per customer)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_subscription (
    subscription_id  SERIAL PRIMARY KEY,
    customer_id      VARCHAR(20) NOT NULL UNIQUE
                     REFERENCES dim_customer (customer_id) ON DELETE CASCADE,
    tenure_months    INTEGER       NOT NULL CHECK (tenure_months >= 0),
    monthly_charges  NUMERIC(10,2) NOT NULL CHECK (monthly_charges >= 0),
    total_charges    NUMERIC(10,2) NOT NULL CHECK (total_charges >= 0),
    churn            BOOLEAN       NOT NULL,
    num_services     SMALLINT      NOT NULL CHECK (num_services BETWEEN 0 AND 9),
    estimated_ltv    NUMERIC(12,2) NOT NULL CHECK (estimated_ltv >= 0)
);

-- ---------------------------------------------------------------
-- Analytics output: populated in Phases 3-5 (empty after Phase 1)
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customer_segments (
    customer_id          VARCHAR(20) PRIMARY KEY
                         REFERENCES dim_customer (customer_id) ON DELETE CASCADE,
    rfm_recency_score    SMALLINT CHECK (rfm_recency_score   BETWEEN 1 AND 5),
    rfm_frequency_score  SMALLINT CHECK (rfm_frequency_score BETWEEN 1 AND 5),
    rfm_monetary_score   SMALLINT CHECK (rfm_monetary_score  BETWEEN 1 AND 5),
    rfm_combined         TEXT,
    cluster_label        SMALLINT,
    segment_name         TEXT,
    churn_probability    NUMERIC(6,5) CHECK (churn_probability BETWEEN 0 AND 1),
    retention_action     TEXT
);
