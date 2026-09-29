"""Churn prediction: feature preparation, model comparison, scoring and persistence.

Three classifiers are compared (LogisticRegression, RandomForestClassifier,
GradientBoostingClassifier), each wrapped in one scikit-learn Pipeline with the exact
preprocessing, so the saved artifact can be reused as-is by the dashboard and by
src/explainability.py (SHAP).

Class imbalance (~26.5% churn) is handled with balanced class weights for all three models
(sample weights for GradientBoosting, which has no class_weight parameter).

Run as a script to train, evaluate, write out-of-fold probabilities and save the model:
    python -m src.churn_model
"""

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "gender", "senior_citizen", "partner", "dependents",
    "phone_service", "multiple_lines", "internet_service", "online_security",
    "online_backup", "device_protection", "tech_support", "streaming_tv",
    "streaming_movies", "contract_type", "payment_method", "paperless_billing",
]
NUMERIC_FEATURES = [
    "tenure_months", "monthly_charges", "total_charges", "num_services",
    "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "churn"

MODEL_SPECS = {
    "LogisticRegression": (
        LogisticRegression(class_weight="balanced", max_iter=5000),
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=300, class_weight="balanced",
                               random_state=RANDOM_STATE, n_jobs=-1),
        {"model__max_depth": [6, 10, None], "model__min_samples_leaf": [1, 5, 20]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]},
    ),
}
# Models without a class_weight parameter get balanced sample weights at fit time instead
NEEDS_SAMPLE_WEIGHT = {"GradientBoosting"}


def load_modeling_data(engine: Engine) -> pd.DataFrame:
    """Load one row per customer with every model feature and the churn target.

    Joins the star schema with the Phase 3 RFM scores in customer_segments. Boolean
    columns are cast to "Yes"/"No" text so every categorical feature is encoded the same way.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.

    Returns:
        Dataframe with customer_id, FEATURES and the churn target (int 0/1).

    Raises:
        ValueError: If any customer is missing RFM scores (run src.segmentation first).
    """
    query = """
        SELECT c.customer_id,
               c.gender,
               CASE WHEN c.senior_citizen   THEN 'Yes' ELSE 'No' END AS senior_citizen,
               CASE WHEN c.partner          THEN 'Yes' ELSE 'No' END AS partner,
               CASE WHEN c.dependents       THEN 'Yes' ELSE 'No' END AS dependents,
               CASE WHEN s.phone_service    THEN 'Yes' ELSE 'No' END AS phone_service,
               s.multiple_lines, s.internet_service, s.online_security, s.online_backup,
               s.device_protection, s.tech_support, s.streaming_tv, s.streaming_movies,
               k.contract_type, k.payment_method,
               CASE WHEN k.paperless_billing THEN 'Yes' ELSE 'No' END AS paperless_billing,
               f.tenure_months,
               f.monthly_charges::float AS monthly_charges,
               f.total_charges::float   AS total_charges,
               f.num_services,
               cs.rfm_recency_score, cs.rfm_frequency_score, cs.rfm_monetary_score,
               f.churn::int AS churn
        FROM dim_customer c
        JOIN dim_service s        USING (customer_id)
        JOIN dim_contract k       USING (customer_id)
        JOIN fact_subscription f  USING (customer_id)
        LEFT JOIN customer_segments cs USING (customer_id)
        ORDER BY c.customer_id
    """
    df = pd.read_sql(text(query), engine)
    if df[["rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score"]].isna().any().any():
        raise ValueError("Missing RFM scores in customer_segments; run `python -m src.segmentation`")
    return df


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categoricals (binary columns collapse to one indicator) and scale numerics.

    Scaling matters for LogisticRegression and is harmless for the tree models, so all three
    models share one preprocessing definition.

    Returns:
        An unfitted ColumnTransformer.
    """
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ], verbose_feature_names_out=False)


def build_pipeline(estimator) -> Pipeline:
    """Wrap an estimator with the shared preprocessing into a single Pipeline.

    Args:
        estimator: An unfitted scikit-learn classifier.

    Returns:
        Pipeline with steps "prep" and "model".
    """
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split of the modelling data.

    Args:
        df: Output of load_modeling_data.
        test_size: Fraction held out for testing.

    Returns:
        X_train, X_test, y_train, y_test.
    """
    return train_test_split(df[FEATURES], df[TARGET], test_size=test_size,
                            stratify=df[TARGET], random_state=RANDOM_STATE)


def _fit_params(name: str, y: pd.Series) -> dict:
    """Return balanced sample weights for models that need them, else no extra fit params."""
    if name in NEEDS_SAMPLE_WEIGHT:
        return {"model__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def tune_models(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> dict:
    """Grid-search each model with stratified k-fold CV on the training set only (ROC-AUC).

    Args:
        X_train: Training features.
        y_train: Training target.
        cv_folds: Number of CV folds.

    Returns:
        Mapping of model name to the fitted GridSearchCV object (best model refit on X_train).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    searches = {}
    for name, (estimator, grid) in MODEL_SPECS.items():
        search = GridSearchCV(build_pipeline(estimator), grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train, **_fit_params(name, y_train))
        searches[name] = search
    return searches


def evaluate(model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    """Compute Accuracy, Precision, Recall, F1 and ROC-AUC at a given decision threshold.

    Args:
        model: Fitted pipeline with predict_proba.
        X: Features.
        y: True labels.
        threshold: Probability at or above which a customer is predicted to churn.

    Returns:
        Dictionary of metric name to value.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def compare_models(searches: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Build the model comparison table on the held-out test set.

    Args:
        searches: Output of tune_models.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        One row per model with CV ROC-AUC, test metrics and best hyperparameters,
        sorted by test ROC-AUC then recall (the agreed selection rule).
    """
    rows = []
    for name, search in searches.items():
        metrics = evaluate(search.best_estimator_, X_test, y_test)
        rows.append({"model": name, "cv_roc_auc": search.best_score_, **metrics,
                     "best_params": {k.replace("model__", ""): v
                                     for k, v in search.best_params_.items()}})
    table = pd.DataFrame(rows).set_index("model")
    return table.sort_values(["roc_auc", "recall"], ascending=False)


def threshold_table(y_true: pd.Series, proba: np.ndarray, thresholds=None) -> pd.DataFrame:
    """Precision, recall and confusion counts across decision thresholds.

    Args:
        y_true: True labels.
        proba: Predicted churn probabilities.
        thresholds: Thresholds to evaluate (default 0.10 to 0.90 in 0.05 steps).

    Returns:
        One row per threshold with tp, fp, fn, tn, precision, recall and flagged share.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": tp / (tp + fp) if tp + fp else np.nan,
                     "recall": tp / (tp + fn), "flagged_pct": 100 * (tp + fp) / len(pred)})
    return pd.DataFrame(rows).set_index("threshold")


def permutation_importances(model, X: pd.DataFrame, y: pd.Series,
                            n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance of each *original* feature, measured as drop in ROC-AUC.

    Because the pipeline takes raw columns, each business feature (e.g. contract_type) is
    shuffled as a whole rather than one one-hot column at a time.

    Args:
        model: Fitted pipeline.
        X: Held-out features.
        y: Held-out labels.
        n_repeats: Shuffles per feature.

    Returns:
        Dataframe with importance_mean and importance_std, sorted descending.
    """
    result = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    return (pd.DataFrame({"importance_mean": result.importances_mean,
                          "importance_std": result.importances_std}, index=X.columns)
            .sort_values("importance_mean", ascending=False))


FEATURE_GROUPS = {
    "Tenure & lifetime spend": ["tenure_months", "total_charges",
                                "rfm_recency_score", "rfm_monetary_score"],
    "Contract type": ["contract_type"],
    "Internet service type": ["internet_service"],
    "Monthly price": ["monthly_charges"],
    "Protective add-ons": ["online_security", "tech_support", "online_backup", "device_protection"],
    "Streaming add-ons": ["streaming_tv", "streaming_movies"],
    "Payment & billing": ["payment_method", "paperless_billing"],
    "Service breadth": ["num_services", "rfm_frequency_score"],
    "Phone services": ["phone_service", "multiple_lines"],
    "Demographics": ["gender", "senior_citizen", "partner", "dependents"],
}


def grouped_permutation_importances(model, X: pd.DataFrame, y: pd.Series,
                                    groups: dict = FEATURE_GROUPS,
                                    n_repeats: int = 10) -> pd.DataFrame:
    """Permutation importance for groups of related features, shuffled together.

    Correlated features (e.g. tenure, total_charges and the R/M scores) share signal: shuffling
    one alone barely hurts because the others still carry it, so single-feature importance
    understates the group. Shuffling the whole group with one permutation measures the
    information the group carries jointly.

    Args:
        model: Fitted pipeline.
        X: Held-out features.
        y: Held-out labels.
        groups: Mapping of group name to list of feature columns; must cover every feature.
        n_repeats: Shuffles per group.

    Returns:
        Dataframe with importance_mean and importance_std (drop in ROC-AUC), sorted descending.

    Raises:
        ValueError: If the groups do not cover each feature exactly once.
    """
    covered = [c for cols in groups.values() for c in cols]
    if sorted(covered) != sorted(X.columns):
        raise ValueError("Feature groups must cover each feature exactly once")

    rng = np.random.default_rng(RANDOM_STATE)
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            idx = rng.permutation(len(X))
            shuffled[cols] = X[cols].iloc[idx].to_numpy()
            drops.append(baseline - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        rows[name] = {"importance_mean": np.mean(drops), "importance_std": np.std(drops)}
    return pd.DataFrame(rows).T.sort_values("importance_mean", ascending=False)


def builtin_importances(model) -> pd.Series:
    """Model-native importances on the encoded features (impurity importance or |coefficient|).

    Args:
        model: Fitted pipeline.

    Returns:
        Series indexed by encoded feature name, sorted descending.
    """
    names = model.named_steps["prep"].get_feature_names_out()
    est = model.named_steps["model"]
    values = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_[0])
    return pd.Series(values, index=names).sort_values(ascending=False)


def out_of_fold_probabilities(name: str, best_params: dict, df: pd.DataFrame,
                              cv_folds: int = 5) -> np.ndarray:
    """Churn probability for every customer from a model that never saw that customer.

    Args:
        name: Model name (key of MODEL_SPECS).
        best_params: Tuned hyperparameters (with the "model__" prefix).
        df: Full modelling data.
        cv_folds: Number of folds.

    Returns:
        Array of out-of-fold churn probabilities aligned with df.
    """
    estimator, _ = MODEL_SPECS[name]
    pipeline = build_pipeline(clone(estimator)).set_params(**best_params)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipeline, df[FEATURES], df[TARGET], cv=cv, method="predict_proba",
                             params=_fit_params(name, df[TARGET]))[:, 1]


def fit_final_model(name: str, best_params: dict, df: pd.DataFrame) -> Pipeline:
    """Refit the chosen model and its preprocessing on all customers.

    Args:
        name: Model name (key of MODEL_SPECS).
        best_params: Tuned hyperparameters (with the "model__" prefix).
        df: Full modelling data.

    Returns:
        Fitted Pipeline.
    """
    estimator, _ = MODEL_SPECS[name]
    pipeline = build_pipeline(clone(estimator)).set_params(**best_params)
    return pipeline.fit(df[FEATURES], df[TARGET], **_fit_params(name, df[TARGET]))


def save_model(pipeline: Pipeline, name: str, best_params: dict, test_metrics: dict,
               path: Path = MODEL_PATH) -> Path:
    """Persist the fitted pipeline plus the metadata needed to reuse it.

    Args:
        pipeline: Fitted pipeline (preprocessing + model).
        name: Model name.
        best_params: Tuned hyperparameters.
        test_metrics: Held-out test metrics of the tuned model.
        path: Destination .pkl path.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": pipeline,
        "model_name": name,
        "best_params": best_params,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "features": FEATURES,
        "test_metrics": test_metrics,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, path)
    return path


def load_model(path: Path = MODEL_PATH) -> dict:
    """Load the saved model artifact (a dict with "pipeline", "features" and metadata).

    Args:
        path: Path to churn_model.pkl.

    Returns:
        The artifact dictionary written by save_model.
    """
    return joblib.load(path)


def write_probabilities(engine: Engine, customer_ids, probabilities) -> int:
    """Write churn_probability into customer_segments for existing rows.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.
        customer_ids: Customer IDs.
        probabilities: Churn probabilities aligned with customer_ids.

    Returns:
        Number of rows with a non-null churn_probability after the update.
    """
    stmt = text("UPDATE customer_segments SET churn_probability = :p WHERE customer_id = :cid")
    stmt = stmt.bindparams(bindparam("p"), bindparam("cid"))
    records = [{"cid": c, "p": round(float(p), 5)} for c, p in zip(customer_ids, probabilities)]
    with engine.begin() as conn:
        conn.execute(stmt, records)
        return conn.execute(text(
            "SELECT COUNT(churn_probability) FROM customer_segments")).scalar_one()


def run_pipeline(engine: Engine) -> dict:
    """Train, compare, score and persist; returns every intermediate result for reporting.

    Args:
        engine: SQLAlchemy engine for the SubscribeIQ database.

    Returns:
        Dictionary with data, splits, searches, comparison table, best model name,
        out-of-fold probabilities, final pipeline and model path.
    """
    df = load_modeling_data(engine)
    X_train, X_test, y_train, y_test = split_data(df)
    searches = tune_models(X_train, y_train)
    comparison = compare_models(searches, X_test, y_test)
    best_name = comparison.index[0]
    best_params = searches[best_name].best_params_

    oof = out_of_fold_probabilities(best_name, best_params, df)
    final = fit_final_model(best_name, best_params, df)
    test_metrics = comparison.loc[best_name].drop(["best_params", "cv_roc_auc"]).to_dict()
    path = save_model(final, best_name, best_params, test_metrics)
    written = write_probabilities(engine, df["customer_id"], oof)

    return {"df": df, "X_train": X_train, "X_test": X_test, "y_train": y_train,
            "y_test": y_test, "searches": searches, "comparison": comparison,
            "best_name": best_name, "best_params": best_params, "oof": oof,
            "final": final, "model_path": path, "rows_written": written}


def main() -> None:
    """Run the full churn-model pipeline and print a summary."""
    from src.db_connection import get_engine

    results = run_pipeline(get_engine())
    pd.set_option("display.width", 200)
    print(results["comparison"].drop(columns="best_params").round(4).to_string())
    print(f"\nBest model: {results['best_name']}  params: {results['best_params']}")
    print(f"Saved: {results['model_path']} ({results['model_path'].stat().st_size / 1e6:.2f} MB)")
    print(f"churn_probability written for {results['rows_written']} customers")


if __name__ == "__main__":
    main()
