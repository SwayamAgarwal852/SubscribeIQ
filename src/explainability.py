"""SHAP explanations of the deployed churn model.

The saved artifact is a CalibratedClassifierCV(ensemble=False): calibrated probability =
clip(isotonic(margin)), where margin is the log-odds output (decision_function) of the base
GradientBoosting pipeline. SHAP's TreeExplainer explains that margin exactly, so:

    margin(customer) = base value + sum of the customer's SHAP values

and both ends of an explanation map exactly onto calibrated probabilities through the same
isotonic step (to_probability). Individual SHAP values are in log-odds units; isotonic
calibration is monotonic, so a positive SHAP value always pushes the calibrated probability
up (or leaves it on the same flat step), never down.

Explanations are interventional against a fixed random background of real customers, so the
base value is the average customer's margin, not the class-balanced training baseline.
One-hot columns are summed back to the original business feature (e.g. contract_type), so
every customer has exactly one SHAP value per model feature.

Run as a script to print global SHAP importance and one example customer:
    python -m src.explainability
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.churn_model import (CATEGORICAL_FEATURES, FEATURE_GROUPS, FEATURES, RANDOM_STATE,
                             clip_probabilities)

BACKGROUND_SIZE = 200

FEATURE_LABELS = {
    "gender": "Gender", "senior_citizen": "Senior citizen", "partner": "Has partner",
    "dependents": "Has dependents", "phone_service": "Phone service",
    "multiple_lines": "Multiple lines", "internet_service": "Internet service",
    "online_security": "Online Security", "online_backup": "Online Backup",
    "device_protection": "Device Protection", "tech_support": "Tech Support",
    "streaming_tv": "Streaming TV", "streaming_movies": "Streaming Movies",
    "contract_type": "Contract", "payment_method": "Payment method",
    "paperless_billing": "Paperless billing", "tenure_months": "Tenure (months)",
    "monthly_charges": "Monthly charges", "total_charges": "Total billed",
    "num_services": "Number of services", "rfm_recency_score": "Recency score",
    "rfm_frequency_score": "Frequency score", "rfm_monetary_score": "Monetary score",
}


@dataclass
class ChurnExplainer:
    """A fitted TreeExplainer plus what is needed to explain raw customer rows.

    Attributes:
        base_pipeline: Uncalibrated preprocessing + GradientBoosting pipeline.
        calibrated: The calibrated model (for mapping margins to probabilities).
        explainer: shap.TreeExplainer on the base model, interventional.
        encoded_to_feature: Original feature name for each encoded column.
    """
    base_pipeline: Pipeline
    calibrated: CalibratedClassifierCV
    explainer: shap.TreeExplainer
    encoded_to_feature: list

    @property
    def base_value(self) -> float:
        """Average background customer's margin (log-odds) — the start of every explanation."""
        return float(np.ravel(self.explainer.expected_value)[0])


def encoded_feature_map(base_pipeline: Pipeline) -> list:
    """Map each encoded column of the preprocessor back to its original feature.

    Categorical columns are named "<feature>_<category>"; the longest matching feature prefix
    wins. Numeric columns keep their name.

    Args:
        base_pipeline: Fitted pipeline with a "prep" ColumnTransformer.

    Returns:
        List of original feature names, one per encoded column.

    Raises:
        ValueError: If an encoded column matches no model feature.
    """
    mapping = []
    for name in base_pipeline.named_steps["prep"].get_feature_names_out():
        if name in FEATURES:
            mapping.append(name)
            continue
        matches = [f for f in CATEGORICAL_FEATURES if name.startswith(f + "_")]
        if not matches:
            raise ValueError(f"Encoded column {name!r} matches no model feature")
        mapping.append(max(matches, key=len))
    return mapping


def build_explainer(artifact: dict, X_reference: pd.DataFrame,
                    background_size: int = BACKGROUND_SIZE) -> ChurnExplainer:
    """Build an interventional TreeExplainer for the saved model.

    Args:
        artifact: Model artifact from churn_model.load_model (needs "pipeline", "base_pipeline").
        X_reference: Raw feature rows to draw the background sample from (all customers).
        background_size: Number of background customers (fixed seed).

    Returns:
        A ChurnExplainer.
    """
    base = artifact["base_pipeline"]
    background = X_reference[FEATURES].sample(min(background_size, len(X_reference)),
                                              random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(base.named_steps["model"],
                                   data=base.named_steps["prep"].transform(background),
                                   feature_perturbation="interventional",
                                   model_output="raw")
    return ChurnExplainer(base, artifact["pipeline"], explainer, encoded_feature_map(base))


def shap_values(ce: ChurnExplainer, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP value of each original feature for each customer (log-odds units).

    Args:
        ce: ChurnExplainer.
        X: Raw feature rows.

    Returns:
        Dataframe indexed like X with one column per model feature (FEATURES order).
        Each row sums to (margin - base value).
    """
    encoded = ce.base_pipeline.named_steps["prep"].transform(X[FEATURES])
    values = np.asarray(ce.explainer.shap_values(encoded, check_additivity=True))
    per_encoded = pd.DataFrame(values, index=X.index, columns=ce.encoded_to_feature)
    return per_encoded.T.groupby(level=0).sum().T[FEATURES]


def margins(ce: ChurnExplainer, X: pd.DataFrame) -> np.ndarray:
    """Base-model log-odds margin (the input to the isotonic calibrator)."""
    return ce.base_pipeline.decision_function(X[FEATURES])


def to_probability(ce: ChurnExplainer, margin) -> np.ndarray:
    """Map margins to calibrated, clipped probabilities (same as churn_model.predict_churn).

    Args:
        ce: ChurnExplainer.
        margin: Scalar or array of log-odds margins.

    Returns:
        Calibrated probabilities, same shape as the input.
    """
    isotonic = ce.calibrated.calibrated_classifiers_[0].calibrators[0]
    return clip_probabilities(isotonic.predict(np.atleast_1d(np.asarray(margin, dtype=float))))


def global_importance(shap_df: pd.DataFrame, groups: dict = None) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, or per feature group.

    For groups, each customer's SHAP values are summed within the group first, so features
    that push in opposite directions partly cancel (the group's net effect per customer).

    Args:
        shap_df: Output of shap_values.
        groups: Optional mapping of group name to feature list (e.g. FEATURE_GROUPS).

    Returns:
        Dataframe with mean_abs_shap and share (of the total), sorted descending.
    """
    if groups is not None:
        shap_df = pd.DataFrame({g: shap_df[cols].sum(axis=1) for g, cols in groups.items()})
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"mean_abs_shap": mean_abs, "share": mean_abs / mean_abs.sum()})


def format_value(feature: str, value) -> str:
    """Human-readable value of a raw model feature."""
    if feature in ("monthly_charges", "total_charges"):
        return f"${float(value):,.2f}"
    if feature == "tenure_months":
        return f"{int(value)} month{'' if int(value) == 1 else 's'}"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def customer_explanation(shap_row: pd.Series, feature_row: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """One customer's top SHAP contributions, with the rest folded into one "other" row.

    Args:
        shap_row: One row of shap_values.
        feature_row: The same customer's raw feature row.
        top_n: Number of individual features to keep.

    Returns:
        Dataframe with feature, label, value (display string) and shap, ordered by |shap|
        descending; the last row is "Other N features" when anything was folded.
    """
    order = shap_row.abs().sort_values(ascending=False).index
    rows = [{"feature": f, "label": FEATURE_LABELS[f], "value": format_value(f, feature_row[f]),
             "shap": float(shap_row[f])} for f in order[:top_n]]
    rest = order[top_n:]
    if len(rest):
        rows.append({"feature": "other", "label": f"Other {len(rest)} features", "value": "",
                     "shap": float(shap_row[rest].sum())})
    return pd.DataFrame(rows)


def main() -> None:
    """Print global SHAP importance over all customers and one example explanation."""
    from src.churn_model import load_model, load_modeling_data
    from src.db_connection import get_engine

    data = load_modeling_data(get_engine())
    X = data.set_index("customer_id")[FEATURES]
    ce = build_explainer(load_model(), X)
    values = shap_values(ce, X)
    pd.set_option("display.width", 200)
    print(f"Base value (average customer): margin {ce.base_value:+.3f} "
          f"-> calibrated {to_probability(ce, ce.base_value)[0]:.1%}\n")
    print("Mean |SHAP| by feature group (all customers):")
    print(global_importance(values, FEATURE_GROUPS).round(4).to_string())
    print("\nMean |SHAP| by feature, top 10:")
    print(global_importance(values).head(10).round(4).to_string())

    cid = X.index[0]
    margin = margins(ce, X.loc[[cid]])[0]
    print(f"\nExample {cid}: calibrated {to_probability(ce, margin)[0]:.1%} (margin {margin:+.3f})")
    print(customer_explanation(values.loc[cid], X.loc[cid]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
