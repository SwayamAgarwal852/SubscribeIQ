"""Tests for src/segmentation.py (no database required)."""

import numpy as np
import pandas as pd
import pytest

from src.segmentation import SCORE_COLUMNS, name_clusters, quintile_score, score_rfm


def test_quintile_score_ties_share_a_score():
    s = pd.Series([1, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9] * 10)
    scores = quintile_score(s)
    assert scores.min() == 1
    assert scores.max() <= 5
    assert (pd.DataFrame({"v": s, "sc": scores}).groupby("v")["sc"].nunique() == 1).all()


def test_quintile_score_is_monotonic():
    s = pd.Series(np.arange(100))
    scores = quintile_score(s)
    assert scores.is_monotonic_increasing
    assert scores.value_counts().tolist() == [20] * 5


def test_score_rfm_combined_string():
    df = pd.DataFrame({
        "tenure_months": np.arange(50),
        "num_services": np.arange(50) % 9 + 1,
        "total_charges": np.arange(50) * 10.0,
    })
    out = score_rfm(df)
    row = out.iloc[-1]
    expected = f"{row.rfm_recency_score}{row.rfm_frequency_score}{row.rfm_monetary_score}"
    assert row.rfm_combined == expected
    assert out["rfm_combined"].str.len().eq(3).all()


def test_name_clusters_uses_profile_not_label_number():
    # Cluster labels deliberately scrambled relative to their profiles
    profiles = {
        7: (1, 2, 1),   # lowest R            -> New & Uncommitted
        3: (4, 1, 3),   # lowest F of the rest -> Loyal Basics
        0: (5, 5, 5),   # highest R remaining -> Established Power Users
        5: (3, 4, 3),   # leftover            -> Growing Fiber Users
    }
    rows = [dict(zip(SCORE_COLUMNS, p), cluster_label=k) for k, p in profiles.items()]
    names = name_clusters(pd.DataFrame(rows))
    assert names == {
        7: "New & Uncommitted",
        3: "Loyal Basics",
        0: "Established Power Users",
        5: "Growing Fiber Users",
    }


def test_name_clusters_requires_four_clusters():
    df = pd.DataFrame([dict(zip(SCORE_COLUMNS, (1, 1, 1)), cluster_label=0)])
    with pytest.raises(ValueError):
        name_clusters(df)
