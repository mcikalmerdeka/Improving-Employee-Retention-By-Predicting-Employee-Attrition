"""Pure analytics helpers for the attrition app.

No Streamlit imports: the notebook and the app share these functions, and the
tests exercise them directly.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import schema


def risk_tier(probability: float, cutoffs: dict) -> str:
    """Map an attrition probability to ``High``, ``Moderate``, or ``Low``."""
    if probability >= cutoffs["high"]:
        return "High"
    if probability >= cutoffs["moderate"]:
        return "Moderate"
    return "Low"


def risk_percentile(
    probability: float, sorted_train_probabilities: Iterable[float]
) -> float:
    """Share of training employees scored at or below this probability."""
    values = np.sort(np.asarray(list(sorted_train_probabilities), dtype=float))
    if values.size == 0:
        return 0.0
    position = np.searchsorted(values, probability, side="right")
    return float(position / values.size * 100.0)


def model_metrics(artifact: dict) -> dict:
    """Copy the saved test metrics and expose the test-set size."""
    metrics = dict(artifact.get("metrics", {}))
    metrics["n_test"] = artifact.get("training_summary", {}).get("n_test")
    return metrics


def coefficient_table(artifact: dict) -> pd.DataFrame:
    """Return coefficients sorted by absolute effect, or an empty frame."""
    coefficients = artifact.get("coefficients")
    if coefficients is None or coefficients.empty:
        return pd.DataFrame(
            columns=["feature", "coefficient", "abs_coefficient"]
        )
    return (
        coefficients.sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def feature_display_name(feature: str) -> str:
    """Turn a model column into a label that reads well in charts."""
    for column in schema.NOMINAL_COLUMNS:
        prefix = f"{column}_"
        if feature.startswith(prefix):
            label = schema.FEATURE_LABELS.get(column, column)
            return f"{label}: {feature[len(prefix):]}"
    return schema.FEATURE_LABELS.get(feature, feature)


def target_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Employee counts per target class."""
    counts = (
        frame[schema.TARGET_COLUMN]
        .map(schema.DISPLAY_TARGET_LABELS)
        .value_counts()
    )
    return counts.rename_axis("outcome").reset_index(name="employees")


def attrition_by_group(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Attrition rate for every value of a categorical column."""
    grouped = (
        frame.groupby(column, dropna=False)[schema.TARGET_COLUMN]
        .agg(employees="count", leavers="sum")
        .reset_index()
    )
    grouped["attrition_rate"] = grouped["leavers"] / grouped["employees"]
    return grouped.sort_values("attrition_rate", ascending=False).reset_index(
        drop=True
    )


def cumulative_gains_frame(
    y_true: Iterable[int], y_proba: Iterable[float]
) -> pd.DataFrame:
    """Cumulative share of leavers captured as employees are targeted."""
    truth = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_proba), dtype=float)
    order = np.argsort(-scores, kind="stable")
    truth = truth[order]

    total_leavers = int(truth.sum())
    captured = np.cumsum(truth)
    share_captured = (
        captured / total_leavers
        if total_leavers
        else np.zeros(len(truth), dtype=float)
    )
    share_targeted = np.arange(1, len(truth) + 1) / len(truth)

    return pd.DataFrame(
        {
            "share_targeted": share_targeted,
            "share_captured": share_captured,
            "random_baseline": share_targeted,
        }
    )


def gains_table(
    y_true: Iterable[int],
    y_proba: Iterable[float],
    points: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5),
) -> pd.DataFrame:
    """Leavers captured and lift at fixed targeting capacities."""
    truth = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(y_proba), dtype=float)
    order = np.argsort(-scores, kind="stable")
    truth = truth[order]

    total_leavers = int(truth.sum())
    total_employees = len(truth)
    rows = []

    for point in points:
        targeted = int(np.ceil(point * total_employees))
        captured = int(truth[:targeted].sum())
        captured_share = (
            captured / total_leavers if total_leavers else 0.0
        )
        rows.append(
            {
                "targeted_share": point,
                "employees_targeted": targeted,
                "leavers_captured": captured,
                "captured_share": captured_share,
                "lift": captured_share / point if point else 0.0,
            }
        )

    return pd.DataFrame(rows)


def batch_summary(scored: pd.DataFrame) -> dict:
    """Headline numbers for a scored batch."""
    return {
        "rows_scored": int(len(scored)),
        "high_risk": int((scored["risk_tier"] == "High").sum()),
        "moderate_risk": int((scored["risk_tier"] == "Moderate").sum()),
        "average_probability": float(
            scored["attrition_probability"].mean()
        ),
    }
