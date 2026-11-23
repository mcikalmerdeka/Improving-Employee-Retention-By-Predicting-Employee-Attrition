"""Shared presentation helpers for the attrition app."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .analytics import feature_display_name

ACCENT = "#4F46E5"
AMBER = "#D97706"
POSITIVE = "#B91C1C"
NEGATIVE = "#15803D"
INK = "#1C1917"

RISK_COLORS = {
    "High": "#B91C1C",
    "Moderate": "#D97706",
    "Low": "#15803D",
}


def fmt_percent(value: float) -> str:
    """Format a probability without rounding extremes to 0% or 100%."""
    if value >= 0.9995:
        return ">99.9%"
    if value <= 0.0005:
        return "<0.1%"
    return f"{value:.1%}"


def risk_color(tier: str) -> str:
    return RISK_COLORS.get(tier, INK)


def probability_progress(probability: float, tier: str) -> None:
    st.progress(
        min(max(float(probability), 0.0), 1.0),
        text=f"Attrition probability — {fmt_percent(probability)} "
        f"({tier} risk)",
    )


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "feature" in display.columns:
        display["display"] = display["feature"].map(feature_display_name)
    return display


def contribution_chart(contributions: pd.DataFrame) -> alt.Chart:
    """Horizontal bars showing one prediction's top drivers."""
    display = _display_frame(contributions)
    order = display["display"].tolist()
    bars = (
        alt.Chart(display)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("contribution:Q", title="Effect on attrition log-odds"),
            y=alt.Y("display:N", title=None, sort=order),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=[
                        "Increases attrition odds",
                        "Decreases attrition odds",
                    ],
                    range=[POSITIVE, NEGATIVE],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("display:N", title="Feature"),
                alt.Tooltip("input:Q", title="Standardized input", format=".3f"),
                alt.Tooltip("contribution:Q", title="Contribution", format="+.3f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"zero": [0]}))
        .mark_rule(color="#A8A29E", strokeWidth=1)
        .encode(x="zero:Q")
    )
    return (bars + rule).properties(height=alt.Step(30))


def coefficient_chart(table: pd.DataFrame) -> alt.Chart:
    """Global standardized coefficients for the model."""
    display = _display_frame(table)
    order = display["display"].tolist()
    return (
        alt.Chart(display)
        .mark_bar(cornerRadius=3)
        .encode(
            x=alt.X("coefficient:Q", title="Coefficient (log-odds per SD)"),
            y=alt.Y("display:N", title=None, sort=order),
            color=alt.condition(
                "datum.coefficient > 0", alt.value(POSITIVE), alt.value(NEGATIVE)
            ),
            tooltip=[
                alt.Tooltip("display:N", title="Feature"),
                alt.Tooltip("coefficient:Q", title="Coefficient", format="+.3f"),
            ],
        )
        .properties(height=alt.Step(26))
    )


def importance_chart(table: pd.DataFrame) -> alt.Chart:
    """Global permutation importance for non-linear models."""
    display = _display_frame(table)
    order = display["display"].tolist()
    return (
        alt.Chart(display)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("importance_mean:Q", title="Permutation importance (ROC-AUC)"),
            y=alt.Y("display:N", title=None, sort=order),
            tooltip=[
                alt.Tooltip("display:N", title="Feature"),
                alt.Tooltip(
                    "importance_mean:Q", title="Importance", format="+.3f"
                ),
            ],
        )
        .properties(height=alt.Step(26))
    )


def confusion_chart(matrix: np.ndarray) -> alt.Chart:
    """2x2 confusion matrix with employee counts."""
    rows = ["Actual — still working", "Actual — resigned"]
    columns = ["Predicted — still working", "Predicted — resigned"]
    frame = (
        pd.DataFrame(matrix, index=rows, columns=columns)
        .rename_axis("actual")
        .reset_index()
        .melt(id_vars="actual", var_name="predicted", value_name="employees")
    )

    base = alt.Chart(frame).mark_rect(cornerRadius=4).encode(
        x=alt.X("predicted:N", title=None, sort=columns),
        y=alt.Y("actual:N", title=None, sort=rows),
        color=alt.Color(
            "employees:Q",
            scale=alt.Scale(range=["#EEF2FF", "#A5B4FC", "#4F46E5"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("actual:N", title="Actual"),
            alt.Tooltip("predicted:N", title="Predicted"),
            alt.Tooltip("employees:Q", title="Employees"),
        ],
    )
    text = base.mark_text(fontSize=20, fontWeight="bold", color=INK).encode(
        text="employees:Q"
    )
    return (base + text).properties(height=220)


def gains_chart(frame: pd.DataFrame) -> alt.Chart:
    """Cumulative gains against random targeting."""
    long = frame.melt(
        id_vars="share_targeted",
        value_vars=["share_captured", "random_baseline"],
        var_name="series",
        value_name="share",
    )
    labels = {"share_captured": "Model", "random_baseline": "Random"}
    long["series"] = long["series"].map(labels)
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("share_targeted:Q", title="Share of employees targeted"),
            y=alt.Y("share:Q", title="Share of leavers captured"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Model", "Random"], range=[ACCENT, "#A8A29E"]
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("share_targeted:Q", format=".0%"),
                alt.Tooltip("series:N"),
                alt.Tooltip("share:Q", format=".0%"),
            ],
        )
        .properties(height=320)
    )


def attrition_chart(frame: pd.DataFrame, column: str) -> alt.Chart:
    """Attrition rate by a categorical column."""
    display = frame.rename(columns={column: "group"}).copy()
    return (
        alt.Chart(display)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("attrition_rate:Q", title="Attrition rate", axis=alt.Axis(format="%")),
            y=alt.Y("group:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("group:N", title=column),
                alt.Tooltip("employees:Q", title="Employees"),
                alt.Tooltip("leavers:Q", title="Leavers"),
                alt.Tooltip("attrition_rate:Q", title="Attrition", format=".1%"),
            ],
        )
        .properties(height=alt.Step(28))
    )


def target_chart(balance: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(balance)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("outcome:N", title=None),
            y=alt.Y("employees:Q", title="Employees"),
            tooltip=["outcome:N", "employees:Q"],
        )
        .properties(height=240)
    )


def distribution_chart(
    frame: pd.DataFrame, column: str, hue: str | None = None
) -> alt.Chart:
    """Histogram of one continuous feature, optionally split by outcome."""
    display = frame.copy()
    encoding = {
        "x": alt.X(f"{column}:Q", bin=alt.Bin(maxbins=28), title=column),
        "y": alt.Y("count():Q", title="Employees"),
        "tooltip": [alt.Tooltip("count():Q", title="Employees")],
    }
    if hue and hue in display.columns:
        display[hue] = display[hue].map({0: "Still working", 1: "Resigned"})
        encoding["color"] = alt.Color(
            f"{hue}:N",
            scale=alt.Scale(
                domain=["Still working", "Resigned"], range=[ACCENT, AMBER]
            ),
            title=None,
        )
    return (
        alt.Chart(display)
        .mark_bar(opacity=0.8, cornerRadius=2)
        .encode(**encoding)
        .properties(height=260)
    )
