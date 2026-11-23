"""Batch scoring view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import schema
from ..analytics import batch_summary, risk_tier
from ..config import DATA_PATH
from ..pipeline import clean_raw, predict_frame

DISPLAY_COLUMNS = ["Source row", *schema.MODEL_FEATURES, "attrition_probability", "risk_tier"]


def _drop_unnamed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[
        :,
        [
            column
            for column in frame.columns
            if not str(column).startswith("Unnamed:")
        ],
    ]


def _missing_columns(frame: pd.DataFrame) -> list[str]:
    """Required raw inputs absent from an uploaded batch."""
    return [
        column
        for column in schema.RAW_INPUT_COLUMNS
        if column not in frame.columns
    ]


def _display_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Add a percent-scaled copy for the progress column."""
    display = results.copy()
    display["attrition_probability_pct"] = (
        display["attrition_probability"] * 100
    )
    return display


def _read_upload(uploaded) -> pd.DataFrame | None:
    try:
        return _drop_unnamed(pd.read_csv(uploaded)).reset_index(drop=True)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return None


def _warn_unknowns(clean: pd.DataFrame) -> None:
    for column, options in schema.CATEGORY_OPTIONS.items():
        if column not in clean.columns:
            continue
        unknown = sorted(set(clean[column].dropna()) - set(options))
        if unknown:
            st.warning(
                f"Unrecognized values in {column}: "
                + ", ".join(map(str, unknown))
                + ". The encoder treats them as unknown."
            )


def render_batch_tab(artifact: dict) -> None:
    st.markdown(
        "Upload an employee CSV and score every row. Results include the "
        "attrition probability, a relative risk tier, and a CSV download."
    )

    source = st.segmented_control(
        "Data source",
        ["Upload CSV", "Example data"],
        default="Upload CSV",
        key="batch_source",
    )

    raw = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Employee CSV",
            type="csv",
            help="Use the same columns as the training data. PII columns are accepted but dropped.",
        )
        if uploaded is not None:
            raw = _read_upload(uploaded)
    elif source == "Example data":
        try:
            raw = _drop_unnamed(pd.read_csv(DATA_PATH).head(8)).reset_index(
                drop=True
            )
            st.caption("Using the first eight rows of the training data.")
        except Exception as exc:
            st.error(f"Could not load the example data: {exc}")
            return

    if raw is None:
        with st.expander("Expected columns"):
            st.markdown(
                "\n".join(f"- `{column}`" for column in schema.RAW_INPUT_COLUMNS)
            )
            st.caption(
                "PII columns (Username, EnterpriseID, NomorHP, Email) are "
                "accepted but removed before scoring and display."
            )
        return

    if raw.empty:
        st.error("The uploaded file has no rows.")
        return

    missing = _missing_columns(raw)
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    raw = raw.drop(
        columns=[column for column in schema.PII_COLUMNS if column in raw.columns]
    )

    try:
        clean = clean_raw(raw)
    except Exception as exc:
        st.error(f"Could not prepare the batch: {exc}")
        return

    _warn_unknowns(clean)

    try:
        scored = predict_frame(artifact, raw)
        scored["risk_tier"] = scored["attrition_probability"].map(
            lambda probability: risk_tier(
                probability, artifact["risk_tier_cutoffs"]
            )
        )
    except Exception as exc:
        st.error(f"Could not score the batch: {exc}")
        return

    if scored.empty:
        st.error("No rows remain after cleaning and scoring.")
        return

    results = clean[schema.MODEL_FEATURES].copy()
    results.insert(0, "Source row", clean.index + 1)
    results["attrition_probability"] = scored["attrition_probability"].values
    results["risk_tier"] = scored["risk_tier"].values
    results = results.sort_values(
        "attrition_probability", ascending=False
    ).reset_index(drop=True)

    summary = batch_summary(results)
    with st.container(horizontal=True):
        st.metric("Rows scored", f"{summary['rows_scored']:,}", border=True)
        st.metric("High risk", f"{summary['high_risk']:,}", border=True)
        st.metric(
            "Average probability",
            f"{summary['average_probability']:.1%}",
            border=True,
        )

    display = _display_frame(results)
    display_columns = [
        "Source row",
        *schema.MODEL_FEATURES,
        "attrition_probability_pct",
        "risk_tier",
    ]
    st.dataframe(
        display[display_columns],
        hide_index=True,
        column_config={
            "Source row": st.column_config.NumberColumn("Row", format="%d"),
            "attrition_probability_pct": st.column_config.ProgressColumn(
                "Attrition probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
        },
    )

    st.download_button(
        "Download predictions as CSV",
        data=results[DISPLAY_COLUMNS].to_csv(index=False).encode("utf-8"),
        file_name="attrition_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    with st.expander("Prepared input preview"):
        st.dataframe(clean[schema.MODEL_FEATURES].head(20), hide_index=True)
