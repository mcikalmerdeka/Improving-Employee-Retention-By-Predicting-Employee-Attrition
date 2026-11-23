"""Model and dataset insights view."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import schema
from ..analytics import (
    attrition_by_group,
    coefficient_table,
    cumulative_gains_frame,
    gains_table,
    model_metrics,
    target_balance,
)
from ..ui import (
    attrition_chart,
    coefficient_chart,
    confusion_chart,
    distribution_chart,
    gains_chart,
    importance_chart,
    target_chart,
)


def _gains_table_frame(artifact: dict) -> pd.DataFrame:
    predictions = artifact["test_predictions"]
    return gains_table(predictions["y_true"], predictions["y_proba"])


def render_insights_tab(artifact: dict, training: pd.DataFrame) -> None:
    st.markdown(
        "How the model performs on the held-out split, what drives its "
        "scores, and what the training population looks like."
    )

    metrics = model_metrics(artifact)
    with st.container(horizontal=True):
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", border=True)
        st.metric(
            "PR-AUC",
            f"{metrics.get('average_precision', 0):.3f}",
            border=True,
        )
        st.metric(
            "Recall", f"{metrics.get('recall', 0):.1%}", border=True
        )
        st.metric(
            "Precision", f"{metrics.get('precision', 0):.1%}", border=True
        )
        st.metric("Test rows", f"{metrics.get('n_test', 0):,}", border=True)

    st.caption(
        "Held-out 30% stratified split, random_state=42. Accuracy is not "
        "shown as a headline because 69% accuracy is free on this data."
    )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Confusion matrix at 0.5**")
            st.altair_chart(
                confusion_chart(np.array(metrics["confusion_matrix"])),
                width="stretch",
            )
            st.caption("Rows are actual outcomes; columns are predictions.")
    with right:
        with st.container(border=True):
            st.markdown("**Cumulative gains**")
            predictions = artifact["test_predictions"]
            st.altair_chart(
                gains_chart(
                    cumulative_gains_frame(
                        predictions["y_true"], predictions["y_proba"]
                    )
                ),
                width="stretch",
            )
            st.caption(
                "If HR targets the highest-risk employees first, how many "
                "actual leavers are captured?"
            )

    with st.container(border=True):
        st.markdown("**Capacity table**")
        gains = _gains_table_frame(artifact).copy()
        gains["targeted_share"] = gains["targeted_share"] * 100
        gains["captured_share"] = gains["captured_share"] * 100
        st.dataframe(
            gains,
            hide_index=True,
            column_config={
                "targeted_share": st.column_config.NumberColumn(
                    "Targeted share", format="%.0f%%"
                ),
                "captured_share": st.column_config.NumberColumn(
                    "Leavers captured", format="%.1f%%"
                ),
                "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
            },
        )

    coefficients = coefficient_table(artifact)
    with st.container(border=True):
        if not coefficients.empty:
            st.markdown("**Global drivers (logistic coefficients)**")
            st.altair_chart(
                coefficient_chart(coefficients.head(15)), width="stretch"
            )
            st.caption(
                "Standardized coefficients. Positive values raise attrition "
                "odds; bar length is effect size."
            )
        else:
            st.markdown("**Global drivers (permutation importance)**")
            importance = artifact.get("permutation_importance")
            if importance is not None and not importance.empty:
                st.altair_chart(
                    importance_chart(importance.head(15)), width="stretch"
                )
            else:
                st.info("No global importance table is available.")

    st.subheader("Attrition patterns")
    left, right = st.columns(2, gap="large")
    with left:
        for column, label in [
            ("Pekerjaan", "job family"),
            ("PerformancePegawai", "performance rating"),
        ]:
            with st.container(border=True):
                st.markdown(f"**Attrition by {label}**")
                table = attrition_by_group(training, column)
                st.altair_chart(
                    attrition_chart(table, column), width="stretch"
                )
    with right:
        for column, label in [
            ("JenjangKarir", "career level"),
            ("SkorSurveyEngagement", "engagement score"),
        ]:
            with st.container(border=True):
                st.markdown(f"**Attrition by {label}**")
                table = attrition_by_group(training, column)
                st.altair_chart(
                    attrition_chart(table, column), width="stretch"
                )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Target balance**")
            st.altair_chart(
                target_chart(target_balance(training)), width="stretch"
            )
    with right:
        with st.container(border=True):
            st.markdown("**Age at hire by outcome**")
            st.altair_chart(
                distribution_chart(
                    training, "UsiaSaatMasuk", hue=schema.TARGET_COLUMN
                ),
                width="stretch",
            )

    st.caption(
        "The features separate leavers from stayers only weakly. Use the "
        "ranking as a screening aid and validate drivers with HR."
    )
