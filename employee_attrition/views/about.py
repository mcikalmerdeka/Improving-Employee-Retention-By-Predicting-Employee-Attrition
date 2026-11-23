"""About and model card view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import schema


def _data_dictionary() -> pd.DataFrame:
    frame = (
        pd.DataFrame(schema.FEATURE_DEFINITIONS)
        .T.rename_axis("Feature")
        .reset_index()
    )
    frame["Type"] = frame["data_type"] + " — " + frame["specific_type"]
    return frame.rename(
        columns={
            "description": "Description",
            "used_by_model": "Used by model",
            "reason": "Reason",
        }
    )[["Feature", "Description", "Type", "Used by model", "Reason"]]


def render_about_tab(artifact: dict, training: pd.DataFrame) -> None:
    st.markdown(
        "An Indonesian technology company loses about 31% of its employees "
        "to resignation. This app ranks employees by risk so HR can focus "
        "retention conversations, and shows the drivers behind each score."
    )

    st.markdown("**Model card**")
    with st.container(border=True):
        summary = artifact["training_summary"]
        st.markdown(
            f"- **Algorithm** — {artifact['model_name']} (binary classification)\n"
            f"- **Target** — `{schema.TARGET_COLUMN}`: 1 = resigned, 0 = still working\n"
            f"- **Training rows** — {summary['n_rows']:,} "
            f"({summary['n_train']:,} train / {summary['n_test']:,} test)\n"
            f"- **Positive rate** — {summary['positive_rate']:.1%}\n"
            f"- **Encoded features** — {summary['n_features']}\n"
            f"- **Evaluation** — held-out 30% stratified split, random_state=42\n"
            f"- **Artifact** — `models/attrition_model.joblib`"
        )

    metrics = artifact.get("metrics", {})
    with st.container(horizontal=True):
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", border=True)
        st.metric("PR-AUC", f"{metrics.get('average_precision', 0):.3f}", border=True)
        st.metric("Recall", f"{metrics.get('recall', 0):.1%}", border=True)
        st.metric("Precision", f"{metrics.get('precision', 0):.1%}", border=True)

    st.subheader("Data dictionary")
    st.caption(
        "All 25 source columns plus the derived target. PII and leaky "
        "columns are documented even though the model never sees them."
    )
    st.dataframe(
        _data_dictionary(),
        hide_index=True,
        column_config={
            "Description": st.column_config.TextColumn(width="large"),
            "Reason": st.column_config.TextColumn(width="large"),
        },
    )

    with st.expander("Training data preview (model features only)"):
        st.dataframe(
            training[schema.MODEL_FEATURES].head(10), hide_index=True
        )

    st.subheader("Limitations")
    st.markdown(
        "- **Small sample** — 287 rows, 89 leavers, 87 test rows. Estimates are noisy.\n"
        "- **Weak signal** — these features separate leavers from stayers only weakly; use the score for ranking, not for confident individual verdicts.\n"
        "- **Date inconsistency** — 64 of 89 leavers have an assessment date after resignation, so assessment-date features are excluded.\n"
        "- **Correlation, not causation** — a driver is not necessarily a lever.\n"
        "- **Snapshot, not a forecast** — the model predicts a static outcome, not a six-month resignation probability.\n"
        "- **No PII** — usernames, employee IDs, phone numbers, and emails are excluded from the model, artifact, and app."
    )

    st.subheader("Author")
    st.markdown(
        "Developed by **Muhammad Cikal Merdeka** — Data Analyst / Data Scientist\n\n"
        "[Email](mailto:mcikalmerdeka@gmail.com) — "
        "[GitHub](https://github.com/mcikalmerdeka) — "
        "[LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)"
    )

    st.caption(
        "Light and dark themes are available from the app menu "
        "(⋮ → Settings → Theme)."
    )
