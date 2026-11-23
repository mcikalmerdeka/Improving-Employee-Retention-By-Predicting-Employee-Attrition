"""Employee Retention Analyzer - Streamlit entry point."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from employee_attrition.config import DATA_PATH, MODEL_PATH
from employee_attrition.pipeline import clean_raw, load_artifact
from employee_attrition.views import (
    render_about_tab,
    render_batch_tab,
    render_insights_tab,
    render_predict_tab,
)

st.set_page_config(
    page_title="Employee Retention Analyzer",
    page_icon=":material/groups:",
    layout="wide",
)

if not MODEL_PATH.exists():
    st.error(
        f"Model artifact not found at {MODEL_PATH}. "
        "Run notebook.ipynb first to generate models/attrition_model.joblib."
    )
    st.stop()

try:
    artifact = load_artifact(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model artifact: {exc}")
    st.stop()

try:
    raw = pd.read_csv(DATA_PATH)
    training = clean_raw(raw)
except Exception as exc:
    st.error(f"Could not load the training data: {exc}")
    st.stop()

st.title("Employee Retention Analyzer")
st.caption(
    "Estimate resignation risk, prioritise retention outreach, and see what "
    "drives the score."
)

with st.container(horizontal=True):
    st.badge(
        artifact["model_name"],
        icon=":material/model_training:",
        color="blue",
    )
    st.badge(
        f"{len(artifact['feature_columns'])} encoded features",
        icon=":material/tune:",
        color="gray",
    )
    st.badge(
        f"{artifact['training_summary']['n_rows']:,} employees",
        icon=":material/database:",
        color="gray",
    )
    st.badge(
        f"{artifact['metrics']['roc_auc']:.3f} ROC-AUC",
        icon=":material/analytics:",
        color="gray",
    )

tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["Predict", "Batch", "Insights", "About"]
)

with tab_predict:
    render_predict_tab(artifact)

with tab_batch:
    render_batch_tab(artifact)

with tab_insights:
    render_insights_tab(artifact, training)

with tab_about:
    render_about_tab(artifact, training)
