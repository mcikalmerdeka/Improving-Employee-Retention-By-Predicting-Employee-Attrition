"""Individual employee attrition-risk view."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from .. import schema
from ..analytics import feature_display_name, risk_percentile, risk_tier
from ..pipeline import explain_row, predict_frame, transform_raw
from ..ui import (
    contribution_chart,
    fmt_percent,
    importance_chart,
    probability_progress,
)

RESULT_KEY = "predict_result"
KEYS = {
    "StatusPernikahan": "predict_status_pernikahan",
    "JenisKelamin": "predict_jenis_kelamin",
    "StatusKepegawaian": "predict_status_kepegawaian",
    "Pekerjaan": "predict_pekerjaan",
    "JenjangKarir": "predict_jenjang_karir",
    "PerformancePegawai": "predict_performance",
    "AsalDaerah": "predict_asal_daerah",
    "HiringPlatform": "predict_hiring_platform",
    "SkorSurveyEngagement": "predict_engagement",
    "SkorKepuasanPegawai": "predict_satisfaction",
    "JumlahKeikutsertaanProjek": "predict_projects",
    "JumlahKeterlambatanSebulanTerakhir": "predict_late",
    "JumlahKetidakhadiran": "predict_absence",
    "TingkatPendidikan": "predict_education",
    "TanggalLahir": "predict_birth_date",
    "TanggalHiring": "predict_hire_date",
}

RECOMMENDATIONS = [
    ("Engagement score", "Check in on workload, recognition, and team fit."),
    ("Satisfaction score", "Run a stay interview focused on day-to-day satisfaction."),
    ("Absences", "Ask whether health, burnout, or scheduling is driving absence."),
    ("Late arrivals", "Review schedule flexibility and commute constraints."),
    ("Projects joined", "Discuss growth opportunities and project fit."),
    ("Career level", "Review the promotion path and the next role."),
    ("Performance rating", "Check whether the last rating conversation changed plans."),
    ("Employment type", "Review contract stability and the conversion path."),
    ("Job family", "Compare this role's attrition with the team baseline."),
    ("Hiring source", "Review onboarding quality for this hiring channel."),
]


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    return date(1990, 1, 1)


def _load_example(example: dict) -> None:
    for column, key in KEYS.items():
        if column in ("TanggalLahir", "TanggalHiring"):
            st.session_state[key] = _as_date(example.get(column))
        else:
            st.session_state[key] = example.get(column)
    st.session_state.pop(RESULT_KEY, None)


def _select_index(options: list, value) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def _recommendation(explanations: pd.DataFrame) -> str:
    if explanations.empty:
        return (
            "Review the model's global drivers above and use the score to "
            "prioritise a conversation, not to make the decision alone."
        )
    positive = explanations[explanations["contribution"] > 0]
    if positive.empty:
        return "No single feature raises the risk; review the full profile."
    label = feature_display_name(positive.iloc[0]["feature"])
    for prefix, message in RECOMMENDATIONS:
        if label.startswith(prefix):
            return message
    return (
        f"The largest risk-raising factor is {label}. Validate it in a "
        "conversation before acting."
    )


def _render_result(artifact: dict, result: dict) -> None:
    st.markdown("### Prediction")

    with st.container(horizontal=True):
        st.metric("Risk tier", result["tier"], border=True)
        st.metric(
            "Attrition probability",
            fmt_percent(result["probability"]),
            border=True,
        )
        st.metric(
            "Risk percentile",
            f"{result['percentile']:.0f}th",
            border=True,
        )

    probability_progress(result["probability"], result["tier"])
    st.caption(
        "This profile scores higher than "
        f"{result['percentile']:.0f}% of the training population. "
        "The percentile is relative, not a calibrated chance of leaving."
    )

    explanations = result["explanations"]
    with st.container(border=True):
        st.markdown("**Why this prediction**")
        if not explanations.empty:
            st.altair_chart(contribution_chart(explanations), width="stretch")
            st.caption(
                "Each bar is a coefficient multiplied by the standardized "
                "model input. Positive values push toward resignation; "
                "negative values push away."
            )
        else:
            st.info(
                "The deployed model is non-linear, so per-employee "
                "contributions are not available. The chart below shows the "
                "model's global permutation importance instead."
            )
            importance = artifact.get("permutation_importance")
            if importance is not None and not importance.empty:
                st.altair_chart(
                    importance_chart(importance.head(12)), width="stretch"
                )

    st.markdown("**Suggested next step**")
    st.info(_recommendation(explanations))

    with st.expander("Model input (encoded features)"):
        encoded = result["encoded"].T.reset_index()
        encoded.columns = ["feature", "value"]
        encoded["feature"] = encoded["feature"].map(feature_display_name)
        st.dataframe(encoded, hide_index=True)


def render_predict_tab(artifact: dict) -> None:
    st.markdown(
        "Score one employee from their profile, engagement, and attendance "
        "data. Only the 16 fields the model uses are requested."
    )

    defaults = artifact.get("default_values", {})
    examples = artifact.get("examples", {})

    with st.container(horizontal=True):
        if "high_risk" in examples:
            st.button(
                "Load high-risk example",
                icon=":material/trending_up:",
                type="tertiary",
                on_click=_load_example,
                args=(examples["high_risk"],),
            )
        if "low_risk" in examples:
            st.button(
                "Load low-risk example",
                icon=":material/trending_down:",
                type="tertiary",
                on_click=_load_example,
                args=(examples["low_risk"],),
            )

    options = schema.CATEGORY_OPTIONS
    birth_default = _as_date(defaults.get("TanggalLahir"))
    hire_default = _as_date(defaults.get("TanggalHiring"))

    with st.form("predict_form", border=True):
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**Demographics**")
            birth_date = st.date_input(
                "Date of birth",
                value=birth_default,
                min_value=date(1940, 1, 1),
                max_value=date.today(),
                key=KEYS["TanggalLahir"],
            )
            gender = st.selectbox(
                "Gender",
                options["JenisKelamin"],
                index=_select_index(
                    options["JenisKelamin"], defaults.get("JenisKelamin")
                ),
                key=KEYS["JenisKelamin"],
            )
            marital = st.selectbox(
                "Marital status",
                options["StatusPernikahan"],
                index=_select_index(
                    options["StatusPernikahan"], defaults.get("StatusPernikahan")
                ),
                key=KEYS["StatusPernikahan"],
            )
            education = st.selectbox(
                "Education",
                options["TingkatPendidikan"],
                index=_select_index(
                    options["TingkatPendidikan"],
                    defaults.get("TingkatPendidikan"),
                ),
                key=KEYS["TingkatPendidikan"],
            )
            region = st.selectbox(
                "Home region",
                options["AsalDaerah"],
                index=_select_index(
                    options["AsalDaerah"], defaults.get("AsalDaerah")
                ),
                key=KEYS["AsalDaerah"],
            )

            st.markdown("**Employment**")
            hire_date = st.date_input(
                "Hire date",
                value=hire_default,
                min_value=date(1990, 1, 1),
                max_value=date.today(),
                key=KEYS["TanggalHiring"],
            )
            employment = st.selectbox(
                "Employment type",
                options["StatusKepegawaian"],
                index=_select_index(
                    options["StatusKepegawaian"],
                    defaults.get("StatusKepegawaian"),
                ),
                key=KEYS["StatusKepegawaian"],
            )
            job_family = st.selectbox(
                "Job family",
                options["Pekerjaan"],
                index=_select_index(
                    options["Pekerjaan"], defaults.get("Pekerjaan")
                ),
                key=KEYS["Pekerjaan"],
            )
            career = st.selectbox(
                "Career level",
                options["JenjangKarir"],
                index=_select_index(
                    options["JenjangKarir"], defaults.get("JenjangKarir")
                ),
                key=KEYS["JenjangKarir"],
            )
            source = st.selectbox(
                "Hiring source",
                options["HiringPlatform"],
                index=_select_index(
                    options["HiringPlatform"],
                    defaults.get("HiringPlatform"),
                ),
                key=KEYS["HiringPlatform"],
            )

        with right:
            st.markdown("**Performance and engagement**")
            performance = st.selectbox(
                "Performance rating",
                options["PerformancePegawai"],
                index=_select_index(
                    options["PerformancePegawai"],
                    defaults.get("PerformancePegawai"),
                ),
                key=KEYS["PerformancePegawai"],
            )
            engagement = st.slider(
                "Engagement score",
                min_value=1,
                max_value=5,
                value=int(round(float(defaults.get("SkorSurveyEngagement", 3)))),
                key=KEYS["SkorSurveyEngagement"],
            )
            satisfaction = st.slider(
                "Satisfaction score",
                min_value=1,
                max_value=5,
                value=int(round(float(defaults.get("SkorKepuasanPegawai", 4)))),
                key=KEYS["SkorKepuasanPegawai"],
            )
            projects = st.number_input(
                "Projects joined",
                min_value=0,
                max_value=30,
                value=int(round(float(defaults.get("JumlahKeikutsertaanProjek", 0)))),
                step=1,
                key=KEYS["JumlahKeikutsertaanProjek"],
            )
            late = st.number_input(
                "Late arrivals (last month)",
                min_value=0,
                max_value=30,
                value=int(
                    round(
                        float(
                            defaults.get(
                                "JumlahKeterlambatanSebulanTerakhir", 0
                            )
                        )
                    )
                ),
                step=1,
                key=KEYS["JumlahKeterlambatanSebulanTerakhir"],
            )
            absence = st.number_input(
                "Absences",
                min_value=0,
                max_value=60,
                value=int(round(float(defaults.get("JumlahKetidakhadiran", 10)))),
                step=1,
                key=KEYS["JumlahKetidakhadiran"],
            )

        submitted = st.form_submit_button(
            "Predict attrition risk",
            type="primary",
            icon=":material/bolt:",
        )

    if submitted:
        raw = pd.DataFrame(
            [
                {
                    "StatusPernikahan": marital,
                    "JenisKelamin": gender,
                    "StatusKepegawaian": employment,
                    "Pekerjaan": job_family,
                    "JenjangKarir": career,
                    "PerformancePegawai": performance,
                    "AsalDaerah": region,
                    "HiringPlatform": source,
                    "SkorSurveyEngagement": float(engagement),
                    "SkorKepuasanPegawai": float(satisfaction),
                    "JumlahKeikutsertaanProjek": float(projects),
                    "JumlahKeterlambatanSebulanTerakhir": float(late),
                    "JumlahKetidakhadiran": float(absence),
                    "TingkatPendidikan": education,
                    "TanggalLahir": birth_date,
                    "TanggalHiring": hire_date,
                }
            ]
        )
        try:
            scored = predict_frame(artifact, raw)
            probability = float(scored["attrition_probability"].iloc[0])
            encoded = transform_raw(artifact, raw)
            explanations = explain_row(artifact, encoded)
            st.session_state[RESULT_KEY] = {
                "probability": probability,
                "tier": risk_tier(
                    probability, artifact["risk_tier_cutoffs"]
                ),
                "percentile": risk_percentile(
                    probability, artifact["train_probabilities_sorted"]
                ),
                "encoded": encoded,
                "explanations": explanations,
            }
        except Exception as exc:  # surface the real reason in the UI
            st.error(f"Could not score this employee: {exc}")

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.caption("Submit the form to score an employee.")
        return

    _render_result(artifact, result)
