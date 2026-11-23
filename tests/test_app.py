from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from employee_attrition.views import batch as batch_view

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = PROJECT_ROOT / "main.py"
MODEL_PATH = PROJECT_ROOT / "models" / "attrition_model.joblib"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Run notebook.ipynb to create the model artifact.",
)


def test_app_runs_without_exception():
    at = AppTest.from_file(MAIN_PATH).run(timeout=60)
    assert not at.exception
    assert any("Retention" in title.value for title in at.title)


def test_app_has_all_four_views():
    at = AppTest.from_file(MAIN_PATH).run(timeout=60)
    assert not at.exception
    assert len(at.tabs) == 4

    text = "\n".join(element.value for element in at.markdown)
    assert "Score one employee" in text
    assert "Upload an employee CSV" in text
    assert "How the model performs" in text
    assert "**Model card**" in text
    assert "engagement score" in text


def test_predict_form_submits_and_renders_result():
    at = AppTest.from_file(MAIN_PATH).run(timeout=60)
    buttons = list(at.button) + list(at.form_submit_button)
    submit = next(
        button for button in buttons if button.label == "Predict attrition risk"
    )
    submit.click().run(timeout=60)

    assert not at.exception
    assert not at.error
    labels = {metric.label for metric in at.metric}
    assert {"Risk tier", "Attrition probability", "Risk percentile"} <= labels
    assert any("non-linear" in info.value for info in at.info)


def test_batch_display_frame_scales_probability():
    frame = pd.DataFrame(
        {
            "attrition_probability": [0.6],
            "risk_tier": ["Moderate"],
        }
    )
    display = batch_view._display_frame(frame)
    assert display["attrition_probability_pct"].iloc[0] == 60.0


def test_batch_missing_columns_helper():
    frame = pd.DataFrame({"StatusPernikahan": ["Menikah"]})
    missing = batch_view._missing_columns(frame)
    assert "TanggalLahir" in missing
    assert len(missing) == 15
