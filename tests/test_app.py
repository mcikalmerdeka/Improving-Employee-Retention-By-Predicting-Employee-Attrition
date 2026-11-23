from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
