"""Filesystem paths used by the attrition app."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "employee_performance_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "attrition_model.joblib"
