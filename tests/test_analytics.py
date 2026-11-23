import pandas as pd
import pytest

from employee_attrition import analytics


def test_risk_tier_boundaries():
    cutoffs = {"moderate": 0.3, "high": 0.6}
    assert analytics.risk_tier(0.29, cutoffs) == "Low"
    assert analytics.risk_tier(0.30, cutoffs) == "Moderate"
    assert analytics.risk_tier(0.59, cutoffs) == "Moderate"
    assert analytics.risk_tier(0.60, cutoffs) == "High"


def test_risk_percentile():
    probs = [0.1, 0.2, 0.3, 0.4]
    assert analytics.risk_percentile(0.05, probs) == 0.0
    assert analytics.risk_percentile(0.25, probs) == 50.0
    assert analytics.risk_percentile(0.50, probs) == 100.0


def test_gains_table():
    result = analytics.gains_table(
        [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], points=(0.5,)
    )
    row = result.iloc[0]
    assert row["employees_targeted"] == 2
    assert row["leavers_captured"] == 2
    assert row["captured_share"] == 1.0
    assert row["lift"] == 2.0


def test_cumulative_gains_frame():
    frame = analytics.cumulative_gains_frame(
        [1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]
    )
    assert frame.iloc[0]["share_targeted"] == 0.25
    assert frame.iloc[0]["share_captured"] == 0.5
    assert frame.iloc[-1]["share_captured"] == 1.0
    assert "random_baseline" in frame.columns


def test_attrition_by_group():
    frame = pd.DataFrame(
        {
            "Pekerjaan": ["A", "A", "B", "B"],
            "StatusKerja": [1, 0, 0, 0],
        }
    )
    result = analytics.attrition_by_group(frame, "Pekerjaan").set_index(
        "Pekerjaan"
    )
    assert result.loc["A", "employees"] == 2
    assert result.loc["A", "leavers"] == 1
    assert result.loc["A", "attrition_rate"] == 0.5
    assert result.loc["B", "attrition_rate"] == 0.0


def test_target_balance():
    frame = pd.DataFrame({"StatusKerja": [1, 1, 0, 0, 0]})
    result = analytics.target_balance(frame).set_index("outcome")
    assert result.loc["Resigned", "employees"] == 2
    assert result.loc["Still working", "employees"] == 3


def test_model_metrics_adds_test_size():
    artifact = {
        "metrics": {"roc_auc": 0.61},
        "training_summary": {"n_test": 87},
    }
    result = analytics.model_metrics(artifact)
    assert result["roc_auc"] == 0.61
    assert result["n_test"] == 87


def test_coefficient_table_sorts_by_absolute_value():
    artifact = {
        "coefficients": pd.DataFrame(
            {
                "feature": ["a", "b"],
                "coefficient": [0.1, -0.5],
                "abs_coefficient": [0.1, 0.5],
            }
        )
    }
    result = analytics.coefficient_table(artifact)
    assert result.iloc[0]["feature"] == "b"


def test_feature_display_name_formats_one_hot():
    assert (
        analytics.feature_display_name("Pekerjaan_Data & ML")
        == "Job family: Data & ML"
    )
    assert analytics.feature_display_name("UsiaSaatMasuk") == "Age at hire"


def test_batch_summary():
    frame = pd.DataFrame(
        {
            "attrition_probability": [0.1, 0.5, 0.8],
            "risk_tier": ["Low", "Moderate", "High"],
        }
    )
    summary = analytics.batch_summary(frame)
    assert summary["rows_scored"] == 3
    assert summary["high_risk"] == 1
    assert summary["moderate_risk"] == 1
    assert summary["average_probability"] == pytest.approx(
        0.4666667, rel=1e-3
    )
