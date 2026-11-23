import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from employee_attrition import pipeline, schema

RAW = pd.read_csv("data/employee_performance_data.csv")


def make_raw(n=40):
    raw = RAW.head(n).copy()
    raw.loc[0, "StatusPernikahan"] = "-"
    raw.loc[1, "SkorKepuasanPegawai"] = np.nan
    raw.loc[2, "JumlahKetidakhadiran"] = np.nan
    raw.loc[3, "Pekerjaan"] = "Software Engineer (Back End)"
    return raw


def build_fake_artifact(estimator=None):
    raw = make_raw(len(RAW))
    clean = pipeline.clean_raw(raw)
    X = clean.drop(columns=[schema.TARGET_COLUMN])
    y = clean[schema.TARGET_COLUMN]
    values = pipeline.fit_imputation_values(X)
    X = pipeline.apply_imputation_values(X, values)
    encoded, encoders = pipeline.encode_features(X)
    feature_columns = list(encoded.columns)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "estimator",
                estimator or LogisticRegression(max_iter=1000, random_state=42),
            ),
        ]
    )
    model.fit(encoded, y)
    return {
        "artifact_version": schema.ARTIFACT_VERSION,
        "model": model,
        "encoders": encoders,
        "feature_columns": feature_columns,
        "imputation_values": values,
        "risk_tier_cutoffs": {"moderate": 0.3, "high": 0.6},
    }


def test_clean_raw_drops_pii_and_leakage():
    clean = pipeline.clean_raw(make_raw())
    forbidden = set(schema.PII_COLUMNS) | {
        "PernahBekerja",
        "IkutProgramLOP",
        "AlasanResign",
        "TanggalLahir",
        "TanggalHiring",
        "TanggalPenilaianKaryawan",
        "TanggalResign",
    }
    assert forbidden.isdisjoint(clean.columns)
    assert schema.TARGET_COLUMN in clean.columns


def test_clean_raw_normalizes_and_groups():
    clean = pipeline.clean_raw(make_raw())
    assert "Belum Menikah" in set(clean["StatusPernikahan"].dropna())
    assert "Software Engineer" in set(clean["Pekerjaan"])


def test_clean_raw_treats_dash_as_missing():
    clean = pipeline.clean_raw(make_raw())
    assert pd.isna(clean.loc[0, "StatusPernikahan"])


def test_clean_raw_derives_features():
    clean = pipeline.clean_raw(make_raw())
    assert clean["UsiaSaatMasuk"].between(15, 70).all()
    assert clean["TahunHiring"].between(2000, 2030).all()
    assert clean["BulanHiring"].between(1, 12).all()


def test_clean_raw_does_not_mutate_input():
    raw = make_raw()
    before = raw.copy(deep=True)
    pipeline.clean_raw(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_imputation_leaves_no_missing_values():
    clean = pipeline.clean_raw(make_raw())
    X = clean.drop(columns=[schema.TARGET_COLUMN])
    values = pipeline.fit_imputation_values(X)
    X = pipeline.apply_imputation_values(X, values)
    assert X[schema.NUMERIC_COLUMNS].isna().sum().sum() == 0
    assert X["StatusPernikahan"].isna().sum() == 0


def test_encode_features_fit_and_transform_share_columns():
    clean = pipeline.clean_raw(make_raw())
    X = clean.drop(columns=[schema.TARGET_COLUMN])
    X = pipeline.apply_imputation_values(X, pipeline.fit_imputation_values(X))
    train, encoders = pipeline.encode_features(X.iloc[:20])
    test, _ = pipeline.encode_features(X.iloc[20:], encoders=encoders)
    assert list(train.columns) == list(test.columns)


def test_align_features_fills_missing_columns():
    encoded = pd.DataFrame({"a": [1.0]})
    result = pipeline.align_features(encoded, ["a", "b"])
    assert list(result.columns) == ["a", "b"]
    assert result["b"].iloc[0] == 0


def test_predict_proba_returns_valid_probabilities():
    artifact = build_fake_artifact()
    proba = pipeline.predict_proba(artifact, make_raw())
    assert proba.shape == (len(make_raw()),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_predict_frame_adds_probability():
    artifact = build_fake_artifact()
    scored = pipeline.predict_frame(artifact, make_raw())
    assert "attrition_probability" in scored.columns
    assert scored["attrition_probability"].between(0, 1).all()


def test_explain_row_decomposes_log_odds():
    artifact = build_fake_artifact()
    encoded = pipeline.transform_raw(artifact, make_raw().head(1))
    contributions = pipeline.explain_row(artifact, encoded, top_n=None)
    probability = artifact["model"].predict_proba(encoded)[0, 1]
    intercept = artifact["model"].steps[-1][1].intercept_[0]
    logit = np.log(probability / (1 - probability))
    assert np.isclose(
        contributions["contribution"].sum() + intercept, logit, atol=1e-6
    )


def test_explain_row_returns_empty_for_non_linear():
    artifact = build_fake_artifact(
        DecisionTreeClassifier(max_depth=2, random_state=42)
    )
    encoded = pipeline.transform_raw(artifact, make_raw().head(1))
    assert pipeline.explain_row(artifact, encoded).empty


def test_load_artifact_rejects_unknown_version(tmp_path):
    path = tmp_path / "bad.joblib"
    joblib.dump({"artifact_version": 999}, path)
    with pytest.raises(ValueError, match="artifact version"):
        pipeline.load_artifact(path)
