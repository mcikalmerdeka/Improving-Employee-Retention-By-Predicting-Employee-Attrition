"""Shared cleaning, encoding, prediction, and explanation for attrition models.

The notebook fits the encoders/imputer on the training split and the Streamlit
app reuses the same functions with the persisted artifact, so training and
serving cannot drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils import feature_encoding

from . import schema

_IMPUTATION_NUMERIC = [
    "SkorKepuasanPegawai",
    "JumlahKeikutsertaanProjek",
    "JumlahKeterlambatanSebulanTerakhir",
    "JumlahKetidakhadiran",
    "UsiaSaatMasuk",
    "TahunHiring",
    "BulanHiring",
]

_ARTIFACT_KEYS = [
    "model",
    "encoders",
    "feature_columns",
    "imputation_values",
    "risk_tier_cutoffs",
    "model_name",
    "metrics",
    "training_summary",
    "test_predictions",
    "train_probabilities_sorted",
    "examples",
    "default_values",
    "coefficients",
    "permutation_importance",
]

_METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "confusion_matrix",
    "threshold",
]

_SUMMARY_KEYS = ["n_rows", "n_train", "n_test", "positive_rate", "n_features"]


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw employee rows into the model feature frame.

    Deterministic: no fitting happens here. The target is created when
    ``TanggalResign`` is present, then the resignation column is dropped.
    """
    _require_columns(raw, schema.RAW_INPUT_COLUMNS)
    frame = raw.copy()

    object_columns = [
        column
        for column in frame.columns
        if pd.api.types.is_object_dtype(frame[column].dtype)
        or pd.api.types.is_string_dtype(frame[column].dtype)
    ]
    for column in object_columns:
        frame[column] = frame[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    frame["TanggalLahir"] = pd.to_datetime(
        frame["TanggalLahir"], errors="coerce", format="mixed"
    )
    frame["TanggalHiring"] = pd.to_datetime(
        frame["TanggalHiring"], errors="coerce", format="mixed"
    )

    has_target = "TanggalResign" in frame.columns
    if has_target:
        resign = pd.to_datetime(
            frame["TanggalResign"], errors="coerce", format="mixed"
        )
        frame[schema.TARGET_COLUMN] = resign.notna().astype(int)
        frame = frame.drop(columns=["TanggalResign"])

    for column, mapping in schema.CATEGORY_MAPS.items():
        if column in frame.columns:
            frame[column] = frame[column].replace(mapping)

    frame["Pekerjaan"] = (
        frame["Pekerjaan"].map(schema.JOB_FAMILY_MAP).fillna("Other")
    )

    for column in schema.RAW_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    tenure_days = (frame["TanggalHiring"] - frame["TanggalLahir"]).dt.days
    frame["UsiaSaatMasuk"] = (tenure_days / 365.25).where(tenure_days >= 0)
    frame["TahunHiring"] = frame["TanggalHiring"].dt.year
    frame["BulanHiring"] = frame["TanggalHiring"].dt.month

    columns = list(schema.MODEL_FEATURES)
    if has_target:
        columns.append(schema.TARGET_COLUMN)
    return frame[columns].reset_index(drop=True)


def fit_imputation_values(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the training values used to fill missing model inputs."""
    values: dict[str, Any] = {}

    if "StatusPernikahan" in frame.columns:
        mode = frame["StatusPernikahan"].mode()
        if mode.empty:
            raise ValueError("Cannot impute StatusPernikahan: no observed values")
        values["StatusPernikahan"] = mode.iloc[0]

    for column in _IMPUTATION_NUMERIC:
        if column not in frame.columns:
            continue
        median = frame[column].median()
        values[column] = 0.0 if pd.isna(median) else float(median)

    return values


def apply_imputation_values(
    frame: pd.DataFrame, values: dict[str, Any]
) -> pd.DataFrame:
    """Fill missing values with the values fitted on the training split."""
    missing = [column for column in values if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing imputation columns: {', '.join(missing)}"
        )
    imputed = frame.copy()
    for column, value in values.items():
        imputed[column] = imputed[column].fillna(value)
    return imputed


def encode_features(
    frame: pd.DataFrame, encoders: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ordinal- and one-hot-encode the categorical model inputs."""
    return feature_encoding(
        frame,
        ordinal_columns=schema.ORDINAL_COLUMNS,
        ordinal_categories=schema.ORDINAL_CATEGORIES,
        nominal_columns=schema.NOMINAL_COLUMNS,
        drop_first=True,
        handle_unknown="ignore",
        encoders=encoders,
    )


def align_features(
    encoded: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Force the model's exact column order, filling absent dummies with 0."""
    return encoded.reindex(columns=feature_columns, fill_value=0)


def load_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate a persisted attrition artifact."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Artifact must be a dictionary")

    version = artifact.get("artifact_version")
    if version != schema.ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version: {version!r}; "
            f"expected {schema.ARTIFACT_VERSION}"
        )

    missing = [key for key in _ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ValueError(f"Artifact is missing keys: {', '.join(missing)}")

    model = artifact["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("Artifact model must expose predict_proba")

    if not isinstance(artifact["encoders"], dict):
        raise ValueError("Artifact encoders must be a dictionary")
    if not isinstance(artifact["feature_columns"], list):
        raise ValueError("Artifact feature_columns must be a list")
    if not isinstance(artifact["imputation_values"], dict):
        raise ValueError("Artifact imputation_values must be a dictionary")

    cutoffs = artifact["risk_tier_cutoffs"]
    if not isinstance(cutoffs, dict) or not {"moderate", "high"} <= set(cutoffs):
        raise ValueError(
            "Artifact risk_tier_cutoffs must contain moderate and high"
        )

    metrics = artifact["metrics"]
    if not isinstance(metrics, dict):
        raise ValueError("Artifact metrics must be a dictionary")
    missing_metrics = [key for key in _METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise ValueError(
            f"Artifact metrics missing keys: {', '.join(missing_metrics)}"
        )

    summary = artifact["training_summary"]
    if not isinstance(summary, dict):
        raise ValueError("Artifact training_summary must be a dictionary")
    missing_summary = [key for key in _SUMMARY_KEYS if key not in summary]
    if missing_summary:
        raise ValueError(
            "Artifact training_summary missing keys: "
            + ", ".join(missing_summary)
        )

    predictions = artifact["test_predictions"]
    if not isinstance(predictions, pd.DataFrame) or not {
        "y_true",
        "y_proba",
    } <= set(predictions.columns):
        raise ValueError(
            "Artifact test_predictions must be a DataFrame with "
            "y_true and y_proba"
        )

    if not isinstance(artifact["train_probabilities_sorted"], (list, tuple)):
        raise ValueError(
            "Artifact train_probabilities_sorted must be a sequence"
        )

    examples = artifact["examples"]
    if not isinstance(examples, dict) or not {
        "high_risk",
        "low_risk",
    } <= set(examples):
        raise ValueError(
            "Artifact examples must contain high_risk and low_risk"
        )
    for name in ("high_risk", "low_risk"):
        missing_example = [
            column
            for column in schema.RAW_INPUT_COLUMNS
            if column not in examples[name]
        ]
        if missing_example:
            raise ValueError(
                f"Artifact example '{name}' is missing columns: "
                + ", ".join(missing_example)
            )

    if not isinstance(artifact["default_values"], dict):
        raise ValueError("Artifact default_values must be a dictionary")
    if not isinstance(artifact["permutation_importance"], pd.DataFrame):
        raise ValueError(
            "Artifact permutation_importance must be a DataFrame"
        )
    coefficients = artifact["coefficients"]
    if coefficients is not None and not isinstance(coefficients, pd.DataFrame):
        raise ValueError(
            "Artifact coefficients must be a DataFrame or None"
        )

    return artifact


def transform_raw(artifact: dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    """Run clean -> impute -> encode -> align for a raw employee frame."""
    clean = clean_raw(raw)
    if schema.TARGET_COLUMN in clean.columns:
        clean = clean.drop(columns=[schema.TARGET_COLUMN])
    clean = apply_imputation_values(clean, artifact["imputation_values"])
    encoded, _ = encode_features(clean, encoders=artifact["encoders"])
    return align_features(encoded, artifact["feature_columns"])


def predict_proba(artifact: dict[str, Any], raw: pd.DataFrame) -> np.ndarray:
    """Attrition probability for every raw row."""
    model_input = transform_raw(artifact, raw)
    return artifact["model"].predict_proba(model_input)[:, 1]


def predict_frame(artifact: dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    """Return the raw rows with an ``attrition_probability`` column."""
    result = raw.copy().reset_index(drop=True)
    result["attrition_probability"] = predict_proba(artifact, raw)
    return result


def _split_model(model: Any) -> tuple[Any, Any]:
    """Return (preprocessor, estimator) for a pipeline or a bare estimator."""
    steps = getattr(model, "steps", None)
    if steps:
        estimator = steps[-1][1]
        if len(steps) > 1:
            return model[:-1], estimator
        return None, estimator
    return None, model


def explain_row(
    artifact: dict[str, Any],
    encoded_row: pd.DataFrame,
    top_n: int | None = 8,
) -> pd.DataFrame:
    """Per-row linear contribution: standardized input times coefficient."""
    columns = ["feature", "input", "coefficient", "contribution", "direction"]
    preprocessor, estimator = _split_model(artifact["model"])
    if not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=columns)

    if preprocessor is None:
        model_input = encoded_row.to_numpy()
    else:
        model_input = preprocessor.transform(encoded_row)

    input_row = np.asarray(model_input)[0]
    coefficients = np.asarray(estimator.coef_).reshape(-1)

    contributions = pd.DataFrame(
        {
            "feature": artifact["feature_columns"],
            "input": input_row,
            "coefficient": coefficients,
        }
    )
    contributions["contribution"] = (
        contributions["coefficient"] * contributions["input"]
    )
    contributions["direction"] = np.where(
        contributions["contribution"] >= 0,
        "Increases attrition odds",
        "Decreases attrition odds",
    )

    order = contributions["contribution"].abs().sort_values(
        ascending=False
    ).index
    contributions = contributions.loc[order]
    if top_n is not None:
        contributions = contributions.head(top_n)
    return contributions.reset_index(drop=True)
