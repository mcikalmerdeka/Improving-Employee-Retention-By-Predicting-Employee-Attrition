import pandas as pd

from employee_attrition import schema

RAW = pd.read_csv("data/employee_performance_data.csv")


def test_raw_input_columns_match_spec():
    assert len(schema.RAW_INPUT_COLUMNS) == 16
    assert schema.RAW_INPUT_COLUMNS[-2:] == ["TanggalLahir", "TanggalHiring"]


def test_model_features_are_complete():
    expected = (
        set(schema.NUMERIC_COLUMNS)
        | set(schema.ORDINAL_COLUMNS)
        | set(schema.NOMINAL_COLUMNS)
    )
    assert set(schema.MODEL_FEATURES) == expected
    assert len(schema.MODEL_FEATURES) == 17


def test_ordinal_categories_cover_columns():
    assert set(schema.ORDINAL_CATEGORIES) == set(schema.ORDINAL_COLUMNS)


def test_every_job_value_maps():
    assert not (set(RAW["Pekerjaan"]) - set(schema.JOB_FAMILY_MAP))


def test_feature_definitions_cover_every_raw_column():
    assert set(schema.FEATURE_DEFINITIONS) == set(RAW.columns) | {schema.TARGET_COLUMN}


def test_pii_columns_are_not_model_features():
    assert set(schema.PII_COLUMNS).isdisjoint(schema.MODEL_FEATURES)


def test_category_options_cover_form_columns():
    assert set(schema.CATEGORY_OPTIONS) == {
        "StatusPernikahan",
        "JenisKelamin",
        "StatusKepegawaian",
        "Pekerjaan",
        "JenjangKarir",
        "PerformancePegawai",
        "AsalDaerah",
        "HiringPlatform",
        "TingkatPendidikan",
    }
