# Improving Employee Retention By Predicting Employee Attrition

![Project header](assets/header.png)

**Employee attrition risk ranking for an Indonesian technology company — a leakage-safe notebook, a versioned model contract, and a Streamlit decision-support app.**

![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/dependencies-uv-DE5FE9)
![Streamlit](https://img.shields.io/badge/Streamlit-1.67-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-43-2ea44f)

## Overview

An Indonesian technology company loses about 31% of its employees to resignation. HR cannot tell which employees are
most likely to leave, so retention conversations and engagement budgets are spread evenly instead of focused where
they can change an outcome.

This project builds a **leakage-safe attrition classifier** from 287 employee records and ships it as a four-tab
Streamlit app. The analysis is deliberately conservative: identifiers and post-resignation fields are excluded,
imputation and encoding are fitted on the training split only, the model is selected on repeated stratified
cross-validation, and the held-out split is reported once.

> **Honest result:** the tuned Decision Tree reaches 0.661 CV ROC-AUC but 0.544 on the held-out split
> (95% bootstrap CI 0.407–0.673). The dataset does not support a high-confidence individual predictor. The project
> therefore ships a **ranking and screening aid**, the audit that proves the limit, and a working deployment path —
> not a claimed accuracy figure.

## Headline Results

The 30% stratified holdout contains 87 employees, 27 of whom resigned. The deployed model is a tuned Decision Tree;
Logistic Regression was the explainability fallback if it came within 0.02 CV ROC-AUC, but it finished 0.082 behind.

| Deployed model          | CV ROC-AUC | Holdout ROC-AUC | Recall | Precision | PR-AUC |
| ----------------------- | ---------: | --------------: | -----: | --------: | -----: |
| **Decision Tree (tuned)** |     0.661 |           0.544 |  0.185 |     0.278 |  0.335 |

Capacity gains on the holdout:

| Employees targeted | Leavers captured | Share of leavers | Lift |
| -----------------: | ---------------: | ---------------: | ---: |
|          10% (9)  |                3 |            11.1% | 1.11 |
|          20% (18) |                5 |            18.5% | 0.93 |
|          30% (27) |               10 |            37.0% | 1.23 |
|          50% (44) |               16 |            59.3% | 1.19 |

The top 20% captures slightly **less** than random; the 30% and 50% cuts beat random modestly. The model is not ready
for high-stakes targeting. A label-permutation test on the deployed tree returns p=0.007, but the fixed Logistic
Regression returns p=0.106 and the holdout CI includes 0.5 — the honest read is a weak, unstable signal, not a
deployable predictor.

## Engineering Highlights

- **Leakage-safe by design.** `Username`, `EnterpriseID`, `NomorHP`, and `Email` are excluded; `AlasanResign` is known
  only after resignation; `TanggalResign` defines the target; `TanggalPenilaianKaryawan` is after resignation for
  64 of 89 leavers. The model never sees them.
- **One training/serving contract.** The notebook and app import the same `employee_attrition/pipeline.py`; category
  maps, imputation, encoding, and feature order cannot drift.
- **Self-describing artifact.** `models/attrition_model.joblib` carries the fitted pipeline, encoders, feature order,
  metrics, risk-tier cutoffs, test predictions, and anonymized examples. The app validates every key it reads and
  refuses a partial artifact with the notebook command.
- **Honest evaluation.** Repeated stratified 5-fold CV (3 repeats) for selection, one held-out 30% split for
  reporting, a majority-class baseline, and capacity gains instead of accuracy theatre.
- **Non-linear fallback.** The deployed tree has no coefficients; the app shows global permutation importance for the
  prediction instead of pretending to decompose it.
- **Tested beyond the happy path.** 43 automated checks cover data contracts, date-object inputs, unknown categories,
  missing columns, partial artifacts, PII absence, batch percent display, the Streamlit boot path, and a real
  Predict-form submit.

## Dataset

Source: `data/employee_performance_data.csv` — 287 employee records, 25 columns, one row per employee.

| | Value |
| --- | ---: |
| Employees | 287 |
| Resigned (`TanggalResign` present) | 89 (31.0%) |
| Still working | 198 (69.0%) |
| Effective model inputs | 17 before encoding / 34 encoded |
| Missing `IkutProgramLOP` | 258 (89.9%) |
| Leavers with assessment after resignation | 64 / 89 |
| Held-out test rows | 87 (27 leavers) |

Data-quality findings that shaped the analysis:

- **PII is dropped, not modelled.** Usernames, employee IDs, phone numbers, and emails never reach the model,
  artifact, or UI.
- **`PernahBekerja` is an invalid constant** — one `yes`, otherwise `1` — and `IkutProgramLOP` is 89.9% missing.
  Both are excluded.
- **The assessment date is not causal.** For most leavers it is recorded after `TanggalResign`; tenure-at-assessment
  features would leak the outcome, so they are excluded.
- **The feature set is thin.** `Pekerjaan` is dominated by software engineers; `AsalDaerah` is entirely Jakarta;
  several job families have fewer than five employees and were grouped before encoding.
- **The signal is weak.** Attrition sits around 23%–40% across job family, career level, performance, and engagement.
  No single category separates leavers from stayers.
- **The target is retrospective.** It is "ever resigned" from a static profile, not "will resign in the next six
  months." A time-aware snapshot or survival model is the natural next step.

## Approach

The notebook is organised into four stages, each with its decisions recorded in markdown:

1. **Exploratory analysis** — audit the raw data, derive the target, document missingness and date inconsistencies,
   and inspect univariate, bivariate, and multivariate patterns.
2. **Preprocessing** — stratified 70/30 split before fitting; train-only imputation and encoding; VIF and univariate
   importance as diagnostics rather than leaky selection.
3. **Modelling** — majority-class baseline, six classifiers, random-search tuning with `utils.tune_pipelines`,
   repeated stratified CV, and the 0.02 explainability rule.
4. **Evaluation and deployment** — confusion matrix, ROC/PR/calibration/threshold diagnostics, permutation importance,
   risk tiers, cumulative gains, business recommendations, and a versioned artifact with a reload test.

## The Studio

`Employee Retention Analyzer` is a Streamlit app over the shipped artifact:

- **Predict** — enter the 16 profile, employment, engagement, and date fields; get a risk tier, attrition probability,
  percentile, and the drivers behind the score. High- and low-risk example profiles load in one click.
- **Batch** — upload a CSV (or score the built-in eight-row example from the training data), validate required columns,
  score every row, and download a PII-free CSV with probability and risk tier.
- **Insights** — ROC-AUC, PR-AUC, recall, precision, confusion matrix, cumulative gains, global drivers, and
  attrition patterns by job family, career level, performance, and engagement.
- **About** — model card, data dictionary for all 25 source columns, limitations, and author links.

## Tech Stack

| Layer       | Tools                                                         |
| ----------- | ------------------------------------------------------------- |
| Language    | Python 3.14                                                   |
| Packaging   | **uv**, `pyproject.toml`, `uv.lock`, `.python-version`        |
| Analysis    | pandas, scikit-learn, XGBoost, `utils/` (vendored toolkit)    |
| App         | Streamlit, Altair                                             |
| Persistence | joblib artifact                                               |
| Testing     | pytest, Streamlit `AppTest`                                   |

## Quickstart

The project is managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates `.venv` from the lockfile, and
`uv run` executes inside it.

```powershell
# 1. Install uv (Windows)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the locked environment
uv sync

# 3. Regenerate the model artifact (about 10 minutes; tuning dominates)
uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb --ExecutePreprocessor.timeout=1800

# 4. Launch the app
uv run streamlit run main.py

# 5. Run the test suite
uv run pytest -q
```

The notebook writes `models/attrition_model.joblib` as its final step. The app refuses to start without it and tells
you to run the notebook. If you prefer a Jupyter kernel, point your editor at the `.venv` interpreter created by
`uv sync`.

## Project Structure

```
.
├── notebook.ipynb                  four-stage analysis; writes models/
├── main.py                         Streamlit entry point
├── employee_attrition/             app and serving package
│   ├── config.py                   paths only
│   ├── schema.py                   feature contract, categories, data dictionary
│   ├── pipeline.py                 cleaning, imputation, encoding, prediction, explanation
│   ├── analytics.py                risk tiers, gains, attrition-by-group helpers
│   ├── ui.py                       Altair builders (no Streamlit import)
│   └── views/                      predict, batch, insights, about
├── utils/                          vendored ML toolkit
├── data/                           employee performance CSV
├── models/                         generated artifact
├── assets/                         README header
├── tests/                          schema, pipeline, analytics, and AppTest suite
├── .streamlit/config.toml          indigo/amber theme
├── pyproject.toml
├── uv.lock
└── .python-version
```

## Testing

```powershell
uv run pytest -q
```

Covers the feature contract, `clean_raw` date objects and mixed numeric/object columns, missing columns, unknown
categories, imputation, encoding alignment, prediction order, artifact version/key/nested-metric refusal, PII
absence, batch percent display, app boot with four tabs, and a real Predict-form submit. The suite also passes from a
fresh clone after `uv sync`.

## Limitations

- **Small sample.** 287 rows, 89 leavers, 87 test rows. The bootstrap 95% CI on holdout AUC is 0.407–0.673; the
  estimate is noisy.
- **Weak, unstable signal.** The tuned tree scores 0.661 CV ROC-AUC but 0.544 on the holdout; the top-20% gains lift
  is 0.93. It is not a production predictor.
- **Date inconsistency.** 64 of 89 leavers have an assessment date after resignation, so assessment-date features are
  unusable.
- **Snapshot, not a time series.** The target is "ever resigned," not a six-month forecast. A survival or time-to-event
  model is the next step if dated snapshots become available.
- **One dominant feature.** Permutation importance is concentrated on age at hire; that is not a policy lever and
  should not be read as causation.
- **No PII.** Identifiers, phone numbers, and emails are excluded from the model, artifact, and UI. Example profiles
  use a synthetic hire anchor and a derived birth date that preserves age at hire.

## Future Work

- Collect the missing retention drivers: compensation, manager quality, promotion velocity, commute, exit-interview
  themes, and repeated engagement-survey waves.
- Reframe the target as a fixed-horizon resignation risk from a clean snapshot, and evaluate with a chronological or
  rolling-origin split.
- Add nested cross-validation and probability calibration, then choose the outreach threshold from HR capacity and
  the gains curve.
- Monitor drift and retrain after each review cycle.

## References

- Dataset: [`data/employee_performance_data.csv`](data/employee_performance_data.csv)
- Shared toolkit: [`utils/`](utils/)
- Notebook: [`notebook.ipynb`](notebook.ipynb)

## Author

**Muhammad Cikal Merdeka** — Data Analyst / Data Scientist

[GitHub](https://github.com/mcikalmerdeka) · [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
