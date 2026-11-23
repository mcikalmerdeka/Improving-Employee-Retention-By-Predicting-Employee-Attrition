"""Single source of truth for the employee-attrition feature contract.

Both the training notebook and the Streamlit app import these constants, so a
column rename or category change happens in exactly one place.
"""

from __future__ import annotations

TARGET_COLUMN = "StatusKerja"
TARGET_LABELS = {0: "Masih Bekerja", 1: "Tidak Bekerja"}
DISPLAY_TARGET_LABELS = {0: "Still working", 1: "Resigned"}
ARTIFACT_VERSION = 1

RAW_INPUT_COLUMNS = [
    "StatusPernikahan",
    "JenisKelamin",
    "StatusKepegawaian",
    "Pekerjaan",
    "JenjangKarir",
    "PerformancePegawai",
    "AsalDaerah",
    "HiringPlatform",
    "SkorSurveyEngagement",
    "SkorKepuasanPegawai",
    "JumlahKeikutsertaanProjek",
    "JumlahKeterlambatanSebulanTerakhir",
    "JumlahKetidakhadiran",
    "TingkatPendidikan",
    "TanggalLahir",
    "TanggalHiring",
]

NUMERIC_COLUMNS = [
    "UsiaSaatMasuk",
    "TahunHiring",
    "BulanHiring",
    "SkorSurveyEngagement",
    "SkorKepuasanPegawai",
    "JumlahKeikutsertaanProjek",
    "JumlahKeterlambatanSebulanTerakhir",
    "JumlahKetidakhadiran",
]

ORDINAL_COLUMNS = [
    "JenisKelamin",
    "JenjangKarir",
    "PerformancePegawai",
    "TingkatPendidikan",
]

ORDINAL_CATEGORIES = {
    "JenisKelamin": ["Pria", "Wanita"],
    "JenjangKarir": ["Freshgraduate Program", "Mid Level", "Senior Level"],
    "PerformancePegawai": [
        "Sangat Kurang",
        "Kurang",
        "Biasa",
        "Bagus",
        "Sangat Bagus",
    ],
    "TingkatPendidikan": ["Sarjana", "Magister", "Doktor"],
}

NOMINAL_COLUMNS = [
    "StatusPernikahan",
    "StatusKepegawaian",
    "Pekerjaan",
    "AsalDaerah",
    "HiringPlatform",
]

MODEL_FEATURES = NUMERIC_COLUMNS + ORDINAL_COLUMNS + NOMINAL_COLUMNS

RAW_NUMERIC_COLUMNS = [
    "SkorSurveyEngagement",
    "SkorKepuasanPegawai",
    "JumlahKeikutsertaanProjek",
    "JumlahKeterlambatanSebulanTerakhir",
    "JumlahKetidakhadiran",
]

RAW_CATEGORICAL_COLUMNS = ORDINAL_COLUMNS + NOMINAL_COLUMNS

JOB_FAMILY_MAP = {
    "Software Engineer (Back End)": "Software Engineer",
    "Software Engineer (Front End)": "Software Engineer",
    "Software Engineer (Android)": "Software Engineer",
    "Software Engineer (iOS)": "Software Engineer",
    "Software Architect": "Platform & Architecture",
    "DevOps Engineer": "Platform & Architecture",
    "Product Design (UI & UX)": "Product Design",
    "Product Design (UX Researcher)": "Product Design",
    "Product Manager": "Product Management",
    "Digital Product Manager": "Product Management",
    "Data Analyst": "Data & ML",
    "Data Engineer": "Data & ML",
    "Machine Learning Engineer": "Data & ML",
    "Scrum Master": "Other",
    # Identity entries keep clean_raw idempotent for already-grouped frames.
    "Software Engineer": "Software Engineer",
    "Platform & Architecture": "Platform & Architecture",
    "Product Design": "Product Design",
    "Product Management": "Product Management",
    "Data & ML": "Data & ML",
    "Other": "Other",
}

STATUS_PERNIKAHAN_MAP = {
    "Belum_menikah": "Belum Menikah",
    "Menikah": "Menikah",
    "Lainnya": "Lainnya",
    "Bercerai": "Bercerai",
    "-": float("nan"),
}

JENJANG_KARIR_MAP = {
    "Freshgraduate_program": "Freshgraduate Program",
    "Mid_level": "Mid Level",
    "Senior_level": "Senior Level",
}

PERFORMANCE_MAP = {
    "Sangat_kurang": "Sangat Kurang",
    "Kurang": "Kurang",
    "Biasa": "Biasa",
    "Bagus": "Bagus",
    "Sangat_bagus": "Sangat Bagus",
}

HIRING_PLATFORM_MAP = {
    "Google_Search": "Google Search",
    "Diversity_Job_Fair": "Diversity Job Fair",
    "Employee_Referral": "Employee Referral",
    "On-line_Web_application": "On-line Web Application",
    "Indeed": "Indeed",
    "LinkedIn": "LinkedIn",
    "CareerBuilder": "CareerBuilder",
    "Website": "Website",
    "Other": "Other",
}

CATEGORY_MAPS = {
    "StatusPernikahan": STATUS_PERNIKAHAN_MAP,
    "JenjangKarir": JENJANG_KARIR_MAP,
    "PerformancePegawai": PERFORMANCE_MAP,
    "HiringPlatform": HIRING_PLATFORM_MAP,
}

CATEGORY_OPTIONS = {
    "StatusPernikahan": ["Belum Menikah", "Menikah", "Lainnya", "Bercerai"],
    "JenisKelamin": ORDINAL_CATEGORIES["JenisKelamin"],
    "StatusKepegawaian": ["FullTime", "Outsource", "Internship"],
    "Pekerjaan": [
        "Software Engineer",
        "Product Design",
        "Product Management",
        "Data & ML",
        "Platform & Architecture",
        "Other",
    ],
    "JenjangKarir": ORDINAL_CATEGORIES["JenjangKarir"],
    "PerformancePegawai": ORDINAL_CATEGORIES["PerformancePegawai"],
    "AsalDaerah": [
        "Jakarta Pusat",
        "Jakarta Timur",
        "Jakarta Barat",
        "Jakarta Utara",
        "Jakarta Selatan",
    ],
    "HiringPlatform": [
        "Indeed",
        "LinkedIn",
        "Google Search",
        "Diversity Job Fair",
        "Employee Referral",
        "CareerBuilder",
        "Website",
        "Other",
        "On-line Web Application",
    ],
    "TingkatPendidikan": ORDINAL_CATEGORIES["TingkatPendidikan"],
}

FEATURE_LABELS = {
    "StatusPernikahan": "Marital status",
    "JenisKelamin": "Gender",
    "StatusKepegawaian": "Employment type",
    "Pekerjaan": "Job family",
    "JenjangKarir": "Career level",
    "PerformancePegawai": "Performance rating",
    "AsalDaerah": "Home region",
    "HiringPlatform": "Hiring source",
    "SkorSurveyEngagement": "Engagement score",
    "SkorKepuasanPegawai": "Satisfaction score",
    "JumlahKeikutsertaanProjek": "Projects joined",
    "JumlahKeterlambatanSebulanTerakhir": "Late arrivals (last month)",
    "JumlahKetidakhadiran": "Absences",
    "TingkatPendidikan": "Education",
    "TanggalLahir": "Date of birth",
    "TanggalHiring": "Hire date",
    "UsiaSaatMasuk": "Age at hire",
    "TahunHiring": "Hiring year",
    "BulanHiring": "Hiring month",
    TARGET_COLUMN: "Attrition",
}

PII_COLUMNS = ["Username", "EnterpriseID", "NomorHP", "Email"]
DATE_COLUMNS = [
    "TanggalLahir",
    "TanggalHiring",
    "TanggalPenilaianKaryawan",
    "TanggalResign",
]
DROPPED_COLUMNS = [
    *PII_COLUMNS,
    "PernahBekerja",
    "IkutProgramLOP",
    "AlasanResign",
    "TanggalLahir",
    "TanggalHiring",
    "TanggalPenilaianKaryawan",
    "TanggalResign",
]


def _definition(
    description: str,
    data_type: str,
    specific_type: str,
    used_by_model: str,
    reason: str = "",
) -> dict[str, str]:
    return {
        "description": description,
        "data_type": data_type,
        "specific_type": specific_type,
        "used_by_model": used_by_model,
        "reason": reason,
    }


FEATURE_DEFINITIONS = {
    "Username": _definition(
        "Username of the employee account", "Categorical", "Nominal", "No",
        "Employee identifier (PII)",
    ),
    "EnterpriseID": _definition(
        "Internal employee ID", "Categorical", "Nominal", "No",
        "Employee identifier (PII)",
    ),
    "StatusPernikahan": _definition(
        "Marital status of the employee", "Categorical", "Nominal", "Yes",
    ),
    "JenisKelamin": _definition(
        "Gender of the employee", "Categorical", "Nominal", "Yes",
    ),
    "StatusKepegawaian": _definition(
        "Employment status", "Categorical", "Nominal", "Yes",
    ),
    "Pekerjaan": _definition(
        "Role of the employee, grouped into job families", "Categorical",
        "Nominal", "Yes",
    ),
    "JenjangKarir": _definition(
        "Career level of the employee", "Categorical", "Ordinal", "Yes",
    ),
    "PerformancePegawai": _definition(
        "Employee performance category", "Categorical", "Ordinal", "Yes",
    ),
    "AsalDaerah": _definition(
        "Employee region of origin", "Categorical", "Nominal", "Yes",
    ),
    "HiringPlatform": _definition(
        "Platform where the application was accepted", "Categorical",
        "Nominal", "Yes",
    ),
    "SkorSurveyEngagement": _definition(
        "Employee engagement survey score (1-5)", "Numerical", "Discrete", "Yes",
    ),
    "SkorKepuasanPegawai": _definition(
        "Employee satisfaction score (1-5)", "Numerical", "Discrete", "Yes",
    ),
    "JumlahKeikutsertaanProjek": _definition(
        "Number of projects the employee joined", "Numerical", "Discrete", "Yes",
    ),
    "JumlahKeterlambatanSebulanTerakhir": _definition(
        "Number of late arrivals in the last month", "Numerical", "Discrete",
        "Yes",
    ),
    "JumlahKetidakhadiran": _definition(
        "Number of absences", "Numerical", "Discrete", "Yes",
    ),
    "NomorHP": _definition(
        "Employee phone number", "Categorical", "Nominal", "No",
        "Personal contact information (PII)",
    ),
    "Email": _definition(
        "Employee email address", "Categorical", "Nominal", "No",
        "Personal contact information (PII)",
    ),
    "TingkatPendidikan": _definition(
        "Highest education level", "Categorical", "Ordinal", "Yes",
    ),
    "PernahBekerja": _definition(
        "Whether the employee had prior work experience", "Categorical",
        "Binary", "No", "Invalid constant (one 'yes' value, otherwise '1')",
    ),
    "IkutProgramLOP": _definition(
        "Whether the employee joined the LOP program", "Categorical",
        "Binary", "No", "~90% missing; imputation would fabricate data",
    ),
    "AlasanResign": _definition(
        "Reason for resignation", "Categorical", "Nominal", "No",
        "Known only after resignation; leaks the target",
    ),
    "TanggalLahir": _definition(
        "Employee birth date", "Date", "Date", "No",
        "Used only to derive UsiaSaatMasuk; raw date dropped",
    ),
    "TanggalHiring": _definition(
        "Employee hire date", "Date", "Date", "No",
        "Used only to derive TahunHiring/BulanHiring; raw date dropped",
    ),
    "TanggalPenilaianKaryawan": _definition(
        "Employee assessment date", "Date", "Date", "No",
        "64 of 89 leavers have an assessment date after resignation; "
        "causally invalid",
    ),
    "TanggalResign": _definition(
        "Employee resignation date", "Date", "Date", "No",
        "Defines the target StatusKerja",
    ),
    "StatusKerja": _definition(
        "Whether the employee resigned (target variable)", "Categorical",
        "Nominal (Binary)", "Target",
    ),
}
