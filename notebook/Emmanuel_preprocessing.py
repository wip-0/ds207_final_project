"""
Preprocessing Pipeline — Diabetic Patient Readmission Data
==========================================================
Covers:
  1. Drop uninformative / leakage columns
  2. Target encoding  (binary or 3-class)
  3. Missing value handling
  4. ICD-9 diagnosis bucketing
  5. Feature engineering
  6. Column-type definitions for ColumnTransformer
  7. Full sklearn Pipeline (numeric + categorical)
  8. Train/test split (patient-aware to avoid leakage)
  9. Quick sanity-check output

Usage
-----
Run standalone:
    python preprocessing.py

Or import into a notebook:
    from preprocessing import build_pipeline, load_data, encode_target
"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit

# ─────────────────────────────────────────────
# 0. CONFIGURATION  — tweak here as needed
# ─────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/wip-0/ds207_final_project"
    "/main/data/source/diabetic_data.csv"
)

TARGET_MODE = "binary"   # "binary"  → NO=0, readmitted=1
                          # "three_class" → NO=0, >30=1, <30=2

TEST_SIZE   = 0.2         # fraction held out for testing
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
def load_data(path: str = DATA_URL) -> pd.DataFrame:
    """Load raw CSV, treating '?' as NaN."""
    df = pd.read_csv(path, na_values="?", low_memory=False)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 2. DROP UNINFORMATIVE / HIGH-LEAKAGE COLUMNS
# ─────────────────────────────────────────────
# We keep patient_nbr only to do a patient-aware split, then drop it.
COLS_TO_DROP = [
    "encounter_id",          # pure ID
    "examide",               # 100% 'No' — zero variance
    "citoglipton",           # 100% 'No' — zero variance
    "weight",                # 97% missing, clinical not collected systematically
    "payer_code",            # high missingness, not predictive of readmission
    "medical_specialty",     # 72 levels, high missingness — drop or group later
]

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=cols)


# ─────────────────────────────────────────────
# 3. TARGET ENCODING
# ─────────────────────────────────────────────
def encode_target(df: pd.DataFrame, mode: str = TARGET_MODE) -> pd.DataFrame:
    """
    binary     : NO → 0, '<30' or '>30' → 1
    three_class: NO → 0,  '>30' → 1,  '<30' → 2
    """
    df = df.copy()
    if mode == "binary":
        df["readmitted"] = (df["readmitted"] != "NO").astype(int)
        print("Target encoded (binary): 0=NO readmission, 1=readmitted")
    elif mode == "three_class":
        mapping = {"NO": 0, ">30": 1, "<30": 2}
        df["readmitted"] = df["readmitted"].map(mapping)
        print("Target encoded (3-class): 0=NO, 1=>30 days, 2=<30 days")
    else:
        raise ValueError("mode must be 'binary' or 'three_class'")
    return df


# ─────────────────────────────────────────────
# 4. ICD-9 DIAGNOSIS BUCKETING
# ─────────────────────────────────────────────
def bucket_icd9(code) -> str:
    """Map an ICD-9 code string to a broad disease category."""
    if pd.isna(code):
        return "Unknown"
    code = str(code).strip()
    # V/E codes
    if code.startswith("V"):
        return "Supplementary"
    if code.startswith("E"):
        return "External_Cause"
    try:
        num = float(code)
    except ValueError:
        return "Other"

    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    elif 460 <= num <= 519 or num == 786:
        return "Respiratory"
    elif 520 <= num <= 579 or num == 787:
        return "Digestive"
    elif 250 <= num < 251:
        return "Diabetes"
    elif 800 <= num <= 999:
        return "Injury"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    else:
        return "Other"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- ICD-9 bucketing ---
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(bucket_icd9)

    # --- A1C / glucose: binary "was measured" flags ---
    for col in ["A1Cresult", "max_glu_serum"]:
        if col in df.columns:
            df[f"{col}_measured"] = (df[col].notna()).astype(int)

    # --- Age: ordinal integer (decade buckets) ---
    age_map = {
        "[0-10)":   0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
        "[40-50)":  4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7,
        "[80-90)":  8, "[90-100)":9
    }
    if "age" in df.columns:
        df["age"] = df["age"].map(age_map)

    # --- Medication columns: collapse to binary (0=No, 1=active) ---
    med_cols = [
        "metformin","repaglinide","nateglinide","chlorpropamide",
        "glimepiride","acetohexamide","glipizide","glyburide","tolbutamide",
        "pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone",
        "tolazamide","insulin","glyburide-metformin","glipizide-metformin",
        "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone"
    ]
    for col in med_cols:
        if col in df.columns:
            df[col] = (df[col] != "No").astype(int)

    # --- Gender: drop the 3 'Unknown/Invalid' rows ---
    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"]

    # --- Drop A1C / glucose raw columns (replaced by flag) ---
    df = df.drop(columns=["A1Cresult", "max_glu_serum"], errors="ignore")

    return df


# ─────────────────────────────────────────────
# 6. DEFINE COLUMN GROUPS FOR COLUMNTRANSFORMER
# ─────────────────────────────────────────────
NUMERIC_COLS = [
    "age",                       # now ordinal integer 0–9
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    # binary medication flags (already 0/1, scaling won't hurt)
    "metformin","repaglinide","nateglinide","chlorpropamide",
    "glimepiride","acetohexamide","glipizide","glyburide","tolbutamide",
    "pioglitazone","rosiglitazone","acarbose","miglitol","troglitazone",
    "tolazamide","insulin","glyburide-metformin","glipizide-metformin",
    "glimepiride-pioglitazone","metformin-rosiglitazone","metformin-pioglitazone",
    # engineered binary flags
    "A1Cresult_measured", "max_glu_serum_measured",
]

CATEGORICAL_COLS = [
    "race",
    "gender",
    "diag_1", "diag_2", "diag_3",   # bucketed ICD-9
    "change",
    "diabetesMed",
]


# ─────────────────────────────────────────────
# 7. BUILD SKLEARN PIPELINE
# ─────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Returns a fitted-ready sklearn Pipeline with:
      - Numeric: median imputation → StandardScaler
      - Categorical: most-frequent imputation → OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,  NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ],
        remainder="drop",   # silently drops any leftover columns
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
    ])

    return pipeline


# ─────────────────────────────────────────────
# 8. PATIENT-AWARE TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def patient_aware_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                         random_state: int = RANDOM_SEED):
    """
    Split by patient_nbr so no patient appears in both train and test.
    This prevents data leakage from repeated encounters.
    """
    groups = df["patient_nbr"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                            random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # Drop patient_nbr — not a feature
    train_df = train_df.drop(columns=["patient_nbr"])
    test_df  = test_df.drop(columns=["patient_nbr"])

    print(f"Train: {len(train_df):,} rows  |  Test: {len(test_df):,} rows")
    print(f"Train patients: {df.iloc[train_idx]['patient_nbr'].nunique():,}  "
          f"Test patients: {df.iloc[test_idx]['patient_nbr'].nunique():,}")
    return train_df, test_df


# ─────────────────────────────────────────────
# 9. MAIN — end-to-end demo
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Step 1: load
    df_raw = load_data()

    # Step 2: drop junk columns
    df = drop_columns(df_raw)

    # Step 3: encode target
    df = encode_target(df, mode=TARGET_MODE)

    # Step 4 & 5: engineer features
    df = engineer_features(df)

    # Step 6: patient-aware split  (done BEFORE fitting the pipeline)
    train_df, test_df = patient_aware_split(df)

    # Step 7: separate X / y
    X_train = train_df.drop(columns=["readmitted"])
    y_train = train_df["readmitted"]
    X_test  = test_df.drop(columns=["readmitted"])
    y_test  = test_df["readmitted"]

    # Step 8: build & fit pipeline
    pipeline = build_pipeline()
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc  = pipeline.transform(X_test)

    # Step 9: sanity check
    print(f"\nX_train processed shape: {X_train_proc.shape}")
    print(f"X_test  processed shape: {X_test_proc.shape}")
    print(f"\ny_train distribution:\n{y_train.value_counts().to_string()}")
    print(f"\ny_test distribution:\n{y_test.value_counts().to_string()}")
    print("\n✅  Preprocessing complete — pipeline ready for modelling.")
