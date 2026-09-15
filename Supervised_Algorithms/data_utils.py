"""
data_utils.py
=============
Common data loading, preprocessing pipelines, and evaluation helpers for
all supervised machine learning algorithm scripts in PlacementPredict.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

POSSIBLE_DATA_PATHS = [
    os.path.join(PROJECT_ROOT, "Data", "placement_predict_50k Dataset.csv"),
    os.path.join(CURRENT_DIR, "Data", "placement_predict_50k Dataset.csv"),
    os.path.join(PROJECT_ROOT, "placement_predict_50k Dataset.csv"),
    r"D:\T-2-1\ML\Placement_Prediction\Data\placement_predict_50k Dataset.csv",
    r"D:\T-2-1\ML\placement_predict_50k Dataset.csv",
]

DROP_COLS = ["StudentID", "IsAnomaly", "PlacementStatus", "Salary Package"]
CATEGORICAL_NOMINAL = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
CATEGORICAL_ORDINAL = ["CollegeTier", "CGPA_Tier"]
TIER_ORDER = [["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]]


def get_data_path() -> str:
    """Finds the dataset across project directory variations."""
    for p in POSSIBLE_DATA_PATHS:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        f"Could not locate 'placement_predict_50k Dataset.csv'. Checked locations:\n"
        + "\n".join(POSSIBLE_DATA_PATHS)
    )


def load_dataset(sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    """Loads CSV dataset, optionally sampling records for fast exploration."""
    data_path = get_data_path()
    df = pd.read_csv(data_path)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df


def build_preprocessor(scaling: str = "standard") -> ColumnTransformer:
    """
    Constructs a ColumnTransformer:
    - Numeric features: Median imputation + StandardScaler (or MinMaxScaler)
    - Nominal features: Most-frequent imputation + OneHotEncoder
    - Ordinal features: Most-frequent imputation + OrdinalEncoder
    """
    scaler = StandardScaler() if scaling == "standard" else MinMaxScaler()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    nom_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=TIER_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, make_column_selector(dtype_include=np.number)),
            ("nom", nom_pipe, CATEGORICAL_NOMINAL),
            ("ord", ord_pipe, CATEGORICAL_ORDINAL)
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_classification_data(
    sample_size: Optional[int] = 10000,
    scaling: str = "standard",
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares train and test splits for binary classification:
    Target: PlacementStatus (0 = Not Placed, 1 = Placed)
    """
    df = load_dataset(sample_size=sample_size, random_state=random_state)
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    y = df["PlacementStatus"].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(scaling=scaling)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    try:
        raw_feature_names = preprocessor.get_feature_names_out().tolist()
        feature_names = [
            f.replace("num__", "").replace("nom__", "").replace("ord__", "")
            for f in raw_feature_names
        ]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "class_counts": y.value_counts().to_dict(),
        "total_records": len(df)
    }


def prepare_regression_data(
    sample_size: Optional[int] = 10000,
    scaling: str = "standard",
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Prepares train and test splits for regression:
    Target: Salary Package (LPA, for Placed students with Salary > 0)
    """
    df = load_dataset(sample_size=sample_size, random_state=random_state)
    reg_mask = (df["Salary Package"] > 0) & (df["PlacementStatus"] == 1)
    df_reg = df[reg_mask].copy().reset_index(drop=True)

    X = df_reg.drop(columns=[c for c in DROP_COLS if c in df_reg.columns], errors="ignore")
    y = df_reg["Salary Package"].astype(float)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(scaling=scaling)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    try:
        raw_feature_names = preprocessor.get_feature_names_out().tolist()
        feature_names = [
            f.replace("num__", "").replace("nom__", "").replace("ord__", "")
            for f in raw_feature_names
        ]
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
        "salary_stats": {
            "mean": round(y.mean(), 2),
            "std": round(y.std(), 2),
            "min": round(y.min(), 2),
            "max": round(y.max(), 2)
        },
        "total_records": len(df_reg)
    }


def get_sample_student() -> Dict[str, Any]:
    """Returns a realistic candidate student profile dictionary for live inference demos."""
    return {
        "Gender": "Male",
        "City": "Bangalore",
        "CollegeTier": "Tier1",
        "Stream": "Computer Science",
        "Specialisation": "Data Science",
        "Hostel": "No",
        "HistoryOfBacklogs": "No",
        "SGPA_Sem1": 8.1,
        "SGPA_Sem2": 8.3,
        "SGPA_Sem3": 8.5,
        "SGPA_Sem4": 8.2,
        "SGPA_Sem5": 8.7,
        "SGPA_Sem6": 8.6,
        "SGPA_Sem7": 8.9,
        "SGPA_Sem8": 8.8,
        "CGPA": 8.5,
        "AttendancePercent": 88.5,
        "Internships": 2,
        "Projects": 3,
        "Workshops": 2,
        "Certifications": 3,
        "Publications": 1,
        "AptitudeTestScore": 82.0,
        "SoftSkillsRating": 4.5,
        "CodingTestScore": 85.0,
        "MockInterviewScore": 84.0,
        "ExtraCurricular": 1,
        "CGPA_Tier": "High"
    }


def print_section(title: str, width: int = 80):
    """Utility function to print formatted CLI headers."""
    print("\n" + "=" * width)
    print(f" {title.upper()}")
    print("=" * width)
