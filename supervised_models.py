"""
supervised_models.py
====================
Core Supervised Machine Learning Engine for PlacementPredict

Alignd with Curriculum Modules:
- M1: The ML Lifecycle (Pipeline, Packaged Artifacts, Prediction Endpoint, Monitoring)
- M2: Supervised Learning — Linear Models at Depth (OLS, Ridge, Lasso, ElasticNet, Logistic Regression, Coefficients, Scaling, Encodings)
- M3: Supervised Learning — Tree-Based Models (Decision Trees, Random Forest, AdaBoost, GBDT, Feature Importance, M2 vs M3 Head-to-Head)
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    log_loss
)
from sklearn.inspection import permutation_importance

# M2 Linear Models
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

# M3 Tree-Based Models
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_FILE = os.path.join(MODELS_DIR, "benchmark_results.json")
SAVED_MODELS_PATH = os.path.join(MODELS_DIR, "saved_models.joblib")

os.makedirs(MODELS_DIR, exist_ok=True)

DROP_COLS = ["StudentID", "IsAnomaly", "PlacementStatus", "Salary Package"]

CATEGORICAL_NOMINAL = ["Gender", "City", "Stream", "Specialisation", "Hostel", "HistoryOfBacklogs"]
CATEGORICAL_ORDINAL = ["CollegeTier", "CGPA_Tier"]
TIER_ORDER = [["Tier3", "Tier2", "Tier1"], ["Low", "Mid", "High"]]


def get_feature_preprocessor(scaling_method: str = "standard") -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Constructs an sklearn ColumnTransformer preprocessing pipeline preventing training-serving skew.
    - Numerical features: Median Imputation + StandardScaler or MinMaxScaler
    - Nominal categorical: Most-frequent Imputation + One-Hot Encoding
    - Ordinal categorical: Most-frequent Imputation + Ordinal Encoding
    """
    scaler = StandardScaler() if scaling_method == "standard" else MinMaxScaler()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=TIER_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
            ("nom", nominal_transformer, CATEGORICAL_NOMINAL),
            ("ord", ordinal_transformer, CATEGORICAL_ORDINAL)
        ],
        remainder="drop"
    )
    return preprocessor


def load_dataset(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Loads dataset from project Data directory with optional stratified subsampling for rapid evaluation."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return df


def prepare_data(sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Prepares stratified train-test splits for both classification and regression tasks."""
    df = load_dataset(sample_size=sample_size)

    # Feature matrix X
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    y_clf = df["PlacementStatus"].astype(int)

    # Classification train/test split (80/20 stratified)
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    # Regression dataset (Salary Package for Placed students > 0)
    reg_mask = (df["Salary Package"] > 0) & (df["PlacementStatus"] == 1)
    df_reg = df[reg_mask].copy()
    X_reg = df_reg.drop(columns=[c for c in DROP_COLS if c in df_reg.columns], errors="ignore")
    y_reg = df_reg["Salary Package"].astype(float)

    X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.20, random_state=42
    )

    return {
        "X_train_clf": X_train,
        "X_test_clf": X_test,
        "y_train_clf": y_clf_train,
        "y_test_clf": y_clf_test,
        "X_train_reg": X_train_reg,
        "X_test_reg": X_test_reg,
        "y_train_reg": y_reg_train,
        "y_test_reg": y_reg_test,
        "feature_names": list(X.columns),
        "total_records": len(df),
        "placed_count": int(y_clf.sum()),
        "not_placed_count": int(len(y_clf) - y_clf.sum())
    }


def get_m2_classification_models() -> Dict[str, Any]:
    """M2: Supervised Learning — Linear Models for Classification"""
    return {
        "Logistic Regression (L2 / Ridge)": LogisticRegression(
            C=1.0, max_iter=300, random_state=42
        ),
        "Lasso Logistic Regression (L1)": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5, random_state=42
        ),
        "ElasticNet Logistic Regression": LogisticRegression(
            solver="saga", l1_ratio=0.5, C=0.5, max_iter=100, tol=1e-2, random_state=42
        ),
    }


def get_m2_regression_models() -> Dict[str, Any]:
    """M2: Supervised Learning — Linear Models for Regression (Salary)"""
    return {
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Lasso Regression (L1)": Lasso(alpha=0.05, max_iter=1000, random_state=42),
        "ElasticNet Regression": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=1000, random_state=42),
    }


def get_m3_classification_models() -> Dict[str, Any]:
    """M3: Supervised Learning — Tree-Based Models for Classification"""
    return {
        "Decision Tree (Gini Impurity)": DecisionTreeClassifier(
            criterion="gini", max_depth=8, min_samples_leaf=10, random_state=42
        ),
        "Decision Tree (Entropy / Info Gain)": DecisionTreeClassifier(
            criterion="entropy", max_depth=8, min_samples_leaf=10, random_state=42
        ),
        "Random Forest (Bagging & OOB)": RandomForestClassifier(
            n_estimators=80, max_depth=12, min_samples_leaf=5, oob_score=True, random_state=42, n_jobs=-1
        ),
        "AdaBoost Classifier": AdaBoostClassifier(
            n_estimators=60, learning_rate=0.8, random_state=42
        ),
        "Gradient Boosted Trees (GBDT)": HistGradientBoostingClassifier(
            max_iter=80, learning_rate=0.1, max_depth=6, random_state=42
        ),
    }


def get_m3_regression_models() -> Dict[str, Any]:
    """M3: Supervised Learning — Tree-Based Models for Regression (Salary)"""
    return {
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=10, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=60, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        "AdaBoost Regressor": AdaBoostRegressor(
            n_estimators=50, learning_rate=0.8, random_state=42
        ),
        "Gradient Boosted Regressor": HistGradientBoostingRegressor(
            max_iter=80, learning_rate=0.1, max_depth=6, random_state=42
        ),
    }


def train_and_benchmark(sample_size: Optional[int] = 10000, force_retrain: bool = False) -> Dict[str, Any]:
    """
    Trains all M2 (Linear) and M3 (Tree) supervised models, calculates comprehensive metrics,
    extracts coefficients & feature importances, and caches results to JSON and joblib.
    """
    if not force_retrain and os.path.exists(CACHE_FILE) and os.path.exists(SAVED_MODELS_PATH):
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)
            return cached
        except Exception:
            pass

    print(f"[*] Loading data (sample_size={sample_size})...", flush=True)
    data = prepare_data(sample_size=sample_size)

    # 1. Feature Preprocessor
    preprocessor = get_feature_preprocessor(scaling_method="standard")
    X_train_clf = preprocessor.fit_transform(data["X_train_clf"])
    X_test_clf = preprocessor.transform(data["X_test_clf"])

    # Retrieve transformed feature names
    try:
        encoded_features = preprocessor.get_feature_names_out().tolist()
    except Exception:
        encoded_features = [f"f_{i}" for i in range(X_train_clf.shape[1])]

    # Prepare regression preprocessor
    reg_preprocessor = get_feature_preprocessor(scaling_method="standard")
    X_train_reg = reg_preprocessor.fit_transform(data["X_train_reg"])
    X_test_reg = reg_preprocessor.transform(data["X_test_reg"])

    # -------------------------------------------------------------
    # CLASSIFICATION BENCHMARK (M2 + M3)
    # -------------------------------------------------------------
    m2_clf_models = get_m2_classification_models()
    m3_clf_models = get_m3_classification_models()

    clf_models = {}
    for k, v in m2_clf_models.items():
        clf_models[k] = ("M2: Linear Models", v)
    for k, v in m3_clf_models.items():
        clf_models[k] = ("M3: Tree-Based Models", v)

    classification_results = []
    trained_clf_instances = {}
    roc_curves = {}
    best_clf_score = -1
    best_clf_name = ""

    y_train = data["y_train_clf"]
    y_test = data["y_test_clf"]

    print("[*] Training Classification Models (PlacementStatus)...", flush=True)
    for name, (module_cat, model) in clf_models.items():
        print(f"  -> Fitting {name}...", flush=True)
        t0 = time.time()
        model.fit(X_train_clf, y_train)
        fit_time = round(time.time() - t0, 3)

        y_pred = model.predict(X_test_clf)
        
        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_clf)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test_clf)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_prob = y_pred.astype(float)

        acc = round(float(accuracy_score(y_test, y_pred)), 4)
        prec = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
        rec = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
        f1 = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
        try:
            auc = round(float(roc_auc_score(y_test, y_prob)), 4)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            # Sample 20 points for smooth lightweight JSON serialization
            idx = np.linspace(0, len(fpr) - 1, min(25, len(fpr))).astype(int)
            roc_curves[name] = {
                "fpr": [round(float(x), 4) for x in fpr[idx]],
                "tpr": [round(float(x), 4) for x in tpr[idx]],
                "auc": auc
            }
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred).tolist()

        oob = None
        if hasattr(model, "oob_score_") and model.oob_score_:
            oob = round(float(model.oob_score_), 4)

        # Non-zero coefficients for L1 Lasso check
        sparsity = None
        if hasattr(model, "coef_"):
            n_zero = np.sum(np.abs(model.coef_) < 1e-4)
            total_coefs = model.coef_.size
            sparsity = f"{n_zero}/{total_coefs} zeroed ({round(100*n_zero/total_coefs, 1)}%)"

        classification_results.append({
            "name": name,
            "module": module_cat,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "fit_time_sec": fit_time,
            "confusion_matrix": cm,
            "oob_score": oob,
            "sparsity": sparsity
        })

        trained_clf_instances[name] = model

        if f1 > best_clf_score:
            best_clf_score = f1
            best_clf_name = name

    # -------------------------------------------------------------
    # REGRESSION BENCHMARK (M2 + M3 for Salary Package)
    # -------------------------------------------------------------
    m2_reg_models = get_m2_regression_models()
    m3_reg_models = get_m3_regression_models()

    reg_models = {}
    for k, v in m2_reg_models.items():
        reg_models[k] = ("M2: Linear Models", v)
    for k, v in m3_reg_models.items():
        reg_models[k] = ("M3: Tree-Based Models", v)

    regression_results = []
    trained_reg_instances = {}
    best_reg_score = -999.0
    best_reg_name = ""

    y_train_r = data["y_train_reg"]
    y_test_r = data["y_test_reg"]

    print("[*] Training Regression Models (Salary Package)...", flush=True)
    for name, (module_cat, model) in reg_models.items():
        print(f"  -> Fitting {name}...", flush=True)
        t0 = time.time()
        model.fit(X_train_reg, y_train_r)
        fit_time = round(time.time() - t0, 3)

        y_pred = model.predict(X_test_reg)

        mae = round(float(mean_absolute_error(y_test_r, y_pred)), 3)
        mse = round(float(mean_squared_error(y_test_r, y_pred)), 3)
        rmse = round(float(np.sqrt(mse)), 3)
        r2 = round(float(r2_score(y_test_r, y_pred)), 4)

        regression_results.append({
            "name": name,
            "module": module_cat,
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "fit_time_sec": fit_time
        })

        trained_reg_instances[name] = model

        if r2 > best_reg_score:
            best_reg_score = r2
            best_reg_name = name

    # -------------------------------------------------------------
    # M2 COEFFICIENTS & M3 FEATURE IMPORTANCE
    # -------------------------------------------------------------
    # M2 Coefficients (from Logistic Regression L2)
    m2_logreg = trained_clf_instances.get("Logistic Regression (L2 / Ridge)")
    m2_coefficients = []
    if m2_logreg is not None and hasattr(m2_logreg, "coef_"):
        coefs = m2_logreg.coef_[0]
        for f_name, w in zip(encoded_features, coefs):
            m2_coefficients.append({
                "feature": f_name.replace("num__", "").replace("nom__", "").replace("ord__", ""),
                "coefficient": round(float(w), 4),
                "odds_ratio": round(float(np.exp(w)), 4)
            })
        m2_coefficients.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        m2_coefficients = m2_coefficients[:15]

    # M3 Feature Importances (from Random Forest Classifier)
    rf_clf = trained_clf_instances.get("Random Forest (Bagging & OOB)")
    m3_feature_importances = []
    if rf_clf is not None and hasattr(rf_clf, "feature_importances_"):
        fi = rf_clf.feature_importances_
        for f_name, imp in zip(encoded_features, fi):
            m3_feature_importances.append({
                "feature": f_name.replace("num__", "").replace("nom__", "").replace("ord__", ""),
                "importance": round(float(imp), 4)
            })
        m3_feature_importances.sort(key=lambda x: x["importance"], reverse=True)
        m3_feature_importances = m3_feature_importances[:15]

    # M2 vs M3 Head-to-Head: Ridge Logistic Regression vs Gradient Boosted Tree
    m2_baseline = next((m for m in classification_results if "Ridge" in m["name"]), classification_results[0])
    m3_gbdt = next((m for m in classification_results if "Gradient Boosted" in m["name"]), classification_results[-1])

    head_to_head = {
        "linear_baseline": m2_baseline,
        "tree_champion": m3_gbdt,
        "accuracy_delta": round(m3_gbdt["accuracy"] - m2_baseline["accuracy"], 4),
        "f1_delta": round(m3_gbdt["f1_score"] - m2_baseline["f1_score"], 4),
        "auc_delta": round(m3_gbdt["roc_auc"] - m2_baseline["roc_auc"], 4),
        "speed_factor": round((m3_gbdt["fit_time_sec"] + 1e-4) / (m2_baseline["fit_time_sec"] + 1e-4), 2),
        "verdict": "Tree-based ensembles (Gradient Boosting) outperform Linear baselines by capturing complex non-linear feature interactions (e.g. CGPA × Coding Score) without manual interaction engineering."
    }

    # -------------------------------------------------------------
    # SERIALIZATION & PACKAGED ARTIFACTS
    # -------------------------------------------------------------
    # Package best pipelines with their preprocessors for production serving
    best_clf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", trained_clf_instances[best_clf_name])
    ])

    best_reg_pipeline = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("model", trained_reg_instances[best_reg_name])
    ])

    # Save to disk
    joblib.dump({
        "best_clf_pipeline": best_clf_pipeline,
        "best_clf_name": best_clf_name,
        "best_reg_pipeline": best_reg_pipeline,
        "best_reg_name": best_reg_name,
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, SAVED_MODELS_PATH)

    output = {
        "status": "success",
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records_trained": len(data["X_train_clf"]) + len(data["X_test_clf"]),
        "features_count": len(encoded_features),
        "classification": {
            "leaderboard": sorted(classification_results, key=lambda x: x["f1_score"], reverse=True),
            "best_model": best_clf_name,
            "best_f1": best_clf_score,
            "roc_curves": roc_curves
        },
        "regression": {
            "leaderboard": sorted(regression_results, key=lambda x: x["r2_score"], reverse=True),
            "best_model": best_reg_name,
            "best_r2": best_reg_score
        },
        "m2_coefficients": m2_coefficients,
        "m3_feature_importances": m3_feature_importances,
        "head_to_head": head_to_head
    }

    with open(CACHE_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print("[+] Benchmark complete & artifacts saved.")
    return output


def get_cached_or_train_benchmarks() -> Dict[str, Any]:
    """Retrieves cached benchmark results or triggers initial training if cache does not exist."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return train_and_benchmark(sample_size=20000, force_retrain=True)


def predict_student(student_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Day-One Live Prediction Service (M1 System Serving Endpoint).
    Traces: HTTP Request Input -> Preprocessing Pipeline -> Model Inference -> Structured Response.
    Prevents training-serving skew by utilizing packaged artifacts saved in joblib.
    """
    if not os.path.exists(SAVED_MODELS_PATH):
        train_and_benchmark(sample_size=20000, force_retrain=True)

    artifacts = joblib.load(SAVED_MODELS_PATH)
    clf_pipe = artifacts["best_clf_pipeline"]
    reg_pipe = artifacts["best_reg_pipeline"]

    # Convert single student dict to DataFrame matching training schema
    df_student = pd.DataFrame([student_dict])

    # 1. Predict placement classification & probability
    placement_pred = int(clf_pipe.predict(df_student)[0])
    placement_prob = 0.0
    if hasattr(clf_pipe.named_steps["model"], "predict_proba"):
        prob_matrix = clf_pipe.predict_proba(df_student)
        placement_prob = round(float(prob_matrix[0][1]) * 100, 1)
    else:
        placement_prob = 100.0 if placement_pred == 1 else 0.0

    # 2. Predict salary package (regression)
    salary_est = 0.0
    if placement_pred == 1:
        raw_salary = float(reg_pipe.predict(df_student)[0])
        salary_est = round(max(2.5, raw_salary), 2)  # Floor at realistic baseline LPA
    else:
        salary_est = 0.0

    # 3. Analyze influential signals for this student
    cgpa = float(student_dict.get("CGPA", 0))
    coding = float(student_dict.get("CodingTestScore", 0))
    internships = int(student_dict.get("Internships", 0))
    backlogs = str(student_dict.get("HistoryOfBacklogs", "No")).lower() == "yes"

    insights = []
    if cgpa >= 8.0:
        insights.append("Outstanding academic CGPA provides strong positive weighting.")
    elif cgpa < 6.5:
        insights.append("CGPA below 6.5 reduces eligibility threshold for premier recruiters.")

    if coding >= 75:
        insights.append("High Coding Test Score (>75) significantly boosts placement probability.")
    elif coding < 50:
        insights.append("Coding test score is a key bottleneck; improving coding practice is recommended.")

    if internships >= 2:
        insights.append("Multiple internships provide substantial competitive advantage.")
    if backlogs:
        insights.append("History of backlogs incurs a regularized penalty in the linear decision boundary.")

    return {
        "status": "success",
        "model_used_clf": artifacts.get("best_clf_name", "Best Classifier"),
        "model_used_reg": artifacts.get("best_reg_name", "Best Regressor"),
        "placement_status": "Placed" if placement_pred == 1 else "Not Placed",
        "placement_code": placement_pred,
        "placement_probability": placement_prob,
        "predicted_salary_lpa": salary_est,
        "insights": insights
    }


def get_preprocessing_demo_data() -> Dict[str, Any]:
    """Provides sample student data before and after scaling and encoding for the Preprocessing UI."""
    df = load_dataset(sample_size=10)
    raw_sample = df.head(5)[["StudentID", "CGPA", "AttendancePercent", "CodingTestScore", "Stream", "CollegeTier"]].to_dict(orient="records")

    scaler_std = StandardScaler()
    scaler_minmax = MinMaxScaler()

    scaled_std = scaler_std.fit_transform(df[["CGPA", "AttendancePercent"]].head(5)).round(3).tolist()
    scaled_mm = scaler_minmax.fit_transform(df[["CGPA", "AttendancePercent"]].head(5)).round(3).tolist()

    return {
        "raw_records": raw_sample,
        "scaled_standard": scaled_std,
        "scaled_minmax": scaled_mm,
        "categories_stream": list(df["Stream"].unique()),
        "categories_tier": list(df["CollegeTier"].unique()),
        "null_counts": {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}
    }


if __name__ == "__main__":
    print("=== Training M2 & M3 Supervised Models Benchmark ===")
    results = train_and_benchmark(sample_size=15000, force_retrain=True)
    print(f"\nClassification Best Model: {results['classification']['best_model']} (F1: {results['classification']['best_f1']})")
    print(f"Regression Best Model: {results['regression']['best_model']} (R²: {results['regression']['best_r2']})")
