"""
02_ridge_regression.py
======================
Supervised Learning: Ridge Regression (L2 Regularization)
Targets:
  1. Regression: Salary Package (LPA continuous) via Ridge
  2. Classification: PlacementStatus (Placed vs Not Placed) via RidgeClassifier

Mathematical Formulation:
-------------------------
Objective Function:
min_{beta} (1 / (2*n)) * || y - X * beta ||_2^2 + alpha * || beta ||_2^2
= min_{beta} (1 / (2*n)) * sum_{i=1}^n (y_i - hat{y}_i)^2 + alpha * sum_{j=1}^p beta_j^2

Closed-form Solution:
beta = (X^T * X + alpha * I)^(-1) * X^T * y

Key Characteristics:
- L2 penalty squares coefficients, shrinking them smoothly toward zero.
- Coefficients asymptotically approach zero but NEVER become exactly zero (no sparse feature selection).
- Stabilizes inversion when (X^T * X) is ill-conditioned due to high multicollinearity.
- Alpha controls bias-variance tradeoff:
    - High alpha -> High bias, low variance (underfitting)
    - Low alpha -> Low bias, high variance (closer to OLS)
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import prepare_regression_data, prepare_classification_data, get_sample_student, print_section


def run_ridge_experiments(sample_size: int = 15000):
    print_section("02. Ridge Regression & Classifier (L2 Regularization)")

    # ---------------------------------------------------------
    # PART 1: RIDGE REGRESSION FOR SALARY PREDICTION
    # ---------------------------------------------------------
    print("\n>>> PART 1: RIDGE REGRESSION (Salary Package LPA)")
    reg_data = prepare_regression_data(sample_size=sample_size, scaling="standard")
    X_train_r, X_test_r = reg_data["X_train"], reg_data["X_test"]
    y_train_r, y_test_r = reg_data["y_train"], reg_data["y_test"]
    feature_names = reg_data["feature_names"]

    # Explore Alpha Regularization Path
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
    alpha_results = []

    print(f"[*] Sweeping regularization hyperparameter alpha over {alphas}...")
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    for alpha in alphas:
        r_model = Ridge(alpha=alpha, random_state=42)
        r_model.fit(X_train_r, y_train_r)
        y_pred = r_model.predict(X_test_r)
        
        r2 = r2_score(y_test_r, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))
        l2_norm = np.linalg.norm(r_model.coef_)
        
        alpha_results.append({
            "Alpha": alpha,
            "Test R^2": round(r2, 4),
            "Test RMSE (LPA)": round(rmse, 4),
            "L2 Norm ||w||_2": round(l2_norm, 4)
        })

    alpha_df = pd.DataFrame(alpha_results)
    print("\n--- ALPHA REGULARIZATION PATH & COEFFICIENT SHRINKAGE ---")
    print(alpha_df.to_string(index=False))

    # Optimal Ridge Model
    best_alpha = 1.0
    ridge_reg = Ridge(alpha=best_alpha, random_state=42)
    ridge_reg.fit(X_train_r, y_train_r)
    y_test_pred_r = ridge_reg.predict(X_test_r)

    print(f"\n[+] Optimal Ridge Regressor (alpha={best_alpha}):")
    print(f"  * Test R^2 Score : {r2_score(y_test_r, y_test_pred_r):.4f}")
    print(f"  * Test MAE      : {mean_absolute_error(y_test_r, y_test_pred_r):.4f} LPA")
    print(f"  * Test RMSE     : {np.sqrt(mean_squared_error(y_test_r, y_test_pred_r)):.4f} LPA")

    # ---------------------------------------------------------
    # PART 2: RIDGE CLASSIFIER FOR PLACEMENT PREDICTION
    # ---------------------------------------------------------
    print("\n>>> PART 2: RIDGE CLASSIFIER (PlacementStatus)")
    clf_data = prepare_classification_data(sample_size=sample_size, scaling="standard")
    X_train_c, X_test_c = clf_data["X_train"], clf_data["X_test"]
    y_train_c, y_test_c = clf_data["y_train"], clf_data["y_test"]

    ridge_clf = RidgeClassifier(alpha=1.0, random_state=42)
    t0 = time.time()
    ridge_clf.fit(X_train_c, y_train_c)
    clf_time = round(time.time() - t0, 4)

    y_test_pred_c = ridge_clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_test_pred_c)
    f1 = f1_score(y_test_c, y_test_pred_c)

    print(f"[+] RidgeClassifier fit in {clf_time}s:")
    print(f"  * Accuracy : {acc * 100:.2f}%")
    print(f"  * F1 Score : {f1:.4f}")

    # ---------------------------------------------------------
    # PART 3: LIVE INFERENCE DEMO
    # ---------------------------------------------------------
    print("\n--- LIVE INFERENCE DEMO ---")
    sample_student = get_sample_student()
    df_sample = pd.DataFrame([sample_student])
    
    # Predict Classification
    X_sample_clf = clf_data["preprocessor"].transform(df_sample)
    clf_pred = ridge_clf.predict(X_sample_clf)[0]
    clf_status = "Placed" if clf_pred == 1 else "Not Placed"

    # Predict Salary
    X_sample_reg = reg_data["preprocessor"].transform(df_sample)
    reg_pred = ridge_reg.predict(X_sample_reg)[0]

    print(f"Sample Student: CGPA={sample_student['CGPA']}, Tier={sample_student['CollegeTier']}, CodingScore={sample_student['CodingTestScore']}")
    print(f">>> Predicted Placement Status : {clf_status}")
    print(f">>> Predicted Salary Package   : {reg_pred:.2f} LPA")


if __name__ == "__main__":
    run_ridge_experiments()
