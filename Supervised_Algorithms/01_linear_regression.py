"""
01_linear_regression.py
=======================
Supervised Learning: Ordinary Least Squares (OLS) Linear Regression
Target: Continuous Salary Package (in Lakhs Per Annum - LPA)

Mathematical Formulation:
-------------------------
y = beta_0 + beta_1 * x_1 + beta_2 * x_2 + ... + beta_p * x_p + epsilon

Objective Function (Sum of Squared Errors):
min_{beta} || y - X * beta ||_2^2 = sum_{i=1}^n (y_i - hat{y}_i)^2

Closed-form Normal Equation:
beta = (X^T * X)^(-1) * X^T * y

Key Assumptions:
1. Linearity: Relationship between predictors and target is linear.
2. Homoscedasticity: Constant variance of residuals.
3. Independence: Observations are independent of each other.
4. Normality of Residuals: Residuals are approximately normally distributed.
5. No Multicollinearity: Features should not be perfectly collinear.
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add current directory to path to allow importing data_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import prepare_regression_data, get_sample_student, print_section


def run_linear_regression(sample_size: int = 15000):
    print_section("01. Ordinary Least Squares (OLS) Linear Regression")
    print(f"[*] Training on Salary Package for Placed Candidates (sample_size={sample_size})...")

    # 1. Load and prepare scaled data
    data = prepare_regression_data(sample_size=sample_size, scaling="standard")
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_names = data["feature_names"]

    print(f"[*] Dataset Shape -> Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[*] Mean Salary in Dataset: {data['salary_stats']['mean']} LPA (Range: {data['salary_stats']['min']} - {data['salary_stats']['max']} LPA)")

    # 2. Instantiate and Fit Model
    print("\n[+] Fitting Linear Regression model...")
    t0 = time.time()
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    fit_duration = round(time.time() - t0, 4)
    print(f"[+] Model converged in {fit_duration} seconds.")

    # 3. Model Parameters
    intercept = model.intercept_
    coefficients = model.coef_
    print(f"\n[+] Intercept (beta_0): {intercept:.4f} LPA")

    # Top influential features
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient (LPA change per 1 std)": coefficients,
        "AbsImpact": np.abs(coefficients)
    }).sort_values(by="AbsImpact", ascending=False).reset_index(drop=True)

    print("\n--- TOP 10 INFLUENTIAL COEFFICIENTS ---")
    print(coef_df[["Feature", "Coefficient (LPA change per 1 std)"]].head(10).to_string(index=False))

    # 4. Predictions & Evaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    # Residual Analysis
    residuals = y_test - y_test_pred
    res_mean = residuals.mean()
    res_std = residuals.std()

    print("\n--- PERFORMANCE METRICS ---")
    print(f"  * Training R² Score : {train_r2:.4f}")
    print(f"  * Testing R² Score  : {test_r2:.4f}")
    print(f"  * MAE               : {mae:.4f} LPA")
    print(f"  * MSE               : {mse:.4f}")
    print(f"  * RMSE              : {rmse:.4f} LPA")
    print(f"  * Residual Mean     : {res_mean:.4f} (Ideal: ~0)")
    print(f"  * Residual Std Dev  : {res_std:.4f}")

    # 5. Live Prediction on Sample Candidate
    print("\n--- LIVE INFERENCE DEMO ---")
    sample_student = get_sample_student()
    print(f"Sample Student Profile: CGPA={sample_student['CGPA']}, Tier={sample_student['CollegeTier']}, Coding={sample_student['CodingTestScore']}")

    df_sample = pd.DataFrame([sample_student])
    X_sample_scaled = data["preprocessor"].transform(df_sample)
    predicted_salary = model.predict(X_sample_scaled)[0]

    print(f">>> Predicted Salary Package: {predicted_salary:.2f} LPA")

    return {
        "model": model,
        "metrics": {"r2": round(test_r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4)},
        "top_features": coef_df.head(5).to_dict(orient="records")
    }


if __name__ == "__main__":
    run_linear_regression()
