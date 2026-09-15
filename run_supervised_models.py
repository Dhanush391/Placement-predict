"""
run_supervised_models.py
========================
Standalone CLI Benchmark Runner for M2 & M3 Supervised Algorithms in PlacementPredict.
Displays clear-cut performance tables for M2 Linear Models and M3 Tree-Based Models.
"""

import sys
import pandas as pd
from supervised_models import train_and_benchmark, get_cached_or_train_benchmarks

def print_banner():
    print("=" * 85)
    print("       PLACEMENTPREDICT: SUPERVISED MACHINE LEARNING BENCHMARK SUITE       ")
    print("   [M2: Linear Models at Depth] & [M3: Tree-Based Models & Ensembles]     ")
    print("=" * 85)

def main():
    print_banner()
    force = "--retrain" in sys.argv
    print(f"Loading benchmarks (retrain={force})...\n")
    results = train_and_benchmark(sample_size=15000, force_retrain=force)

    # 1. Classification Leaderboard
    print("\n" + "-" * 85)
    print(" 1. CLASSIFICATION LEADERBOARD (Target: PlacementStatus - Placed vs Not Placed)")
    print("-" * 85)
    clf_df = pd.DataFrame(results["classification"]["leaderboard"])
    cols_clf = ["name", "module", "accuracy", "precision", "recall", "f1_score", "roc_auc", "fit_time_sec"]
    display_clf = clf_df[cols_clf].copy()
    display_clf.columns = ["Algorithm", "Curriculum Module", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Time (s)"]
    print(display_clf.to_string(index=False))
    print(f"\n>>> Best Classifier: {results['classification']['best_model']} (F1: {results['classification']['best_f1']})")

    # 2. Regression Leaderboard
    print("\n" + "-" * 85)
    print(" 2. REGRESSION LEADERBOARD (Target: Salary Package - LPA)")
    print("-" * 85)
    reg_df = pd.DataFrame(results["regression"]["leaderboard"])
    cols_reg = ["name", "module", "mae", "rmse", "r2_score", "fit_time_sec"]
    display_reg = reg_df[cols_reg].copy()
    display_reg.columns = ["Algorithm", "Curriculum Module", "MAE (LPA)", "RMSE (LPA)", "R2 Score", "Time (s)"]
    print(display_reg.to_string(index=False))
    print(f"\n>>> Best Regressor: {results['regression']['best_model']} (R2: {results['regression']['best_r2']})")

    # 3. M2 vs M3 Head-to-Head Comparison
    h2h = results["head_to_head"]
    print("\n" + "-" * 85)
    print(" 3. M2 vs M3 HEAD-TO-HEAD: Linear Baseline vs Gradient Boosted Tree")
    print("-" * 85)
    print(f"Linear Baseline : {h2h['linear_baseline']['name']} -> F1: {h2h['linear_baseline']['f1_score']}, AUC: {h2h['linear_baseline']['roc_auc']}")
    print(f"Tree Champion   : {h2h['tree_champion']['name']} -> F1: {h2h['tree_champion']['f1_score']}, AUC: {h2h['tree_champion']['roc_auc']}")
    print(f"F1 Delta        : {h2h['f1_delta']:+0.4f}")
    print(f"ROC-AUC Delta   : {h2h['auc_delta']:+0.4f}")
    print(f"Speed Ratio     : {h2h['speed_factor']}x")
    print(f"Verdict         : {h2h['verdict']}")

    # 4. Feature Importances & Coefficients
    print("\n" + "-" * 85)
    print(" 4. TOP PLACEMENT DETERMINANTS (M3 Random Forest Feature Importance)")
    print("-" * 85)
    for idx, item in enumerate(results["m3_feature_importances"][:8], 1):
        bar = "#" * int(item["importance"] * 50)
        print(f" {idx}. {item['feature']:<25} | Imp: {item['importance']:.4f} {bar}")

    print("\n" + "=" * 85)
    print(" Benchmark execution completed successfully. UI Dashboard updated.")
    print("=" * 85)

if __name__ == "__main__":
    main()
