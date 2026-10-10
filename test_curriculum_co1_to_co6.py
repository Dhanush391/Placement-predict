"""
test_curriculum_co1_to_co6.py
=============================
Comprehensive Test Suite for Curriculum Modules M1 through M6 (CO1 to CO6).
Verifies that all algorithms, pipelines, metrics, and serving endpoints operate accurately.
"""

import sys
import unittest
import numpy as np

# Import all modules
import modules.m1_lifecycle as m1
import modules.m2_linear_models as m2
import modules.m3_tree_models as m3
import modules.m4_unsupervised as m4
import modules.m5_evaluation_calibration as m5
import modules.m6_ml_engineering as m6
from app import app


class TestCurriculumCO1ToCO6(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.flask_client = app.test_client()

    # --- M1 (CO1): The ML Lifecycle & System ---
    def test_m1_system_lifecycle(self):
        overview = m1.get_m1_system_overview()
        self.assertIn("lifecycle_stages", overview)
        self.assertEqual(len(overview["lifecycle_stages"]), 8)
        self.assertIn("taxonomy", overview)
        self.assertIn("supervised", overview["taxonomy"])
        self.assertIn("unsupervised", overview["taxonomy"])
        self.assertIn("reinforcement", overview["taxonomy"])

        # Trace sample prediction
        trace = overview["sample_trace"]
        self.assertIn("prediction", trace)
        self.assertIn("breakdown", trace)
        self.assertGreater(trace["total_latency_ms"], 0)

    # --- M2 (CO2): Linear Models at Depth ---
    def test_m2_linear_models(self):
        m2_res = m2.get_m2_complete_overview()
        # 1. Normal Equations vs GD
        self.assertIn("normal_eq_vs_gd", m2_res)
        norm_eq = m2_res["normal_eq_vs_gd"]["normal_equations"]
        self.assertGreater(norm_eq["r2_score"], 0)
        # 2. Regularization & Sparsity
        reg = m2_res["regularization_sparsity"]
        self.assertGreater(reg["models"]["lasso"]["sparsity_pct"], 0)
        # 3. Multinomial Softmax
        soft = m2_res["multinomial_softmax"]
        self.assertGreater(soft["accuracy"], 0.70)
        self.assertEqual(len(soft["classes"]), 4)
        # 4. Scaling Invariance on Trees vs Logistic
        scale = m2_res["scaling_invariance"]
        self.assertLess(scale["decision_tree_invariance"]["accuracy_difference"], 0.01)

    # --- M3 (CO3): Tree-Based Models ---
    def test_m3_tree_models(self):
        m3_res = m3.get_m3_complete_overview()
        # 1. Decision Tree & Pruning
        self.assertIn("dt_and_pruning", m3_res)
        self.assertGreater(len(m3_res["dt_and_pruning"]["cost_complexity_pruning"]["alpha_curve"]), 0)
        # 2. Random Forest & OOB Error
        self.assertIn("rf_and_variance", m3_res)
        # 3. XGBoost vs LightGBM
        self.assertIn("xgboost_vs_lightgbm", m3_res)
        self.assertGreater(m3_res["xgboost_vs_lightgbm"]["lightgbm"]["accuracy"], 0.75)
        # 4. Feature Importance Triad
        self.assertIn("feature_importance", m3_res)
        self.assertGreater(len(m3_res["feature_importance"]["triad_table"]), 0)

    # --- M4 (CO4): Unsupervised Learning ---
    def test_m4_unsupervised(self):
        m4_res = m4.get_m4_complete_overview()
        # 1. K-Means Elbow & Silhouette
        self.assertIn("kmeans_elbow_silhouette", m4_res)
        self.assertEqual(len(m4_res["kmeans_elbow_silhouette"]["elbow_and_silhouette"]), 6)
        # 2. Hierarchical Clustering
        self.assertIn("hierarchical_clustering", m4_res)
        # 3. DBSCAN
        self.assertIn("dbscan_clustering", m4_res)
        # 4. PCA Pitfall
        self.assertIn("pca_scree_and_pitfall", m4_res)
        self.assertGreater(m4_res["pca_scree_and_pitfall"]["pca_unscaled_pitfall"]["first_pc_variance_unscaled"], 0.3)
        # 5. Hybrid Pipeline Lift
        self.assertIn("hybrid_pipeline", m4_res)
        self.assertGreater(m4_res["hybrid_pipeline"]["hybrid_pipeline"]["accuracy"], 0.70)

    # --- M5 (CO5): Model Evaluation & Calibration ---
    def test_m5_evaluation_calibration(self):
        m5_res = m5.get_m5_complete_overview()
        # 1. 3-Way Split & Nested CV
        self.assertIn("split_and_nested_cv", m5_res)
        self.assertGreater(m5_res["split_and_nested_cv"]["nested_cross_validation"]["unbiased_nested_mean_accuracy"], 0.70)
        # 2. Classification Metrics
        self.assertIn("classification_metrics", m5_res)
        self.assertGreater(m5_res["classification_metrics"]["metrics"]["f1_score"], 0.70)
        # 3. Probability Calibration
        self.assertIn("probability_calibration", m5_res)
        ece = m5_res["probability_calibration"]["expected_calibration_error_ece"]
        self.assertGreater(ece["uncalibrated_ece"], 0)
        # 4. McNemar's Test
        self.assertIn("mcnemar_significance", m5_res)

    # --- M6 (CO6): ML Engineering & Production ---
    def test_m6_ml_engineering(self):
        m6_res = m6.get_m6_complete_overview()
        # 1. Feature Store Simulator
        self.assertIn("feature_store", m6_res)
        # 2. Training-Serving Skew Detector
        skew = m6_res["training_serving_skew"]
        self.assertTrue(skew["clean_payload_audit"]["is_clean"])
        self.assertFalse(skew["skewed_payload_audit"]["is_clean"])
        # 3. Production Monitoring & Drift
        self.assertIn("production_monitoring", m6_res)
        self.assertIn("latency_metrics_ms", m6_res["production_monitoring"])
        self.assertIn("data_drift", m6_res["production_monitoring"])

    # --- Flask Routes & REST Endpoints Integration ---
    def test_all_flask_routes_and_api(self):
        routes = [
            "/", "/ml-lifecycle", "/m2-linear-models", "/m3-tree-models",
            "/m4-unsupervised", "/m5-evaluation", "/m6-engineering",
            "/predict", "/data-loading", "/eda", "/preprocessing",
            "/models", "/linear-regression", "/ridge-regression",
            "/lasso-regression", "/elasticnet-regression", "/logistic-regression",
            "/decision-tree", "/random-forest", "/xgboost", "/lightgbm"
        ]
        for route in routes:
            response = self.flask_client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with {response.status_code}")

        # Test prediction API
        student = {
            "student_id": "TEST-100",
            "CGPA": 8.7,
            "AttendancePercent": 92.0,
            "Internships": 2,
            "Projects": 3,
            "CodingTestScore": 89.0,
            "AptitudeTestScore": 84.0
        }
        res_trace = self.flask_client.post("/api/v1/trace-prediction", json=student)
        self.assertEqual(res_trace.status_code, 200)

        # Test skew audit API
        res_audit = self.flask_client.post("/api/v1/skew-audit", json=student)
        self.assertEqual(res_audit.status_code, 200)

        # Test online feature store API
        res_feats = self.flask_client.get("/api/v1/online-features/1005")
        self.assertEqual(res_feats.status_code, 200)


if __name__ == "__main__":
    unittest.main()
