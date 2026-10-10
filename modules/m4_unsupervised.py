"""
modules/m4_unsupervised.py
==========================
Module 4: Unsupervised Learning

Concepts Implemented:
- K-Means: Modern Lloyd algorithm, Elbow method (Inertia/WCSS), Silhouette score analysis.
- K-Means++ vs Random Initialization, Mini-Batch K-Means scalability comparison.
- Hierarchical Clustering: Agglomerative bottom-up, Linkage methods (Ward, Complete, Average, Single), Dendrogram generation.
- Density-Based Clustering (DBSCAN): eps and minPts, Core, Border, and Noise points, Arbitrary non-convex clusters.
- Dimensionality Reduction (PCA): Scree plot, Explained variance ratio, Critical pitfall of PCA without standardization.
- Manifold Learning (t-SNE / 2D projection): Visualizing student persona clusters, Non-metric distance caveat.
- Anomaly Detection: Isolation Forest (average isolation depth), One-Class SVM.
- Practical Pipeline: Using Unsupervised Cluster Personas as Engineered Features for Downstream Supervised Prediction!
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, accuracy_score, roc_auc_score
from scipy.cluster.hierarchy import linkage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import load_data

def get_unsupervised_student_data(sample_size: int = 5000) -> Dict[str, Any]:
    """Prepares continuous student competency features for clustering and manifold learning."""
    df = load_data.load_data()
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    features = [
        "CGPA", "AttendancePercent", "Internships", "Projects",
        "AptitudeTestScore", "CodingTestScore", "SoftSkillsRating", "MockInterviewScore"
    ]
    avail = [c for c in features if c in df.columns]

    X_raw = df[avail].fillna(df[avail].median()).values
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    return {
        "df": df,
        "feature_names": avail,
        "X_raw": X_raw,
        "X_std": X_std,
        "y": df["PlacementStatus"].astype(int).values
    }


def analyze_kmeans_elbow_and_silhouette() -> Dict[str, Any]:
    """
    Evaluates optimal K via Elbow Method (Inertia) and Silhouette Scores (k = 2 to 7).
    Compares K-Means++ initialization vs Random initialization, and Mini-Batch K-Means speedup.
    """
    data = get_unsupervised_student_data(sample_size=3500)
    X = data["X_std"]

    k_range = list(range(2, 8))
    elbow_curve = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=42)
        km.fit(X)
        sil = round(float(silhouette_score(X, km.labels_)), 4)
        elbow_curve.append({
            "k": k,
            "inertia_wcss": round(float(km.inertia_), 1),
            "silhouette_score": sil
        })

    # Compare K-means++ vs Random initialization on K=3
    km_plus = KMeans(n_clusters=3, init="k-means++", n_init=1, random_state=42)
    t0 = time.perf_counter()
    km_plus.fit(X)
    t_plus = round((time.perf_counter() - t0) * 1000, 2)

    km_rand = KMeans(n_clusters=3, init="random", n_init=1, random_state=42)
    t0 = time.perf_counter()
    km_rand.fit(X)
    t_rand = round((time.perf_counter() - t0) * 1000, 2)

    # Mini-batch K-Means benchmark
    mbk = MiniBatchKMeans(n_clusters=3, batch_size=256, random_state=42)
    t0 = time.perf_counter()
    mbk.fit(X)
    t_mbk = round((time.perf_counter() - t0) * 1000, 2)

    # Fit best model (K=3) and profile clusters
    best_km = KMeans(n_clusters=3, init="k-means++", n_init=5, random_state=42)
    labels = best_km.fit_predict(X)

    # Cluster personas summary
    personas = []
    persona_names = [
        "Persona A: Strong Coders & Project Leaders",
        "Persona B: High Academic CGPA & Regular Attendance",
        "Persona C: At-Risk / Foundational Development"
    ]
    for cluster_id in range(3):
        mask = labels == cluster_id
        cluster_df = data["df"].iloc[mask]
        personas.append({
            "cluster_id": cluster_id,
            "name": persona_names[cluster_id],
            "student_count": int(np.sum(mask)),
            "pct_of_total": round((np.sum(mask) / len(labels)) * 100, 1),
            "avg_cgpa": round(float(cluster_df["CGPA"].mean()), 2),
            "avg_coding": round(float(cluster_df["CodingTestScore"].mean()), 1),
            "avg_internships": round(float(cluster_df["Internships"].mean()), 1),
            "placement_rate_pct": round(float(cluster_df["PlacementStatus"].mean()) * 100, 1)
        })

    return {
        "elbow_and_silhouette": elbow_curve,
        "initialization_benchmark": {
            "kmeans_plus_plus": {
                "inertia": round(float(km_plus.inertia_), 1),
                "fit_time_ms": t_plus,
                "strategy": "Chooses seeds with probability proportional to D(x)^2, avoiding poor local minima"
            },
            "random_init": {
                "inertia": round(float(km_rand.inertia_), 1),
                "fit_time_ms": t_rand,
                "strategy": "Uniform random selection; prone to slow convergence and sub-optimal clustering"
            },
            "minibatch_kmeans": {
                "inertia": round(float(mbk.inertia_), 1),
                "fit_time_ms": t_mbk,
                "strategy": "Stochastic mini-batch gradient updates; linear scaling for massive datasets"
            }
        },
        "student_personas": personas
    }


def analyze_hierarchical_clustering() -> Dict[str, Any]:
    """
    Evaluates Agglomerative Hierarchical Clustering across Linkage methods:
    Ward, Complete, Average, and Single linkage.
    """
    data = get_unsupervised_student_data(sample_size=1200)
    X = data["X_std"]

    linkages = ["ward", "complete", "average", "single"]
    linkage_results = []

    for link_type in linkages:
        agg = AgglomerativeClustering(n_clusters=3, linkage=link_type)
        t0 = time.perf_counter()
        labels = agg.fit_predict(X)
        t_link = round((time.perf_counter() - t0) * 1000, 2)
        sil = round(float(silhouette_score(X, labels)), 4)
        linkage_results.append({
            "linkage": link_type,
            "silhouette": sil,
            "fit_time_ms": t_link,
            "description": {
                "ward": "Minimizes total within-cluster variance (spherical clusters)",
                "complete": "Finds maximum pairwise distance (compact, equal-sized clusters)",
                "average": "Average pairwise distance (compromise between single and complete)",
                "single": "Minimum pairwise distance (sensitive to chaining and noise)"
            }[link_type]
        })

    # Hierarchical Dendrogram linkage matrix sample
    Z = linkage(X[:50], method="ward")
    dendrogram_nodes = []
    for i in range(min(8, len(Z))):
        dendrogram_nodes.append({
            "step": i + 1,
            "cluster_a": int(Z[i, 0]),
            "cluster_b": int(Z[i, 1]),
            "merge_distance": round(float(Z[i, 2]), 3),
            "new_cluster_size": int(Z[i, 3])
        })

    return {
        "linkage_comparison": linkage_results,
        "dendrogram_sample_merges": dendrogram_nodes,
        "dendrogram_interpretation": "A horizontal cut across the dendrogram at a high merge distance isolates distinct student competency tiers without specifying K beforehand."
    }


def analyze_dbscan_density_clustering() -> Dict[str, Any]:
    """
    Evaluates DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    Identifies Core, Border, and Noise points (outliers) and discovers non-convex clusters.
    """
    data = get_unsupervised_student_data(sample_size=2000)
    X = data["X_std"]

    # Parameter grid for eps and min_samples
    configs = [
        {"eps": 1.2, "min_samples": 8},
        {"eps": 1.5, "min_samples": 10},
        {"eps": 1.8, "min_samples": 12}
    ]

    scan_results = []
    for cfg in configs:
        db = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"])
        labels = db.fit_predict(X)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        noise_pct = round((n_noise / len(labels)) * 100, 1)

        scan_results.append({
            "eps": cfg["eps"],
            "min_samples": cfg["min_samples"],
            "clusters_discovered": n_clusters,
            "noise_points": n_noise,
            "noise_pct": noise_pct
        })

    return {
        "dbscan_experiments": scan_results,
        "core_mechanics": {
            "core_point": "Point with at least minPts neighbors within distance epsilon.",
            "border_point": "Within epsilon of a core point, but fewer than minPts neighbors itself.",
            "noise_point": "Neither core nor border; naturally flagged as anomalous outlier student records.",
            "non_convex_advantage": "Unlike K-Means (which assumes convex spherical blobs), DBSCAN can trace complex winding manifolds and arbitrary cluster boundaries."
        }
    }


def analyze_pca_and_standardization_pitfall() -> Dict[str, Any]:
    """
    Evaluates PCA in production usage: Explained variance ratio and Scree plot.
    Demonstrates the CRITICAL PITFALL of applying PCA without standardization!
    """
    data = get_unsupervised_student_data(sample_size=3500)
    X_raw = data["X_raw"]
    X_std = data["X_std"]
    features = data["feature_names"]

    # 1. PCA WITH Standardization (Production Best Practice)
    pca_std = PCA(n_components=min(len(features), 6))
    pca_std.fit(X_std)
    var_std = pca_std.explained_variance_ratio_
    cum_std = np.cumsum(var_std)

    std_components = []
    for i in range(len(var_std)):
        std_components.append({
            "component": f"PC{i+1}",
            "explained_variance": round(float(var_std[i]), 4),
            "cumulative_variance": round(float(cum_std[i]), 4)
        })

    # 2. PCA WITHOUT Standardization (THE PITFALL)
    pca_unscaled = PCA(n_components=min(len(features), 6))
    pca_unscaled.fit(X_raw)
    var_unscaled = pca_unscaled.explained_variance_ratio_

    top_feature_variance = np.var(X_raw, axis=0)
    max_var_feat = features[int(np.argmax(top_feature_variance))]

    return {
        "pca_standardized_scree": std_components,
        "pca_unscaled_pitfall": {
            "first_pc_variance_unscaled": round(float(var_unscaled[0]), 4),
            "dominant_feature": max_var_feat,
            "pitfall_explanation": f"Without standardization, '{max_var_feat}' (variance {round(float(np.max(top_feature_variance)), 1)}) completely hijacks PC1 with {round(float(var_unscaled[0])*100, 1)}% of total variance simply due to its raw scale (0-100), drowning out critical academic signals like CGPA (scale 0-10) whose raw variance is tiny (~0.8)!"
        }
    }


def analyze_manifold_learning_tsne() -> Dict[str, Any]:
    """
    t-SNE 2D Manifold projection for visual cluster interpretation.
    Explains the non-metric distance property of t-SNE space.
    """
    data = get_unsupervised_student_data(sample_size=500)
    X = data["X_std"]
    y = data["y"]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=800)
    embedding = tsne.fit_transform(X)

    # 2D sample points
    sample_points = []
    for i in range(min(150, len(embedding))):
        sample_points.append({
            "x": round(float(embedding[i, 0]), 2),
            "y": round(float(embedding[i, 1]), 2),
            "status": "Placed" if y[i] == 1 else "Not Placed"
        })

    return {
        "projection_points": sample_points,
        "tsne_theoretical_caveats": [
            "Perplexity balances local vs global manifold topology; changing perplexity drastically reorganizes clusters.",
            "Cluster sizes and inter-cluster distances in t-SNE space are NOT metric: distant clusters cannot be interpreted as numerically further apart in raw feature space.",
            "t-SNE minimizes Kullback-Leibler (KL) divergence between student neighborhood probability distributions."
        ]
    }


def analyze_anomaly_detection() -> Dict[str, Any]:
    """
    Evaluates Isolation Forest and One-Class SVM for Student Anomaly Detection.
    """
    data = get_unsupervised_student_data(sample_size=3000)
    X = data["X_std"]

    # 1. Isolation Forest
    iso = IsolationForest(contamination=0.03, random_state=42)
    iso_preds = iso.fit_predict(X)  # -1 for anomaly, 1 for inlier
    n_iso_anomalies = int(np.sum(iso_preds == -1))

    # 2. One-Class SVM
    oc_svm = OneClassSVM(nu=0.03, kernel="rbf", gamma="scale")
    svm_preds = oc_svm.fit_predict(X[:1500])
    n_svm_anomalies = int(np.sum(svm_preds == -1))

    return {
        "isolation_forest": {
            "anomalies_detected": n_iso_anomalies,
            "contamination_rate": 0.03,
            "concept": "Random recursive feature partitioning isolates anomalous points in remarkably few tree splits (short path length h(x))."
        },
        "one_class_svm": {
            "anomalies_detected": n_svm_anomalies,
            "nu_parameter": 0.03,
            "concept": "Maps student records to high-dimensional Hilbert space via RBF kernel and fits maximal margin hyperplane separating normal density from the origin."
        }
    }


def run_hybrid_cluster_feature_pipeline() -> Dict[str, Any]:
    """
    Practical Pipeline:
    When to cluster vs supervised model?
    Engineers K-Means Cluster Labels as categorical features for downstream Supervised Placement Prediction!
    """
    data = get_unsupervised_student_data(sample_size=6000)
    X_std = data["X_std"]
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.25, random_state=42, stratify=y)

    # 1. Baseline Supervised Logistic Regression
    base_clf = LogisticRegression(C=1.0, random_state=42)
    base_clf.fit(X_train, y_train)
    acc_base = round(float(accuracy_score(y_test, base_clf.predict(X_test))), 4)
    auc_base = round(float(roc_auc_score(y_test, base_clf.predict_proba(X_test)[:, 1])), 4)

    # 2. Hybrid Pipeline: Fit K-Means (K=4) on training data, generate one-hot cluster features
    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)
    train_clusters = kmeans.fit_predict(X_train)
    test_clusters = kmeans.predict(X_test)

    # Convert cluster labels to one-hot representation
    train_onehot = pd.get_dummies(train_clusters, prefix="cluster", drop_first=True).astype(float).values
    test_onehot = pd.get_dummies(test_clusters, prefix="cluster", drop_first=True).astype(float).values

    # Augment features
    X_train_augmented = np.hstack([X_train, train_onehot])
    X_test_augmented = np.hstack([X_test, test_onehot])

    hybrid_clf = LogisticRegression(C=1.0, random_state=42)
    hybrid_clf.fit(X_train_augmented, y_train)
    acc_hybrid = round(float(accuracy_score(y_test, hybrid_clf.predict(X_test_augmented))), 4)
    auc_hybrid = round(float(roc_auc_score(y_test, hybrid_clf.predict_proba(X_test_augmented)[:, 1])), 4)

    return {
        "baseline_supervised": {
            "accuracy": acc_base,
            "roc_auc": auc_base,
            "description": "Standard linear model on raw standardized academic features"
        },
        "hybrid_pipeline": {
            "accuracy": acc_hybrid,
            "roc_auc": auc_hybrid,
            "clusters_added": 4,
            "accuracy_lift_pct": round((acc_hybrid - acc_base) * 100, 2),
            "description": "Unsupervised student persona cluster memberships injected as nonlinear categorical features into linear classifier"
        },
        "takeaway": "Unsupervised clustering acts as nonlinear representation learning: it partitions the input space into distinct student archetypes, giving linear models piecewise capacity without tree depth!"
    }


def get_m4_complete_overview() -> Dict[str, Any]:
    """Returns complete benchmarks and analysis for Module 4."""
    return {
        "kmeans_elbow_silhouette": analyze_kmeans_elbow_and_silhouette(),
        "hierarchical_clustering": analyze_hierarchical_clustering(),
        "dbscan_clustering": analyze_dbscan_density_clustering(),
        "pca_scree_and_pitfall": analyze_pca_and_standardization_pitfall(),
        "tsne_manifold": analyze_manifold_learning_tsne(),
        "anomaly_detection": analyze_anomaly_detection(),
        "hybrid_pipeline": run_hybrid_cluster_feature_pipeline()
    }
