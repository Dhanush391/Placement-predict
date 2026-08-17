import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def get_data_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },
        "preview": df.head(10).to_dict(orient="records"),
    }

    return summary


def get_duplicate_count(path: str = DATA_PATH) -> int:
    df = load_data(path)
    duplicate_count = df.duplicated().sum()
    return int(duplicate_count)


def get_eda_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)

    # Target distribution
    placement_counts = df["PlacementStatus"].value_counts().to_dict()
    total_records = len(df)
    placed_count = int(placement_counts.get(1, placement_counts.get("Placed", 0)))
    not_placed_count = int(placement_counts.get(0, placement_counts.get("Not Placed", 0)))
    placed_pct = round((placed_count / total_records) * 100, 1) if total_records else 0

    # Stream distribution
    stream_dist = df["Stream"].value_counts().head(6).to_dict()

    # Numerical statistics summary
    num_cols = ["CGPA", "AttendancePercent", "Internships", "Projects", "AptitudeTestScore", "CodingTestScore"]
    avail_num_cols = [c for c in num_cols if c in df.columns]
    desc_df = df[avail_num_cols].describe().round(2)
    stats_table = []
    for col in avail_num_cols:
        stats_table.append({
            "feature": col,
            "mean": desc_df.loc["mean", col],
            "std": desc_df.loc["std", col],
            "min": desc_df.loc["min", col],
            "median": desc_df.loc["50%", col],
            "max": desc_df.loc["max", col],
        })

    eda = {
        "total_records": total_records,
        "placed_count": placed_count,
        "not_placed_count": not_placed_count,
        "placed_pct": placed_pct,
        "not_placed_pct": round(100 - placed_pct, 1),
        "stream_dist": stream_dist,
        "stats_table": stats_table,
        "avg_cgpa": round(float(df["CGPA"].mean()), 2) if "CGPA" in df.columns else 0,
        "avg_coding": round(float(df["CodingTestScore"].mean()), 1) if "CodingTestScore" in df.columns else 0,
        "avg_aptitude": round(float(df["AptitudeTestScore"].mean()), 1) if "AptitudeTestScore" in df.columns else 0,
    }

    return eda


def get_columns_metadata(path: str = DATA_PATH) -> dict:
    df = load_data(path)
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    cat_cols = list(df.select_dtypes(exclude=["number"]).columns)
    all_cols = list(df.columns)
    return {
        "all": all_cols,
        "numerical": num_cols,
        "categorical": cat_cols
    }


def get_plot_data(column: str = "CGPA", chart_type: str = "histogram", secondary_col: str = None, path: str = DATA_PATH) -> dict:
    df = load_data(path)

    # Validate column
    if column not in df.columns and chart_type != "heatmap":
        column = "CGPA"

    is_num = column in df.select_dtypes(include=["number"]).columns

    result = {
        "column": column,
        "chart_type": chart_type,
        "is_numerical": bool(is_num),
        "data": {},
        "stats": {},
        "insight": ""
    }

    # 1. CORRELATION HEATMAP
    if chart_type == "heatmap":
        num_df = df.select_dtypes(include=["number"])
        priority_cols = [
            "CGPA", "SGPA_Sem7", "SGPA_Sem8", "AttendancePercent", 
            "Internships", "Projects", "Certifications", "AptitudeTestScore", 
            "CodingTestScore", "SoftSkillsRating", "MockInterviewScore", 
            "PlacementStatus", "Salary Package"
        ]
        active_cols = [c for c in priority_cols if c in num_df.columns]
        corr_matrix = num_df[active_cols].corr().round(2)
        
        result["data"] = {
            "columns": list(active_cols),
            "z": corr_matrix.values.tolist()
        }
        result["stats"] = {
            "Total Features": len(active_cols),
            "Strongest Correlation": f"CGPA & SGPA_Sem7 ({corr_matrix.loc['CGPA', 'SGPA_Sem7'] if 'SGPA_Sem7' in active_cols else 'N/A'})",
            "Target Correlation": f"CodingTestScore ({corr_matrix.loc['CodingTestScore', 'PlacementStatus'] if 'PlacementStatus' in active_cols else 'N/A'})"
        }
        result["insight"] = "Correlation heatmap illustrates linear dependencies among core academic scores, skill tests, and ultimate placement success."
        return result

    # 2. HISTOGRAM / DISTRIBUTION
    if chart_type == "histogram":
        if is_num:
            series = df[column].dropna()
            mean_val = float(series.mean())
            median_val = float(series.median())
            std_val = float(series.std())
            skew_val = float(series.skew())
            min_val = float(series.min())
            max_val = float(series.max())

            counts, bin_edges = np.histogram(series, bins=25)
            bin_labels = [f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}" for i in range(len(counts))]

            placed_vals = df[df["PlacementStatus"] == 1][column].dropna().sample(min(1000, len(df)), random_state=42).tolist() if "PlacementStatus" in df.columns else []
            not_placed_vals = df[df["PlacementStatus"] == 0][column].dropna().sample(min(1000, len(df)), random_state=42).tolist() if "PlacementStatus" in df.columns else []

            sample_size = min(2000, len(series))
            sample_vals = series.sample(sample_size, random_state=42).tolist()

            result["data"] = {
                "bin_labels": bin_labels,
                "counts": counts.tolist(),
                "sample_values": sample_vals,
                "placed_sample": placed_vals,
                "not_placed_sample": not_placed_vals,
            }
            result["stats"] = {
                "Mean": round(mean_val, 2),
                "Median": round(median_val, 2),
                "Std Dev": round(std_val, 2),
                "Skewness": round(skew_val, 2),
                "Min - Max": f"{round(min_val, 2)} - {round(max_val, 2)}",
                "Total Count": len(series)
            }
            result["insight"] = f"The distribution of {column} has an average of {round(mean_val, 2)} (Median: {round(median_val, 2)}) with skewness of {round(skew_val, 2)}."
        else:
            return get_plot_data(column, "bar", secondary_col, path)

    # 3. BAR GRAPH / COUNT PLOT
    elif chart_type == "bar":
        counts = df[column].value_counts().head(15)
        categories = [str(k) for k in counts.index]
        values = counts.values.tolist()
        total = len(df[column].dropna())

        result["data"] = {
            "categories": categories,
            "values": values,
            "percentages": [round((v / total) * 100, 1) for v in values]
        }
        result["stats"] = {
            "Top Category": f"{categories[0]} ({values[0]:,})",
            "Unique Classes": int(df[column].nunique()),
            "Total Entries": total
        }
        result["insight"] = f"Dominant category in {column} is '{categories[0]}' accounting for {round((values[0]/total)*100, 1)}% of all samples."

    # 4. BOX PLOT / QUARTILES
    elif chart_type == "box":
        if is_num:
            series = df[column].dropna()
            q1 = float(series.quantile(0.25))
            median = float(series.median())
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            outliers = int(((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum())

            placed_sample = df[df["PlacementStatus"] == 1][column].dropna().sample(min(1200, len(df)), random_state=42).tolist() if "PlacementStatus" in df.columns else []
            not_placed_sample = df[df["PlacementStatus"] == 0][column].dropna().sample(min(1200, len(df)), random_state=42).tolist() if "PlacementStatus" in df.columns else []

            result["data"] = {
                "placed_sample": placed_sample,
                "not_placed_sample": not_placed_sample,
                "all_sample": series.sample(min(1500, len(series)), random_state=42).tolist(),
            }
            result["stats"] = {
                "Q1 (25th %)": round(q1, 2),
                "Median (50th %)": round(median, 2),
                "Q3 (75th %)": round(q3, 2),
                "IQR Spread": round(iqr, 2),
                "Outliers Count": outliers
            }
            result["insight"] = f"The 50% middle spread for {column} spans between {round(q1, 2)} and {round(q3, 2)} with an IQR of {round(iqr, 2)}."
        else:
            return get_plot_data(column, "bar", secondary_col, path)

    # 5. SCATTER RELATIONSHIP PLOT
    elif chart_type == "scatter":
        target_y = "CGPA" if column != "CGPA" else "CodingTestScore"
        sample_df = df[[column, target_y, "PlacementStatus"]].dropna().sample(min(1000, len(df)), random_state=42)

        result["data"] = {
            "x": sample_df[column].tolist(),
            "y": sample_df[target_y].tolist(),
            "y_col": target_y,
            "placement": sample_df["PlacementStatus"].tolist() if "PlacementStatus" in sample_df.columns else []
        }
        corr_val = round(float(df[column].corr(df[target_y])), 2) if is_num and target_y in df.columns else 0.0
        result["stats"] = {
            "X-Axis": column,
            "Y-Axis": target_y,
            "Correlation (r)": corr_val,
            "Sampled Points": len(sample_df)
        }
        result["insight"] = f"Correlation between {column} and {target_y} is r = {corr_val}."

    # 6. VIOLIN / PROBABILITY DENSITY
    elif chart_type == "violin":
        if is_num:
            placed_sample = df[df["PlacementStatus"] == 1][column].dropna().sample(min(1200, len(df)), random_state=42).tolist() if "PlacementStatus" in df.columns else []
            not_placed_sample = df[df["PlacementStatus"] == 0][column].dropna().sample(min(1200, len(df)), random_state=42).tolist() if "PlacementStatus" in df.columns else []
            result["data"] = {
                "placed_sample": placed_sample,
                "not_placed_sample": not_placed_sample,
            }
            result["stats"] = {
                "Feature": column,
                "Group 1 (Placed) Mean": round(float(df[df["PlacementStatus"] == 1][column].mean()), 2) if "PlacementStatus" in df.columns else "N/A",
                "Group 0 (Not Placed) Mean": round(float(df[df["PlacementStatus"] == 0][column].mean()), 2) if "PlacementStatus" in df.columns else "N/A",
            }
            result["insight"] = f"Violin density displays multimodal distribution and placement likelihood differences across {column}."
        else:
            return get_plot_data(column, "bar", secondary_col, path)

    # 7. CUMULATIVE DISTRIBUTION (CDF)
    elif chart_type == "line":
        if is_num:
            series = df[column].dropna().sort_values()
            n = len(series)
            step = max(1, n // 200)
            sub_series = series.iloc[::step]
            cdf_y = [round((i / n) * 100, 2) for i in range(0, n, step)]
            result["data"] = {
                "x": sub_series.tolist(),
                "y": cdf_y
            }
            result["stats"] = {
                "50th Percentile": round(float(series.quantile(0.50)), 2),
                "90th Percentile": round(float(series.quantile(0.90)), 2),
                "Total Samples": n
            }
            result["insight"] = f"Cumulative distribution demonstrates percentile thresholds for {column}."
        else:
            return get_plot_data(column, "bar", secondary_col, path)

    return result


if __name__ == "__main__":
    data = load_data()
    print("Summary:", get_data_summary())
    print("Duplicates:", get_duplicate_count())
    print("Columns Metadata:", get_columns_metadata())