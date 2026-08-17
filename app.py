from flask import Flask, render_template, request, jsonify
import os

from load_data import (
    get_data_summary,
    get_duplicate_count,
    get_eda_summary,
    get_columns_metadata,
    get_plot_data
)

app = Flask(__name__, template_folder="Templates", static_folder="Static")


@app.route("/")
def index():
    return render_template(
        "dashboard.html",
        active="dashboard"
    )


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    duplicate_count = None

    try:
        summary = get_data_summary()
        duplicate_count = get_duplicate_count()
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "data_loading.html",
        active="data_loading",
        summary=summary,
        Duplicate_count=duplicate_count,
        error=error
    )


@app.route("/eda")
def eda():
    error = None
    eda_data = None
    columns_meta = None
    initial_plot = None

    try:
        eda_data = get_eda_summary()
        columns_meta = get_columns_metadata()
        initial_plot = get_plot_data(column="CGPA", chart_type="histogram")
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        eda=eda_data,
        columns_meta=columns_meta,
        initial_plot=initial_plot,
        error=error
    )


@app.route("/api/plot-data", methods=["GET", "POST"])
def api_plot_data():
    try:
        if request.method == "POST" and request.is_json:
            req_data = request.get_json()
            column = req_data.get("column", "CGPA")
            chart_type = req_data.get("chart_type", "histogram")
        else:
            column = request.args.get("column", "CGPA")
            chart_type = request.args.get("chart_type", "histogram")

        plot_data = get_plot_data(column=column, chart_type=chart_type)
        return jsonify({"success": True, "plot": plot_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)