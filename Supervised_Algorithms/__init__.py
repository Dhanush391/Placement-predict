"""
Supervised Machine Learning Algorithms Package
==============================================
Collection of individual, standalone implementations for supervised learning algorithms
applied to the Placement Prediction & Salary Estimation benchmark.
"""

from .data_utils import (
    load_dataset,
    prepare_classification_data,
    prepare_regression_data,
    get_sample_student
)

__all__ = [
    "load_dataset",
    "prepare_classification_data",
    "prepare_regression_data",
    "get_sample_student"
]
