"""
Models Package for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Pre-built production-ready ML models across multiple frameworks.
"""

from .sklearn_models import (
    train_random_forest,
    train_logistic_regression,
    train_svm,
    train_gradient_boosting
)

from .xgboost_models import train_xgboost_classifier
from .lightgbm_models import train_lightgbm_classifier
from .keras_models import train_shallow_nn, train_deep_nn
from .model_configs import (
    RANDOM_FOREST_CONFIGS,
    LOGISTIC_REGRESSION_CONFIGS,
    SVM_CONFIGS,
    GRADIENT_BOOSTING_CONFIGS,
    XGBOOST_CONFIGS,
    LIGHTGBM_CONFIGS,
    KERAS_NN_CONFIGS,
    get_config,
    list_configs
)

__all__ = [
    'train_random_forest',
    'train_logistic_regression',
    'train_svm',
    'train_gradient_boosting',
    'train_xgboost_classifier',
    'train_lightgbm_classifier',
    'train_shallow_nn',
    'train_deep_nn',
    'get_config',
    'list_configs',
]
