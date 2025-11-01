"""
Utils Package for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

This package provides utility modules for data loading, preprocessing,
evaluation, visualization, and MLflow logging.
"""

from .data_loader import (
    load_customer_churn_data,
    load_iris_data,
    load_data,
    get_dataset_info,
    print_dataset_summary
)

from .preprocessor import (
    DataPreprocessor,
    create_preprocessing_pipeline,
    scale_features,
    select_top_k_features,
    get_preprocessing_config
)

from .evaluator import (
    evaluate_classifier,
    evaluate_binary_classifier,
    evaluate_multiclass_classifier,
    get_confusion_matrix_metrics,
    compare_models,
    print_evaluation_report
)

from .visualizer import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_learning_curve,
    plot_class_distribution,
    plot_prediction_distribution,
    plot_metric_comparison
)

from .mlflow_helpers import (
    log_metrics_dict,
    log_params_dict,
    log_model_signature,
    log_dataset_info,
    log_figure_to_mlflow,
    set_tags_from_dict,
    log_model_info,
    setup_mlflow_tracking,
    get_or_create_experiment
)

__all__ = [
    # Data loading
    'load_customer_churn_data',
    'load_iris_data',
    'load_data',
    'get_dataset_info',
    'print_dataset_summary',
    
    # Preprocessing
    'DataPreprocessor',
    'create_preprocessing_pipeline',
    'scale_features',
    'select_top_k_features',
    'get_preprocessing_config',
    
    # Evaluation
    'evaluate_classifier',
    'evaluate_binary_classifier',
    'evaluate_multiclass_classifier',
    'get_confusion_matrix_metrics',
    'compare_models',
    'print_evaluation_report',
    
    # Visualization
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_feature_importance',
    'plot_learning_curve',
    'plot_class_distribution',
    'plot_prediction_distribution',
    'plot_metric_comparison',
    
    # MLflow helpers
    'log_metrics_dict',
    'log_params_dict',
    'log_model_signature',
    'log_dataset_info',
    'log_figure_to_mlflow',
    'set_tags_from_dict',
    'log_model_info',
    'setup_mlflow_tracking',
    'get_or_create_experiment',
]
