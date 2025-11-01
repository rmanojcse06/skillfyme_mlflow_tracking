"""
MLflow Helper Utilities for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Provides convenient functions for MLflow logging operations.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.keras
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import json
import warnings

warnings.filterwarnings('ignore')


def log_metrics_dict(metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
    """
    Log multiple metrics to MLflow at once.
    
    Args:
        metrics: Dictionary of {metric_name: value}
        step: Optional step number for metric history
        
    Example:
        >>> metrics = {'accuracy': 0.85, 'f1_score': 0.82, 'auc': 0.88}
        >>> log_metrics_dict(metrics)
    """
    # Filter out None values and non-numeric values
    filtered_metrics = {}
    for key, value in metrics.items():
        if value is not None and isinstance(value, (int, float, np.integer, np.floating)):
            # Convert numpy types to Python types
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            filtered_metrics[key] = value
    
    # Log to MLflow
    if filtered_metrics:
        mlflow.log_metrics(filtered_metrics, step=step)


def log_params_dict(params: Dict[str, Any]):
    """
    Log multiple parameters to MLflow at once.
    
    Args:
        params: Dictionary of {param_name: value}
        
    Example:
        >>> params = {'n_estimators': 100, 'max_depth': 10, 'learning_rate': 0.1}
        >>> log_params_dict(params)
    """
    # Convert all values to strings (MLflow requirement)
    str_params = {key: str(value) for key, value in params.items()}
    mlflow.log_params(str_params)


def log_model_signature(model, X_sample, y_sample=None):
    """
    Log model with input/output signature inference.
    
    Args:
        model: Trained model
        X_sample: Sample input data for signature inference
        y_sample: Sample output data (optional)
        
    Example:
        >>> log_model_signature(model, X_train[:5])
    """
    from mlflow.models.signature import infer_signature
    
    # Infer signature
    predictions = model.predict(X_sample)
    signature = infer_signature(X_sample, predictions)
    
    # Log model with signature
    try:
        if hasattr(model, '__module__'):
            if 'sklearn' in model.__module__:
                mlflow.sklearn.log_model(model, "model", signature=signature)
            elif 'xgboost' in model.__module__:
                mlflow.xgboost.log_model(model, "model", signature=signature)
            elif 'lightgbm' in model.__module__:
                mlflow.lightgbm.log_model(model, "model", signature=signature)
            elif 'keras' in model.__module__ or 'tensorflow' in model.__module__:
                mlflow.keras.log_model(model, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)
    except Exception as e:
        print(f"   ⚠️ Could not log model: {e}")


def log_dataset_info(X_train, X_test, y_train, y_test, dataset_name: str = "dataset"):
    """
    Log dataset information as parameters and metrics.
    
    Args:
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
        dataset_name: Name of the dataset
        
    Example:
        >>> log_dataset_info(X_train, X_test, y_train, y_test, "customer_churn")
    """
    # Dataset parameters
    params = {
        f'{dataset_name}_train_samples': len(X_train),
        f'{dataset_name}_test_samples': len(X_test),
        f'{dataset_name}_n_features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0]),
    }
    
    # Class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    params[f'{dataset_name}_train_class_dist'] = str(dict(zip(unique_train, counts_train)))
    params[f'{dataset_name}_test_class_dist'] = str(dict(zip(unique_test, counts_test)))
    
    # Log
    log_params_dict(params)


def log_figure_to_mlflow(fig, artifact_name: str):
    """
    Log a matplotlib figure to MLflow.
    
    Args:
        fig: Matplotlib figure object
        artifact_name: Name for the artifact (should end with .png)
        
    Example:
        >>> fig = plot_confusion_matrix(y_test, y_pred)
        >>> log_figure_to_mlflow(fig, "confusion_matrix.png")
    """
    mlflow.log_figure(fig, artifact_name)


def log_dict_as_json(data: dict, artifact_name: str):
    """
    Log a dictionary as JSON artifact.
    
    Args:
        data: Dictionary to save
        artifact_name: Name for the artifact (should end with .json)
        
    Example:
        >>> config = {'n_estimators': 100, 'max_depth': 10}
        >>> log_dict_as_json(config, "model_config.json")
    """
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f, indent=2)
        temp_path = f.name
    
    mlflow.log_artifact(temp_path, artifact_name)
    
    # Clean up
    Path(temp_path).unlink()


def log_dataframe_as_csv(df: pd.DataFrame, artifact_name: str):
    """
    Log a pandas DataFrame as CSV artifact.
    
    Args:
        df: DataFrame to save
        artifact_name: Name for the artifact (should end with .csv)
        
    Example:
        >>> results_df = pd.DataFrame({'metric': ['acc', 'f1'], 'value': [0.85, 0.82]})
        >>> log_dataframe_as_csv(results_df, "results.csv")
    """
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name
    
    mlflow.log_artifact(temp_path, artifact_name)
    
    # Clean up
    Path(temp_path).unlink()


def log_text_file(content: str, artifact_name: str):
    """
    Log text content as an artifact.
    
    Args:
        content: Text content
        artifact_name: Name for the artifact (should end with .txt)
        
    Example:
        >>> notes = "This model performed well on the test set."
        >>> log_text_file(notes, "notes.txt")
    """
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    mlflow.log_artifact(temp_path, artifact_name)
    
    # Clean up
    Path(temp_path).unlink()


def set_tags_from_dict(tags: Dict[str, str]):
    """
    Set multiple MLflow tags at once.
    
    Args:
        tags: Dictionary of {tag_name: tag_value}
        
    Example:
        >>> tags = {'model_type': 'RandomForest', 'experiment': 'baseline', 'version': 'v1'}
        >>> set_tags_from_dict(tags)
    """
    for key, value in tags.items():
        mlflow.set_tag(key, str(value))


def log_model_info(model, model_name: str):
    """
    Log comprehensive model information as parameters and tags.
    
    Args:
        model: Trained model
        model_name: Name of the model
        
    Example:
        >>> log_model_info(rf_model, "RandomForest")
    """
    # Basic info
    mlflow.set_tag("model_type", model_name)
    mlflow.set_tag("model_class", model.__class__.__name__)
    
    # Try to get model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        log_params_dict(params)


def create_nested_run(run_name: str, parent_run_id: Optional[str] = None):
    """
    Create a nested run context manager.
    
    Args:
        run_name: Name for the nested run
        parent_run_id: ID of parent run (uses active run if None)
        
    Returns:
        MLflow run context manager
        
    Example:
        >>> with mlflow.start_run(run_name="Parent") as parent_run:
        ...     with create_nested_run("Child"):
        ...         mlflow.log_param("test", "value")
    """
    if parent_run_id is None:
        active_run = mlflow.active_run()
        parent_run_id = active_run.info.run_id if active_run else None
    
    return mlflow.start_run(
        run_name=run_name,
        nested=True
    )


def log_confusion_matrix_metrics(y_true, y_pred):
    """
    Calculate and log confusion matrix metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Example:
        >>> log_confusion_matrix_metrics(y_test, y_pred)
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if len(np.unique(y_true)) == 2:
        # Binary classification
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
        }
        
        log_metrics_dict(metrics)


def log_training_history(history: Dict[str, List[float]], prefix: str = ""):
    """
    Log training history metrics (useful for neural networks).
    
    Args:
        history: Dictionary with lists of metrics per epoch
        prefix: Optional prefix for metric names
        
    Example:
        >>> history = {'loss': [0.5, 0.4, 0.3], 'accuracy': [0.8, 0.85, 0.9]}
        >>> log_training_history(history, prefix="train_")
    """
    for metric_name, values in history.items():
        for step, value in enumerate(values):
            mlflow.log_metric(f"{prefix}{metric_name}", value, step=step)


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get experiment ID or create if it doesn't exist.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID
        
    Example:
        >>> exp_id = get_or_create_experiment("My Experiment")
        >>> mlflow.start_run(experiment_id=exp_id)
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    return experiment_id


def log_preprocessing_info(scaler_type: str, feature_count: int, preprocessing_steps: Optional[List[str]] = None):
    """
    Log preprocessing information.
    
    Args:
        scaler_type: Type of scaler used
        feature_count: Number of features after preprocessing
        preprocessing_steps: List of preprocessing steps applied
        
    Example:
        >>> log_preprocessing_info('standard', 15, ['scaling', 'encoding'])
    """
    params = {
        'scaler_type': scaler_type,
        'feature_count_final': feature_count,
    }
    
    if preprocessing_steps:
        params['preprocessing_steps'] = ', '.join(preprocessing_steps)
    
    log_params_dict(params)


def log_hyperparameter_search_summary(results: pd.DataFrame, best_params: dict):
    """
    Log summary of hyperparameter search results.
    
    Args:
        results: DataFrame with search results
        best_params: Dictionary of best hyperparameters
        
    Example:
        >>> results = pd.DataFrame({'accuracy': [0.8, 0.85, 0.9]})
        >>> best = {'n_estimators': 100, 'max_depth': 10}
        >>> log_hyperparameter_search_summary(results, best)
    """
    # Log best parameters
    for param_name, param_value in best_params.items():
        mlflow.log_param(f"best_{param_name}", param_value)
    
    # Log search statistics
    if 'score' in results.columns or 'mean_test_score' in results.columns:
        score_col = 'score' if 'score' in results.columns else 'mean_test_score'
        
        metrics = {
            'search_best_score': results[score_col].max(),
            'search_worst_score': results[score_col].min(),
            'search_mean_score': results[score_col].mean(),
            'search_std_score': results[score_col].std(),
            'search_total_trials': len(results)
        }
        
        log_metrics_dict(metrics)
    
    # Log results table as artifact
    log_dataframe_as_csv(results, "hyperparameter_search_results.csv")


def compare_runs_and_log(run_ids: List[str], metric_name: str = 'accuracy'):
    """
    Compare multiple runs and log comparison summary.
    
    Args:
        run_ids: List of MLflow run IDs to compare
        metric_name: Metric to compare by
        
    Example:
        >>> run_ids = ['abc123', 'def456', 'ghi789']
        >>> compare_runs_and_log(run_ids, 'accuracy')
    """
    client = mlflow.tracking.MlflowClient()
    
    comparison_data = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        comparison_data.append({
            'run_id': run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            metric_name: run.data.metrics.get(metric_name, None)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values(metric_name, ascending=False)
    
    # Log comparison
    log_dataframe_as_csv(df, f"run_comparison_{metric_name}.csv")
    
    return df


def setup_mlflow_tracking(tracking_uri: str = "./mlruns", experiment_name: str = "Default"):
    """
    Setup MLflow tracking with specified URI and experiment.
    
    Args:
        tracking_uri: Path to MLflow tracking directory
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID
        
    Example:
        >>> exp_id = setup_mlflow_tracking("./mlruns", "My Experiment")
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


if __name__ == "__main__":
    """Test MLflow helper utilities."""
    print("🧪 Testing MLflow Helper Utilities\n")
    
    # Setup test environment
    import tempfile
    import shutil
    
    # Create temporary directory for MLflow
    temp_dir = tempfile.mkdtemp()
    mlflow.set_tracking_uri(f"file://{temp_dir}")
    mlflow.set_experiment("test_experiment")
    
    try:
        # Test 1: Log metrics dict
        print("1️⃣ Testing log_metrics_dict...")
        with mlflow.start_run():
            metrics = {'accuracy': 0.85, 'f1_score': 0.82}
            log_metrics_dict(metrics)
            print(f"   ✅ Logged {len(metrics)} metrics")
        
        # Test 2: Log params dict
        print("2️⃣ Testing log_params_dict...")
        with mlflow.start_run():
            params = {'n_estimators': 100, 'max_depth': 10}
            log_params_dict(params)
            print(f"   ✅ Logged {len(params)} parameters")
        
        # Test 3: Set tags
        print("3️⃣ Testing set_tags_from_dict...")
        with mlflow.start_run():
            tags = {'model_type': 'RF', 'version': 'v1'}
            set_tags_from_dict(tags)
            print(f"   ✅ Set {len(tags)} tags")
        
        print("\n✅ All MLflow helper tests passed!\n")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
