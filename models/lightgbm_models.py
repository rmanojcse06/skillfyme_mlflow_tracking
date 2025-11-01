"""
LightGBM Models for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Production-ready LightGBM model training functions.
"""

import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.evaluator import evaluate_classifier
from utils.visualizer import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
import warnings

warnings.filterwarnings('ignore')


def train_lightgbm_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators: int = 100,
    max_depth: int = -1,
    learning_rate: float = 0.1,
    num_leaves: int = 31,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0,
    reg_lambda: float = 0,
    random_state: int = 42,
    log_to_mlflow: bool = False
):
    """
    Train a LightGBM Classifier.
    
    LightGBM is known for:
    - Fast training speed
    - Low memory usage
    - High accuracy
    - Handling large-scale data
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_estimators: Number of boosting iterations (default: 100)
        max_depth: Maximum tree depth, -1 means no limit (default: -1)
        learning_rate: Learning rate (default: 0.1)
        num_leaves: Maximum number of leaves in one tree (default: 31)
        subsample: Subsample ratio (default: 1.0)
        colsample_bytree: Feature subsample ratio (default: 1.0)
        reg_alpha: L1 regularization (default: 0)
        reg_lambda: L2 regularization (default: 0)
        random_state: Random seed (default: 42)
        log_to_mlflow: Whether to log to MLflow (default: False)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
        
    Example:
        >>> model, metrics = train_lightgbm_classifier(X_train, y_train, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    print(f"\n{'='*60}")
    print(f"💡 Training LightGBM Classifier")
    print(f"{'='*60}")
    
    # Initialize model
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    
    # Train model
    print(f"📊 Training with hyperparameters:")
    print(f"   - n_estimators: {n_estimators}")
    print(f"   - max_depth: {max_depth}")
    print(f"   - learning_rate: {learning_rate}")
    print(f"   - num_leaves: {num_leaves}")
    
    model.fit(X_train, y_train)
    print(f"✅ Training complete!")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    metrics = evaluate_classifier(y_test, y_pred, y_pred_proba)
    
    print(f"\n📈 Test Set Performance:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
    
    # MLflow logging
    if log_to_mlflow:
        import mlflow
        from utils.mlflow_helpers import log_metrics_dict, log_params_dict
        
        params = {
            'model_type': 'LightGBM',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state
        }
        log_params_dict(params)
        log_metrics_dict(metrics)
        mlflow.lightgbm.log_model(model, "model")
        
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        
        roc_fig = plot_roc_curve(y_test, y_pred_proba)
        mlflow.log_figure(roc_fig, "roc_curve.png")
        
        fi_fig = plot_feature_importance(feature_names, model.feature_importances_, top_n=15)
        mlflow.log_figure(fi_fig, "feature_importance.png")
        
        print(f"\n✅ Logged to MLflow")
    
    print(f"{'='*60}\n")
    
    return model, metrics


if __name__ == "__main__":
    """Test LightGBM models."""
    print("🧪 Testing LightGBM Models\n")
    
    from utils.data_loader import load_customer_churn_data
    
    X_train, X_test, y_train, y_test = load_customer_churn_data()
    print(f"✅ Loaded data: {len(X_train)} train samples, {len(X_test)} test samples\n")
    
    model, metrics = train_lightgbm_classifier(X_train, y_train, X_test, y_test, n_estimators=50)
    
    print("\n✅ LightGBM model tested successfully!\n")
