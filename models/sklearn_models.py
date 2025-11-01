"""
Scikit-learn Models for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Production-ready scikit-learn model training functions.
Each function trains a model and returns the trained model plus metrics.

DO NOT modify the model training logic - only modify hyperparameters!
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.evaluator import evaluate_classifier
from utils.visualizer import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
import warnings

warnings.filterwarnings('ignore')


def train_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = 'sqrt',
    random_state: int = 42,
    log_to_mlflow: bool = False
):
    """
    Train a Random Forest Classifier.
    
    This is a PRODUCTION-READY function. Students should:
    - Modify hyperparameters
    - Enable/disable MLflow logging
    - NOT modify the training logic
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_estimators: Number of trees (default: 100)
        max_depth: Maximum tree depth (default: 10)
        min_samples_split: Minimum samples to split node (default: 2)
        min_samples_leaf: Minimum samples in leaf (default: 1)
        max_features: Number of features to consider (default: 'sqrt')
        random_state: Random seed (default: 42)
        log_to_mlflow: Whether to log to MLflow (default: False)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
        
    Example:
        >>> model, metrics = train_random_forest(X_train, y_train, X_test, y_test)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    print(f"\n{'='*60}")
    print(f"🌲 Training Random Forest Classifier")
    print(f"{'='*60}")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model
    print(f"📊 Training with hyperparameters:")
    print(f"   - n_estimators: {n_estimators}")
    print(f"   - max_depth: {max_depth}")
    print(f"   - min_samples_split: {min_samples_split}")
    print(f"   - min_samples_leaf: {min_samples_leaf}")
    print(f"   - max_features: {max_features}")
    
    model.fit(X_train, y_train)
    print(f"✅ Training complete!")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Evaluate
    metrics = evaluate_classifier(y_test, y_pred, y_pred_proba)
    
    print(f"\n📈 Test Set Performance:")
    print(f"   - Accuracy: {metrics['accuracy']:.4f}")
    print(f"   - Precision: {metrics['precision']:.4f}")
    print(f"   - Recall: {metrics['recall']:.4f}")
    print(f"   - F1-Score: {metrics['f1_score']:.4f}")
    if metrics.get('roc_auc'):
        print(f"   - ROC AUC: {metrics['roc_auc']:.4f}")
    
    # MLflow logging
    if log_to_mlflow:
        import mlflow
        from utils.mlflow_helpers import log_metrics_dict, log_params_dict
        
        # Log parameters
        params = {
            'model_type': 'RandomForest',
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state
        }
        log_params_dict(params)
        
        # Log metrics
        log_metrics_dict(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Generate and log plots
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


def train_logistic_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    C: float = 1.0,
    penalty: str = 'l2',
    solver: str = 'lbfgs',
    max_iter: int = 1000,
    random_state: int = 42,
    log_to_mlflow: bool = False
):
    """
    Train a Logistic Regression Classifier.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        C: Inverse of regularization strength (default: 1.0)
        penalty: Regularization type 'l1', 'l2', or 'elasticnet' (default: 'l2')
        solver: Optimization algorithm (default: 'lbfgs')
        max_iter: Maximum iterations (default: 1000)
        random_state: Random seed (default: 42)
        log_to_mlflow: Whether to log to MLflow (default: False)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"📊 Training Logistic Regression")
    print(f"{'='*60}")
    
    # Initialize model
    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Train model
    print(f"📊 Training with hyperparameters:")
    print(f"   - C: {C}")
    print(f"   - penalty: {penalty}")
    print(f"   - solver: {solver}")
    print(f"   - max_iter: {max_iter}")
    
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
            'model_type': 'LogisticRegression',
            'C': C,
            'penalty': penalty,
            'solver': solver,
            'max_iter': max_iter,
            'random_state': random_state
        }
        log_params_dict(params)
        log_metrics_dict(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        
        roc_fig = plot_roc_curve(y_test, y_pred_proba)
        mlflow.log_figure(roc_fig, "roc_curve.png")
        
        print(f"\n✅ Logged to MLflow")
    
    print(f"{'='*60}\n")
    
    return model, metrics


def train_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    C: float = 1.0,
    kernel: str = 'rbf',
    gamma: str = 'scale',
    random_state: int = 42,
    log_to_mlflow: bool = False
):
    """
    Train a Support Vector Machine Classifier.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        C: Regularization parameter (default: 1.0)
        kernel: Kernel type 'linear', 'poly', 'rbf', 'sigmoid' (default: 'rbf')
        gamma: Kernel coefficient (default: 'scale')
        random_state: Random seed (default: 42)
        log_to_mlflow: Whether to log to MLflow (default: False)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"🎯 Training SVM Classifier")
    print(f"{'='*60}")
    
    # Initialize model
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=random_state,
        probability=True  # Enable probability predictions
    )
    
    # Train model
    print(f"📊 Training with hyperparameters:")
    print(f"   - C: {C}")
    print(f"   - kernel: {kernel}")
    print(f"   - gamma: {gamma}")
    
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
            'model_type': 'SVM',
            'C': C,
            'kernel': kernel,
            'gamma': gamma,
            'random_state': random_state
        }
        log_params_dict(params)
        log_metrics_dict(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        
        roc_fig = plot_roc_curve(y_test, y_pred_proba)
        mlflow.log_figure(roc_fig, "roc_curve.png")
        
        print(f"\n✅ Logged to MLflow")
    
    print(f"{'='*60}\n")
    
    return model, metrics


def train_gradient_boosting(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: int = 42,
    log_to_mlflow: bool = False
):
    """
    Train a Gradient Boosting Classifier.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_estimators: Number of boosting stages (default: 100)
        learning_rate: Learning rate (default: 0.1)
        max_depth: Maximum tree depth (default: 3)
        min_samples_split: Minimum samples to split (default: 2)
        min_samples_leaf: Minimum samples in leaf (default: 1)
        subsample: Fraction of samples for fitting trees (default: 1.0)
        random_state: Random seed (default: 42)
        log_to_mlflow: Whether to log to MLflow (default: False)
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training Gradient Boosting Classifier")
    print(f"{'='*60}")
    
    # Initialize model
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state
    )
    
    # Train model
    print(f"📊 Training with hyperparameters:")
    print(f"   - n_estimators: {n_estimators}")
    print(f"   - learning_rate: {learning_rate}")
    print(f"   - max_depth: {max_depth}")
    print(f"   - subsample: {subsample}")
    
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
            'model_type': 'GradientBoosting',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'random_state': random_state
        }
        log_params_dict(params)
        log_metrics_dict(metrics)
        mlflow.sklearn.log_model(model, "model")
        
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
    """Test sklearn models."""
    print("🧪 Testing Scikit-learn Models\n")
    
    from utils.data_loader import load_customer_churn_data
    
    # Load data
    X_train, X_test, y_train, y_test = load_customer_churn_data()
    print(f"✅ Loaded data: {len(X_train)} train samples, {len(X_test)} test samples\n")
    
    # Test each model
    print("Testing Random Forest...")
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, n_estimators=50)
    
    print("\nTesting Logistic Regression...")
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("\nTesting SVM...")
    svm_model, svm_metrics = train_svm(X_train, y_train, X_test, y_test)
    
    print("\nTesting Gradient Boosting...")
    gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_test, y_test, n_estimators=50)
    
    print("\n✅ All sklearn models tested successfully!\n")
