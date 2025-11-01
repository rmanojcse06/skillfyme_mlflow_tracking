"""
Visualization Utilities for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Provides functions to generate professional plots for MLflow artifacts.
All functions return matplotlib Figure objects ready for MLflow logging.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)
from typing import Optional, List, Union
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    cmap: str = "Blues"
):
    """
    Generate confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes (optional)
        normalize: Whether to normalize the matrix
        title: Plot title
        cmap: Color map
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_confusion_matrix(y_test, y_pred, ['No Churn', 'Churn'])
        >>> mlflow.log_figure(fig, "confusion_matrix.png")
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        square=True,
        cbar=True,
        ax=ax,
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true,
    y_pred_proba,
    title: str = "ROC Curve",
    class_label: Optional[str] = None
):
    """
    Generate ROC curve plot.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        class_label: Label for the positive class
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_roc_curve(y_test, y_proba)
        >>> mlflow.log_figure(fig, "roc_curve.png")
    """
    # Handle 2D probability arrays
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(
        fpr, tpr,
        color='darkorange',
        lw=2,
        label=f'ROC Curve (AUC = {roc_auc:.3f})'
    )
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true,
    y_pred_proba,
    title: str = "Precision-Recall Curve"
):
    """
    Generate precision-recall curve plot.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_precision_recall_curve(y_test, y_proba)
        >>> mlflow.log_figure(fig, "precision_recall.png")
    """
    # Handle 2D probability arrays
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot curve
    ax.plot(
        recall, precision,
        color='darkorange',
        lw=2,
        label=f'PR Curve (AUC = {pr_auc:.3f})'
    )
    
    # Plot baseline
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance"
):
    """
    Generate feature importance bar plot.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        top_n: Number of top features to show
        title: Plot title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_feature_importance(feature_names, model.feature_importances_)
        >>> mlflow.log_figure(fig, "feature_importance.png")
    """
    # Create DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values
    })
    
    # Sort and get top N
    df = df.sort_values('importance', ascending=True).tail(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    
    # Plot
    bars = ax.barh(df['feature'], df['importance'], color='steelblue', alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width, bar.get_y() + bar.get_height()/2,
            f'{width:.4f}',
            ha='left', va='center',
            fontsize=9,
            color='black'
        )
    
    plt.tight_layout()
    return fig


def plot_learning_curve(
    train_scores: List[float],
    val_scores: List[float],
    metric_name: str = "Accuracy",
    title: str = "Learning Curve"
):
    """
    Generate learning curve plot showing training and validation scores.
    
    Args:
        train_scores: List of training scores per epoch/iteration
        val_scores: List of validation scores per epoch/iteration
        metric_name: Name of the metric
        title: Plot title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_learning_curve(train_acc, val_acc, "Accuracy")
        >>> mlflow.log_figure(fig, "learning_curve.png")
    """
    epochs = range(1, len(train_scores) + 1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves
    ax.plot(epochs, train_scores, 'o-', label=f'Training {metric_name}', linewidth=2, markersize=6)
    ax.plot(epochs, val_scores, 's-', label=f'Validation {metric_name}', linewidth=2, markersize=6)
    
    # Formatting
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add min/max annotations
    best_val_epoch = np.argmax(val_scores) + 1
    best_val_score = max(val_scores)
    ax.axvline(best_val_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_val_epoch}')
    ax.annotate(
        f'Best: {best_val_score:.4f}',
        xy=(best_val_epoch, best_val_score),
        xytext=(10, 10),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    plt.tight_layout()
    return fig


def plot_class_distribution(
    y,
    class_names: Optional[List[str]] = None,
    title: str = "Class Distribution"
):
    """
    Generate bar plot showing class distribution.
    
    Args:
        y: Target labels
        class_names: Names for classes (optional)
        title: Plot title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_class_distribution(y_train, ['No Churn', 'Churn'])
        >>> mlflow.log_figure(fig, "class_distribution.png")
    """
    # Count classes
    unique, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [str(cls) for cls in unique]
    
    # Calculate percentages
    percentages = 100 * counts / counts.sum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot bars
    bars = ax.bar(class_names, counts, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add count and percentage labels on bars
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, height,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    # Formatting
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_prediction_distribution(
    y_pred_proba,
    threshold: float = 0.5,
    title: str = "Prediction Probability Distribution"
):
    """
    Generate histogram of prediction probabilities.
    
    Args:
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold (default: 0.5)
        title: Plot title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_prediction_distribution(y_proba, threshold=0.5)
        >>> mlflow.log_figure(fig, "prediction_dist.png")
    """
    # Handle 2D probability arrays
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    ax.hist(y_pred_proba, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    
    # Add threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
    
    # Formatting
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metric_comparison(
    metrics_dict: dict,
    metric_names: Optional[List[str]] = None,
    title: str = "Model Comparison"
):
    """
    Generate grouped bar plot comparing metrics across models.
    
    Args:
        metrics_dict: Dict of {model_name: {metric: value}}
        metric_names: List of metrics to include (optional, uses all if None)
        title: Plot title
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> metrics = {
        ...     'RF': {'accuracy': 0.85, 'f1_score': 0.82},
        ...     'LR': {'accuracy': 0.80, 'f1_score': 0.78}
        ... }
        >>> fig = plot_metric_comparison(metrics)
        >>> mlflow.log_figure(fig, "model_comparison.png")
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    if metric_names is not None:
        df = df[metric_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot grouped bars
    df.plot(kind='bar', ax=ax, width=0.8, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Metrics', loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    X,
    feature_names: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix",
    figsize: tuple = (12, 10)
):
    """
    Generate correlation heatmap for features.
    
    Args:
        X: Feature matrix (DataFrame or array)
        feature_names: List of feature names (if X is array)
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> fig = plot_correlation_matrix(X_train)
        >>> mlflow.log_figure(fig, "correlation_matrix.png")
    """
    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
    
    # Calculate correlation
    corr = X.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        corr,
        annot=True if len(corr) <= 15 else False,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_model_report_figure(
    metrics: dict,
    model_name: str = "Model"
):
    """
    Create a comprehensive single-page model report with multiple subplots.
    
    Args:
        metrics: Dictionary with all metrics and data
        model_name: Name of the model
        
    Returns:
        matplotlib Figure object with multiple subplots
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(f'{model_name} - Comprehensive Report', fontsize=16, fontweight='bold')
    
    # Subplot 1: Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    # Add confusion matrix plot here
    
    # Subplot 2: ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    # Add ROC curve plot here
    
    # Subplot 3: Metrics Table
    ax3 = fig.add_subplot(gs[1, 0])
    # Add metrics table here
    
    # Subplot 4: Feature Importance
    ax4 = fig.add_subplot(gs[1, 1])
    # Add feature importance plot here
    
    return fig


if __name__ == "__main__":
    """Test visualization utilities."""
    print("🧪 Testing Visualization Utilities\n")
    
    # Create synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.random(100)
    
    # Test 1: Confusion Matrix
    print("1️⃣ Testing plot_confusion_matrix...")
    fig = plot_confusion_matrix(y_true, y_pred, class_names=['Class 0', 'Class 1'])
    print(f"   ✅ Created confusion matrix plot")
    plt.close(fig)
    
    # Test 2: ROC Curve
    print("2️⃣ Testing plot_roc_curve...")
    fig = plot_roc_curve(y_true, y_pred_proba)
    print(f"   ✅ Created ROC curve plot")
    plt.close(fig)
    
    # Test 3: Feature Importance
    print("3️⃣ Testing plot_feature_importance...")
    features = [f"Feature_{i}" for i in range(10)]
    importance = np.random.random(10)
    fig = plot_feature_importance(features, importance, top_n=10)
    print(f"   ✅ Created feature importance plot")
    plt.close(fig)
    
    # Test 4: Class Distribution
    print("4️⃣ Testing plot_class_distribution...")
    fig = plot_class_distribution(y_true, class_names=['Negative', 'Positive'])
    print(f"   ✅ Created class distribution plot")
    plt.close(fig)
    
    print("\n✅ All visualization tests passed!\n")
