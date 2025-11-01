"""
Model Evaluation Utilities for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Provides comprehensive evaluation metrics for classification models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, matthews_corrcoef, cohen_kappa_score
)
from typing import Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')


def evaluate_classifier(
    y_true,
    y_pred,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = 'binary',
    return_report: bool = False
) -> Dict[str, float]:
    """
    Comprehensive evaluation of a classification model.
    
    Computes all standard classification metrics in one function.
    Perfect for logging to MLflow!
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for AUC)
        average: 'binary', 'macro', 'micro', or 'weighted' (for multiclass)
        return_report: If True, include detailed classification report
        
    Returns:
        Dictionary with all metrics
        
    Example:
        >>> metrics = evaluate_classifier(y_test, y_pred, y_pred_proba)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
        >>> print(f"AUC: {metrics['roc_auc']:.4f}")
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Additional metrics
    try:
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    except:
        metrics['matthews_corrcoef'] = 0.0
    
    try:
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    except:
        metrics['cohen_kappa'] = 0.0
    
    # ROC AUC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                # Binary classification
                if y_pred_proba.ndim == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:
                # Multiclass
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_pred_proba,
                    multi_class='ovr',
                    average=average
                )
        except Exception as e:
            metrics['roc_auc'] = None
    
    # Log Loss (if probabilities provided)
    if y_pred_proba is not None:
        try:
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
        except:
            metrics['log_loss'] = None
    
    # Confusion matrix metrics
    cm = confusion_matrix(y_true, y_pred)
    
    if len(np.unique(y_true)) == 2:
        # Binary classification specific metrics
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Classification report (if requested)
    if return_report:
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
    
    return metrics


def evaluate_binary_classifier(
    y_true,
    y_pred,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Specialized evaluation for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with binary classification metrics
    """
    return evaluate_classifier(
        y_true, y_pred, y_pred_proba,
        average='binary',
        return_report=False
    )


def evaluate_multiclass_classifier(
    y_true,
    y_pred,
    y_pred_proba: Optional[np.ndarray] = None,
    average: str = 'macro'
) -> Dict[str, float]:
    """
    Specialized evaluation for multiclass classification.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        y_pred_proba: Predicted probabilities (optional)
        average: 'macro', 'micro', or 'weighted'
        
    Returns:
        Dictionary with multiclass metrics
    """
    return evaluate_classifier(
        y_true, y_pred, y_pred_proba,
        average=average,
        return_report=False
    )


def get_confusion_matrix_metrics(y_true, y_pred) -> Dict[str, Union[int, float]]:
    """
    Extract all metrics from confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with confusion matrix based metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {}
    
    if len(np.unique(y_true)) == 2:
        # Binary classification
        tn, fp, fn, tp = cm.ravel()
        
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0.0  # False Discovery Rate
    
    else:
        # Multiclass
        metrics['confusion_matrix'] = cm.tolist()
        metrics['num_classes'] = len(np.unique(y_true))
    
    return metrics


def compare_models(
    models_dict: Dict[str, tuple],
    metric: str = 'accuracy'
) -> pd.DataFrame:
    """
    Compare multiple models based on their predictions.
    
    Args:
        models_dict: Dict of {model_name: (y_true, y_pred, y_pred_proba)}
        metric: Metric to rank by
        
    Returns:
        DataFrame with comparison results
        
    Example:
        >>> models = {
        ...     'RandomForest': (y_test, rf_pred, rf_proba),
        ...     'LogisticReg': (y_test, lr_pred, lr_proba)
        ... }
        >>> df = compare_models(models, metric='f1_score')
        >>> print(df)
    """
    results = []
    
    for model_name, (y_true, y_pred, y_pred_proba) in models_dict.items():
        metrics = evaluate_classifier(y_true, y_pred, y_pred_proba)
        metrics['model'] = model_name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Move model column to front
    cols = ['model'] + [col for col in df.columns if col != 'model']
    df = df[cols]
    
    # Sort by specified metric
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)
    
    return df


def print_evaluation_report(
    y_true,
    y_pred,
    y_pred_proba: Optional[np.ndarray] = None,
    model_name: str = "Model"
):
    """
    Print a formatted evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        model_name: Name of the model
    """
    metrics = evaluate_classifier(y_true, y_pred, y_pred_proba)
    
    print(f"\n{'=' * 60}")
    print(f"📊 {model_name} Evaluation Report")
    print(f"{'=' * 60}")
    
    print(f"\n🎯 Classification Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    
    if metrics.get('roc_auc') is not None:
        print(f"   ROC AUC:   {metrics['roc_auc']:.4f}")
    
    if 'true_positives' in metrics:
        print(f"\n🔢 Confusion Matrix Metrics:")
        print(f"   True Positives:  {metrics['true_positives']}")
        print(f"   True Negatives:  {metrics['true_negatives']}")
        print(f"   False Positives: {metrics['false_positives']}")
        print(f"   False Negatives: {metrics['false_negatives']}")
        print(f"   Specificity:     {metrics['specificity']:.4f}")
    
    print(f"\n📈 Additional Metrics:")
    print(f"   Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
    print(f"   Cohen's Kappa:        {metrics['cohen_kappa']:.4f}")
    
    if metrics.get('log_loss') is not None:
        print(f"   Log Loss:             {metrics['log_loss']:.4f}")
    
    print(f"{'=' * 60}\n")


def calculate_classification_threshold_metrics(
    y_true,
    y_pred_proba,
    thresholds: Optional[list] = None
) -> pd.DataFrame:
    """
    Calculate metrics at different classification thresholds.
    
    Useful for finding optimal threshold for your use case.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        thresholds: List of thresholds to test (default: 0.3 to 0.7)
        
    Returns:
        DataFrame with metrics for each threshold
        
    Example:
        >>> df = calculate_classification_threshold_metrics(y_test, y_proba)
        >>> print(df)
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    results = []
    
    for threshold in thresholds:
        # Apply threshold
        if y_pred_proba.ndim == 2:
            y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)
        else:
            y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = evaluate_binary_classifier(y_true, y_pred, y_pred_proba)
        metrics['threshold'] = threshold
        
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['threshold'] + [col for col in df.columns if col != 'threshold']
    df = df[cols]
    
    return df


def get_metric_summary(metrics: Dict[str, float]) -> str:
    """
    Get a one-line summary of key metrics.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string summary
        
    Example:
        >>> summary = get_metric_summary(metrics)
        >>> print(summary)
        Acc: 0.85 | Prec: 0.82 | Rec: 0.79 | F1: 0.80 | AUC: 0.88
    """
    parts = []
    
    if 'accuracy' in metrics:
        parts.append(f"Acc: {metrics['accuracy']:.4f}")
    
    if 'precision' in metrics:
        parts.append(f"Prec: {metrics['precision']:.4f}")
    
    if 'recall' in metrics:
        parts.append(f"Rec: {metrics['recall']:.4f}")
    
    if 'f1_score' in metrics:
        parts.append(f"F1: {metrics['f1_score']:.4f}")
    
    if metrics.get('roc_auc') is not None:
        parts.append(f"AUC: {metrics['roc_auc']:.4f}")
    
    return " | ".join(parts)


if __name__ == "__main__":
    """Test evaluation utilities."""
    print("🧪 Testing Evaluation Utilities\n")
    
    # Create synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.random(100)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Test 1: Basic evaluation
    print("1️⃣ Testing evaluate_classifier...")
    metrics = evaluate_classifier(y_true, y_pred, y_pred_proba)
    print(f"   ✅ Computed {len(metrics)} metrics")
    print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
    print(f"   ✅ F1-Score: {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"   ✅ ROC AUC: {metrics['roc_auc']:.4f}")
    
    print()
    
    # Test 2: Print report
    print("2️⃣ Testing print_evaluation_report...")
    print_evaluation_report(y_true, y_pred, y_pred_proba, "Test Model")
    
    # Test 3: Metric summary
    print("3️⃣ Testing get_metric_summary...")
    summary = get_metric_summary(metrics)
    print(f"   ✅ {summary}")
    
    print("\n✅ All evaluation tests passed!\n")
