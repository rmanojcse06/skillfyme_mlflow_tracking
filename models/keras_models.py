"""
Keras Neural Network Models for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Production-ready Keras model training functions.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.evaluator import evaluate_classifier
from utils.visualizer import plot_confusion_matrix, plot_roc_curve, plot_learning_curve
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


def train_shallow_nn(
    X_train, y_train, X_test, y_test,
    units: int = 64, dropout_rate: float = 0.3,
    learning_rate: float = 0.001, epochs: int = 50,
    batch_size: int = 32, random_state: int = 42,
    log_to_mlflow: bool = False
):
    """Train a shallow neural network (2 hidden layers)."""
    print(f"\n{'='*60}")
    print(f"🧠 Training Shallow Neural Network")
    print(f"{'='*60}")
    
    tf.random.set_seed(random_state)
    np.random.seed(random_state)
    
    # Build model
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(units//2, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📊 Training with hyperparameters:")
    print(f"   - units: {units}")
    print(f"   - dropout_rate: {dropout_rate}")
    print(f"   - learning_rate: {learning_rate}")
    print(f"   - epochs: {epochs}")
    
    # Train with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    print(f"✅ Training complete!")
    
    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    
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
            'model_type': 'ShallowNN',
            'units': units,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'actual_epochs': len(history.history['loss'])
        }
        log_params_dict(params)
        log_metrics_dict(metrics)
        mlflow.keras.log_model(model, "model")
        
        cm_fig = plot_confusion_matrix(y_test, y_pred)
        mlflow.log_figure(cm_fig, "confusion_matrix.png")
        
        roc_fig = plot_roc_curve(y_test, y_pred_proba)
        mlflow.log_figure(roc_fig, "roc_curve.png")
        
        lc_fig = plot_learning_curve(history.history['accuracy'], history.history['val_accuracy'], "Accuracy")
        mlflow.log_figure(lc_fig, "learning_curve.png")
        
        print(f"\n✅ Logged to MLflow")
    
    print(f"{'='*60}\n")
    
    return model, metrics


def train_deep_nn(
    X_train, y_train, X_test, y_test,
    units: int = 128, dropout_rate: float = 0.4,
    learning_rate: float = 0.001, epochs: int = 50,
    batch_size: int = 32, random_state: int = 42,
    log_to_mlflow: bool = False
):
    """Train a deep neural network (4 hidden layers)."""
    print(f"\n{'='*60}")
    print(f"🧠 Training Deep Neural Network")
    print(f"{'='*60}")
    
    tf.random.set_seed(random_state)
    
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(units, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(units//2, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(units//4, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(units//8, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    metrics = evaluate_classifier(y_test, y_pred, y_pred_proba)
    
    print(f"✅ Training complete! Accuracy: {metrics['accuracy']:.4f}")
    
    if log_to_mlflow:
        import mlflow
        from utils.mlflow_helpers import log_metrics_dict, log_params_dict
        
        log_params_dict({'model_type': 'DeepNN', 'units': units, 'dropout_rate': dropout_rate})
        log_metrics_dict(metrics)
        mlflow.keras.log_model(model, "model")
    
    print(f"{'='*60}\n")
    return model, metrics


if __name__ == "__main__":
    print("🧪 Testing Keras Models\n")
    from utils.data_loader import load_customer_churn_data
    X_train, X_test, y_train, y_test = load_customer_churn_data()
    model, metrics = train_shallow_nn(X_train, y_train, X_test, y_test, epochs=10)
    print("\n✅ Keras models tested successfully!\n")
