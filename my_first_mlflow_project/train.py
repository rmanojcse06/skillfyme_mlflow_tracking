"""Training script for MLflow Project
Adapted from Session 5 Random Forest code
"""

import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def train_model(n_estimators, max_depth, random_state):
    """Train Random Forest classifier"""

    print("🎯 Training Random Forest Classifier")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")
    print(f"   random_state: {random_state}")

    # Generate synthetic data (replace with your data)
    print("📊 Loading data...")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    # Train model
    print("🌲 Training model...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    print("📈 Evaluating...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Log to MLflow
    print("📝 Logging to MLflow...")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")

    print("✅ Model logged to MLflow!")

    return model, {"accuracy": accuracy, "f1_score": f1}

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    # Start MLflow run
    with mlflow.start_run():
        train_model(args.n_estimators, args.max_depth, args.random_state)

    print("🎉 Training complete!")
