"""
MLflow Setup Script
MLflow Tracking Starter Kit - Session 5

Initializes MLflow tracking and creates a test experiment to verify setup.

Usage:
    python setup_mlflow.py
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

print("=" * 70)
print("🔧 MLflow Tracking Setup - Starter Kit Initialization")
print("=" * 70)

# Step 1: Set tracking URI
print("\n1️⃣ Setting up MLflow tracking URI...")
tracking_uri = "file:./mlruns"
mlflow.set_tracking_uri(tracking_uri)
print(f"   ✅ Tracking URI set to: {tracking_uri}")

# Verify directory exists
mlruns_path = Path("./mlruns")
if not mlruns_path.exists():
    mlruns_path.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Created mlruns directory")
else:
    print(f"   ✅ mlruns directory already exists")

# Step 2: Create default experiments
print("\n2️⃣ Creating default experiments...")

experiments = [
    "Starter Kit Test Experiment",
    "Baseline_Experiments",
    "Hyperparameter_Tuning",
    "Model_Comparison",
    "Feature_Engineering"
]

for exp_name in experiments:
    try:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = mlflow.create_experiment(exp_name)
            print(f"   ✅ Created experiment: {exp_name} (ID: {exp_id})")
        else:
            print(f"   ℹ️  Experiment already exists: {exp_name}")
    except Exception as e:
        print(f"   ⚠️  Error creating {exp_name}: {e}")

# Step 3: Run a test experiment
print("\n3️⃣ Running test experiment to verify setup...")

mlflow.set_experiment("Starter Kit Test Experiment")

# Generate synthetic test data
X, y = make_classification(
    n_samples=100,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
with mlflow.start_run(run_name="Test_Run"):
    # Log parameters
    mlflow.log_param("test_param", "test_value")
    mlflow.log_param("n_estimators", 10)
    mlflow.log_param("random_state", 42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Log metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)
    
    # Log model
    mlflow.sklearn.log_model(model, "test_model")
    
    # Log tags
    mlflow.set_tag("test", "true")
    mlflow.set_tag("setup_verification", "passed")
    
    print(f"   ✅ Test run completed")
    print(f"   📊 Train accuracy: {train_score:.4f}")
    print(f"   📊 Test accuracy: {test_score:.4f}")

# Step 4: Verify setup
print("\n4️⃣ Verifying MLflow setup...")

try:
    # List all experiments
    experiments = mlflow.search_experiments()
    print(f"   ✅ Found {len(experiments)} experiments")
    
    # Get runs from test experiment
    test_exp = mlflow.get_experiment_by_name("Starter Kit Test Experiment")
    if test_exp:
        runs = mlflow.search_runs(experiment_ids=[test_exp.experiment_id])
        print(f"   ✅ Found {len(runs)} runs in test experiment")
    
    print("\n" + "=" * 70)
    print("✅ MLFLOW SETUP COMPLETE!")
    print("=" * 70)
    
    print("\n🎉 Success! Your MLflow tracking is ready to use.")
    
    print("\n📋 What was created:")
    print("   ✓ MLflow tracking directory (./mlruns)")
    print("   ✓ 5 pre-configured experiments")
    print("   ✓ 1 test run with logged parameters, metrics, and model")
    
    print("\n🚀 Next Steps:")
    print("\n   1. Start MLflow UI:")
    print("      $ mlflow ui")
    print("\n   2. Open browser to:")
    print("      http://127.0.0.1:5000")
    print("\n   3. You should see:")
    print("      - 5 experiments in the sidebar")
    print("      - 1 test run in 'Starter Kit Test Experiment'")
    print("\n   4. Explore the test run:")
    print("      - View logged parameters")
    print("      - View logged metrics")
    print("      - View logged model")
    
    print("\n💡 Quick Test:")
    print("   Run a baseline experiment:")
    print("   $ cd experiments")
    print("   $ python baseline_experiments.py")
    
    print("\n📚 Documentation:")
    print("   See README.md for full usage instructions")
    
    print("\n" + "=" * 70)

except Exception as e:
    print(f"\n❌ ERROR: Setup verification failed")
    print(f"   Error details: {e}")
    print("\n   Troubleshooting:")
    print("   1. Check if mlruns/ directory is writable")
    print("   2. Try running: rm -rf mlruns/ && python setup_mlflow.py")
    print("   3. Check Python environment has MLflow installed")
