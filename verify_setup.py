"""
Starter Kit Verification Script
MLflow Tracking Starter Kit - Session 5

Tests all components of the starter kit to ensure everything works.

Usage:
    python verify_setup.py
"""

import sys
from pathlib import Path

print("=" * 70)
print("🔍 MLflow Tracking Starter Kit - Verification")
print("=" * 70)

# Check Python version
print("\n1️⃣ Checking Python version...")
if sys.version_info < (3, 9):
    print("   ❌ Python 3.9+ required")
    sys.exit(1)
print(f"   ✅ Python {sys.version.split()[0]}")

# Check required packages
print("\n2️⃣ Checking required packages...")
required_packages = [
    'mlflow', 'sklearn', 'pandas', 'numpy', 'matplotlib',
    'seaborn', 'xgboost', 'lightgbm', 'tensorflow'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        print(f"   ❌ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   ⚠️  Missing packages: {', '.join(missing_packages)}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Check datasets
print("\n3️⃣ Checking datasets...")
data_dir = Path("data")
datasets = ["customer_churn.csv", "iris.csv", "data_info.json"]

for dataset in datasets:
    if (data_dir / dataset).exists():
        print(f"   ✅ {dataset}")
    else:
        print(f"   ❌ {dataset} - NOT FOUND")
        print("   Run: cd data && python generate_datasets.py")

# Check directory structure
print("\n4️⃣ Checking directory structure...")
required_dirs = ["data", "models", "utils", "experiments"]
for dir_name in required_dirs:
    if Path(dir_name).exists():
        print(f"   ✅ {dir_name}/")
    else:
        print(f"   ❌ {dir_name}/ - NOT FOUND")

# Test imports
print("\n5️⃣ Testing module imports...")
try:
    from utils.data_loader import load_customer_churn_data
    print("   ✅ utils.data_loader")
    
    from utils.evaluator import evaluate_classifier
    print("   ✅ utils.evaluator")
    
    from utils.visualizer import plot_confusion_matrix
    print("   ✅ utils.visualizer")
    
    from models.sklearn_models import train_random_forest
    print("   ✅ models.sklearn_models")
    
except Exception as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test data loading
print("\n6️⃣ Testing data loading...")
try:
    X_train, X_test, y_train, y_test = load_customer_churn_data()
    print(f"   ✅ Loaded {len(X_train)} training samples")
    print(f"   ✅ Loaded {len(X_test)} test samples")
except Exception as e:
    print(f"   ❌ Data loading error: {e}")
    sys.exit(1)

# Test MLflow
print("\n7️⃣ Testing MLflow...")
try:
    import mlflow
    mlflow.set_tracking_uri("file:./mlruns")
    print("   ✅ MLflow tracking URI set")
    
    # Check if experiments exist
    exp = mlflow.get_experiment_by_name("Starter Kit Test Experiment")
    if exp:
        print("   ✅ Test experiment found")
    else:
        print("   ⚠️  Test experiment not found (run setup_mlflow.py)")
        
except Exception as e:
    print(f"   ❌ MLflow error: {e}")

print("\n" + "=" * 70)
print("✅ VERIFICATION COMPLETE!")
print("=" * 70)

print("\n🎉 Your starter kit is ready to use!")
print("\n📚 Quick Start:")
print("   1. Run setup: python setup_mlflow.py")
print("   2. Start UI: mlflow ui")
print("   3. Run experiment: python experiments/baseline_experiments.py")

print("\n" + "=" * 70)
