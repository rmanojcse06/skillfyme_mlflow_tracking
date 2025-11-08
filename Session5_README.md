# 🚀 MLflow Tracking Starter Kit

**Session 5: Model Tracking with MLflow (Intermediate)**  
*MLOps with Agentic AI - Advanced Certification Course*

---

## 📋 Overview

This starter kit provides **production-ready ML code** for learning advanced MLflow tracking. You'll focus on **tracking, organizing, and comparing experiments** rather than building models from scratch.

### What's Included

- **4 ML frameworks**: scikit-learn, XGBoost, LightGBM, Keras
- **10+ pre-built models**: Ready to train and track
- **2 datasets**: Customer churn (10K rows) + Iris
- **Utility modules**: Data loading, preprocessing, evaluation, visualization
- **Experiment scripts**: Baseline, hyperparameter tuning, model comparison

---

## 🎯 Learning Objectives

By using this starter kit, you will master:

✅ **Advanced MLflow Tracking API** (autolog, nested runs, artifacts)  
✅ **MLflow UI Navigation** (filtering, comparing, organizing)  
✅ **Production-Grade Patterns** (team collaboration, reproducibility)  
✅ **Multi-Framework Integration** (sklearn, XGBoost, LightGBM, Keras)

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.9, 3.10, or 3.11 (3.10 recommended)
- pip package manager
- 500 MB free disk space

### Step 1: Clone/Download Starter Kit

```bash
# If you received this as a ZIP file, extract it
# If in a Git repo, clone it
cd mlflow_tracking_starter_kit
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv mlflow_env

# Activate it
# On Windows:
mlflow_env\Scripts\activate

# On Mac/Linux:
source mlflow_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected installation time**: 5-10 minutes (depending on internet speed)

### Step 4: Generate Datasets

```bash
cd data
python generate_datasets.py
cd ..
```

**Output**: Creates `customer_churn.csv` and `iris.csv` in the `data/` folder

### Step 5: Initialize MLflow

```bash
python setup_mlflow.py
```

**Output**: 
- Creates MLflow tracking directory (`mlruns/`)
- Sets up experiment structure
- Runs a test experiment to verify setup

### Step 6: Verify Setup

```bash
# Start MLflow UI
mlflow ui
```

**Expected behavior**:
- Opens UI at `http://127.0.0.1:5000`
- Shows "Starter Kit Test Experiment"
- Contains at least 1 test run

**✅ If you see the above, your setup is complete!**

---

## 📁 Folder Structure

```
mlflow_tracking_starter_kit/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup_mlflow.py                   # MLflow initialization script
│
├── data/                             
│   ├── generate_datasets.py          # Script to generate synthetic data
│   ├── customer_churn.csv            # Main dataset (generated)
│   ├── iris.csv                      # Secondary dataset (generated)
│   └── data_info.json                # Dataset metadata
│
├── models/                           
│   ├── sklearn_models.py             # RandomForest, LogReg, SVM, GradientBoosting
│   ├── xgboost_models.py            # XGBoost variants
│   ├── lightgbm_models.py           # LightGBM models
│   ├── keras_models.py              # Neural network architectures
│   └── model_configs.py             # Hyperparameter configurations
│
├── utils/                            
│   ├── data_loader.py               # Data loading utilities
│   ├── preprocessor.py              # Preprocessing pipelines
│   ├── evaluator.py                 # Model evaluation
│   ├── visualizer.py                # Plot generation
│   └── mlflow_helpers.py            # Custom MLflow utilities
│
└── experiments/                      
    ├── baseline_experiments.py       # Simple tracking examples
    ├── hyperparameter_tuning.py     # Grid/Random search
    ├── model_comparison.py          # Compare algorithms
    └── feature_engineering.py       # Track feature experiments
```

---

## 🚀 Quick Start Guide

### Run Your First Experiment (3 Commands)

```bash
# 1. Navigate to experiments folder
cd experiments

# 2. Run a baseline experiment
python baseline_experiments.py

# 3. View results in MLflow UI
mlflow ui
```

**What happens**:
1. Trains a Random Forest model on customer churn data
2. Logs parameters, metrics, and artifacts to MLflow
3. Creates visualizations (confusion matrix, ROC curve)

**Check MLflow UI**: You'll see a new run with metrics, parameters, and plots!

---

## 🎓 How to Use This Starter Kit

### Philosophy: Focus on MLflow, Not Model Code

> **You should NOT modify model code.**  
> **You SHOULD modify hyperparameters and tracking configurations.**

### Typical Workflow

1. **Choose a model** from `models/` folder
2. **Modify hyperparameters** in `model_configs.py`
3. **Run experiment** using scripts in `experiments/`
4. **Explore MLflow UI** to compare runs
5. **Iterate** with different configurations

### Example: Train Multiple Random Forest Variants

```python
from models.sklearn_models import train_random_forest
from utils.data_loader import load_customer_churn_data
import mlflow

# Load data
X_train, X_test, y_train, y_test = load_customer_churn_data()

# Set MLflow experiment
mlflow.set_experiment("RF_Hyperparameter_Tuning")

# Try different n_estimators
for n_est in [50, 100, 200, 500]:
    with mlflow.start_run(run_name=f"RF_n{n_est}"):
        model, metrics = train_random_forest(
            X_train, y_train, X_test, y_test,
            n_estimators=n_est,
            max_depth=10,
            random_state=42
        )
        
        # MLflow logging happens inside the function
        print(f"✅ Logged RF with n_estimators={n_est}")
```

**Now check MLflow UI**: Compare all 4 runs side-by-side!

---

## 📊 Available Models

### scikit-learn Models (`models/sklearn_models.py`)

| Function | Model | Key Hyperparameters |
|----------|-------|---------------------|
| `train_random_forest()` | RandomForestClassifier | n_estimators, max_depth, min_samples_split |
| `train_logistic_regression()` | LogisticRegression | C, penalty, solver |
| `train_svm()` | SVC | C, kernel, gamma |
| `train_gradient_boosting()` | GradientBoostingClassifier | n_estimators, learning_rate, max_depth |

### XGBoost Models (`models/xgboost_models.py`)

| Function | Key Hyperparameters |
|----------|---------------------|
| `train_xgboost_classifier()` | n_estimators, max_depth, learning_rate, subsample |

### LightGBM Models (`models/lightgbm_models.py`)

| Function | Key Hyperparameters |
|----------|---------------------|
| `train_lightgbm_classifier()` | n_estimators, max_depth, learning_rate, num_leaves |

### Keras Models (`models/keras_models.py`)

| Function | Architecture | Key Hyperparameters |
|----------|--------------|---------------------|
| `train_shallow_nn()` | 2 hidden layers | units, dropout_rate, learning_rate |
| `train_deep_nn()` | 4 hidden layers | units, dropout_rate, learning_rate |
| `train_wide_nn()` | 1 wide hidden layer | units, dropout_rate, learning_rate |

---

## 🛠️ Utility Modules

### `utils/data_loader.py`

```python
from utils.data_loader import load_customer_churn_data, load_iris_data

# Load customer churn data (train/test split included)
X_train, X_test, y_train, y_test = load_customer_churn_data(test_size=0.2)

# Load iris data
X_train, X_test, y_train, y_test = load_iris_data(test_size=0.3)
```

### `utils/preprocessor.py`

```python
from utils.preprocessor import create_preprocessing_pipeline

# Create a preprocessing pipeline
pipeline = create_preprocessing_pipeline(
    numeric_features=['age', 'tenure_months'],
    categorical_features=['contract_type', 'payment_method'],
    scaler_type='standard'  # or 'minmax'
)

X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

### `utils/evaluator.py`

```python
from utils.evaluator import evaluate_classifier

# Get all metrics
metrics = evaluate_classifier(y_test, y_pred, y_pred_proba)
# Returns: {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.79, ...}
```

### `utils/visualizer.py`

```python
from utils.visualizer import plot_confusion_matrix, plot_roc_curve

# Generate plots
cm_fig = plot_confusion_matrix(y_test, y_pred, class_names=['No Churn', 'Churn'])
roc_fig = plot_roc_curve(y_test, y_pred_proba)

# Log to MLflow
mlflow.log_figure(cm_fig, "confusion_matrix.png")
mlflow.log_figure(roc_fig, "roc_curve.png")
```

### `utils/mlflow_helpers.py`

```python
from utils.mlflow_helpers import log_metrics_dict, log_model_signature

# Log multiple metrics at once
log_metrics_dict(metrics)

# Log model with input/output signature
log_model_signature(model, X_train, y_train)
```

---

## 🧪 Experiment Scripts

### 1. `baseline_experiments.py`

**Purpose**: Learn basic MLflow tracking with single model runs

**What it does**:
- Trains Random Forest on customer churn data
- Logs parameters, metrics, and artifacts
- Good for understanding tracking fundamentals

**Run it**:
```bash
cd experiments
python baseline_experiments.py
```

### 2. `hyperparameter_tuning.py`

**Purpose**: Track hyperparameter tuning experiments with nested runs

**What it does**:
- Runs Grid Search for Random Forest
- Creates parent run for the search
- Creates child runs for each hyperparameter combination
- Logs best parameters and model

**Run it**:
```bash
python hyperparameter_tuning.py
```

### 3. `model_comparison.py`

**Purpose**: Compare different algorithms in a single experiment

**What it does**:
- Trains 5 different models (RF, LogReg, XGBoost, LightGBM, Keras)
- Logs all to same experiment
- Easy to compare in MLflow UI

**Run it**:
```bash
python model_comparison.py
```

### 4. `feature_engineering.py`

**Purpose**: Track impact of different preprocessing strategies

**What it does**:
- Tests StandardScaler vs MinMaxScaler
- Tests different feature selection methods
- Logs preprocessing pipeline as artifact

**Run it**:
```bash
python feature_engineering.py
```

---

## 🎨 MLflow UI Tips

### Start the UI

```bash
# From starter kit root directory
mlflow ui

# Custom port
mlflow ui --port 5001

# Different tracking URI
mlflow ui --backend-store-uri file:///path/to/mlruns
```

### Key UI Features You'll Master

1. **Filtering Runs**: `metrics.accuracy > 0.8`
2. **Comparing Runs**: Select multiple runs → Compare
3. **Parallel Coordinates**: Visualize hyperparameter relationships
4. **Sorting**: Click column headers to sort
5. **Search**: Use experiment search bar

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'mlflow'`

**Solution**: Install requirements
```bash
pip install -r requirements.txt
```

---

### Issue: `FileNotFoundError: data/customer_churn.csv`

**Solution**: Generate datasets
```bash
cd data
python generate_datasets.py
cd ..
```

---

### Issue: MLflow UI shows "No experiments"

**Solution**: Run setup script
```bash
python setup_mlflow.py
```

---

### Issue: `ImportError: cannot import name 'load_customer_churn_data'`

**Solution**: Make sure you're running scripts from the correct directory
```bash
# Should be in starter kit root or experiments/ folder
cd mlflow_tracking_starter_kit
python experiments/baseline_experiments.py
```

---

### Issue: TensorFlow warnings about GPU

**Solution**: These are normal. TensorFlow will use CPU if no GPU available
```bash
# To suppress warnings (optional)
export TF_CPP_MIN_LOG_LEVEL=2  # Mac/Linux
set TF_CPP_MIN_LOG_LEVEL=2     # Windows
```

---

### Issue: Port 5000 already in use

**Solution**: Use a different port
```bash
mlflow ui --port 5001
```

---

## 🎯 Session 5 Learning Path

### Phase 1: Setup & Verification (15 min)
- Install dependencies
- Generate datasets
- Run baseline experiment
- Verify MLflow UI works

### Phase 2: Basic Tracking (30 min)
- Run `baseline_experiments.py`
- Explore MLflow UI
- Understand logged artifacts

### Phase 3: Advanced Tracking (45 min)
- Run `hyperparameter_tuning.py`
- Explore nested runs
- Use UI filtering and comparison

### Phase 4: Multi-Model Experiments (30 min)
- Run `model_comparison.py`
- Compare different algorithms
- Find best model using UI

### Phase 5: Production Patterns (45 min)
- Run `feature_engineering.py`
- Track preprocessing pipelines
- Organize experiments with tags

### Phase 6: Mastery & Practice (45 min)
- Create your own experiments
- Modify hyperparameters
- Practice UI navigation

---

## 📚 Next Steps (Session 6 Preview)

Everything you track in Session 5 will be used in Session 6:

- **Tracked experiments** → MLflow Projects
- **Best models** → Model Registry
- **Hyperparameter configs** → Project entry points
- **Run IDs** → Model registration

**Save your best run_id** from Session 5 - you'll need it in Session 6!

---

## 🙋 Need Help?

### Common Commands

```bash
# View MLflow help
mlflow --help

# List experiments
mlflow experiments list

# Search runs
mlflow runs search --experiment-id 1

# Delete experiment
mlflow experiments delete --experiment-id 1
```

### Resources

- **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
- **Tracking API Reference**: https://mlflow.org/docs/latest/tracking.html
- **Python API Docs**: https://mlflow.org/docs/latest/python_api/index.html

---

## ✅ Pre-Session Checklist

Before Session 5 starts, ensure:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] `pip install -r requirements.txt` completed successfully
- [ ] Datasets generated (`customer_churn.csv` and `iris.csv` exist)
- [ ] `python setup_mlflow.py` runs without errors
- [ ] MLflow UI opens at `http://127.0.0.1:5000`
- [ ] At least 1 test run visible in UI

**🎉 If all checked, you're ready for Session 5!**

---

## 📝 License & Usage

This starter kit is part of the **MLOps with Agentic AI** course materials.  
For educational purposes only.

**Version**: 1.0  
**Last Updated**: October 2025  
**Session**: 5 - Model Tracking with MLflow (Intermediate)

---

**Happy Tracking! 🚀**
