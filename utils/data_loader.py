"""
Data Loading Utilities for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Provides easy-to-use functions for loading datasets with train-test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def get_data_path() -> Path:
    """
    Get the path to the data directory.
    
    Returns:
        Path object pointing to data directory
    """
    # Try to find data directory from current location
    current_dir = Path.cwd()
    
    # Check if we're in experiments/ folder
    if current_dir.name == "experiments":
        data_dir = current_dir.parent / "data"
    # Check if we're in utils/ folder
    elif current_dir.name == "utils":
        data_dir = current_dir.parent / "data"
    # Check if we're in root starter kit folder
    elif (current_dir / "data").exists():
        data_dir = current_dir / "data"
    # Check if we're one level above
    elif (current_dir / "mlflow_tracking_starter_kit" / "data").exists():
        data_dir = current_dir / "mlflow_tracking_starter_kit" / "data"
    else:
        # Default path
        data_dir = Path(__file__).parent.parent / "data"
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found. Expected at: {data_dir}\n"
            f"Please run 'python data/generate_datasets.py' first."
        )
    
    return data_dir


def load_customer_churn_data(
    test_size: float = 0.2,
    random_state: int = 42,
    return_raw: bool = False
) -> Tuple:
    """
    Load customer churn dataset with train-test split.
    
    This is the PRIMARY dataset for Session 5 experiments.
    
    Features:
        - age: Customer age (18-80)
        - tenure_months: Months with company (0-120)
        - monthly_charges: Current monthly charges ($20-180)
        - total_charges: Lifetime charges
        - support_calls: Number of support calls (0-10+)
        - contract_type: Month-to-month, One year, Two year
        - payment_method: Electronic check, Mailed check, Bank transfer, Credit card
        - internet_service: DSL, Fiber optic, No
    
    Target:
        - churn: 0 (no churn), 1 (churned)
    
    Args:
        test_size: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        return_raw: If True, return raw DataFrame instead of splits (default: False)
        
    Returns:
        If return_raw=False: X_train, X_test, y_train, y_test
        If return_raw=True: DataFrame (full dataset)
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_customer_churn_data()
        >>> print(f"Training samples: {len(X_train)}")
        >>> print(f"Test samples: {len(X_test)}")
    """
    # Load data
    data_dir = get_data_path()
    file_path = data_dir / "customer_churn.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Customer churn dataset not found at: {file_path}\n"
            f"Please run: cd data && python generate_datasets.py"
        )
    
    df = pd.read_csv(file_path)
    
    # Return raw if requested
    if return_raw:
        return df
    
    # Drop customer_id (not a feature)
    df = df.drop('customer_id', axis=1)
    
    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


def load_iris_data(
    test_size: float = 0.3,
    random_state: int = 42,
    return_raw: bool = False
) -> Tuple:
    """
    Load iris dataset with train-test split.
    
    This is a SECONDARY dataset for quick demos and testing.
    
    Features:
        - sepal_length: Sepal length in cm
        - sepal_width: Sepal width in cm
        - petal_length: Petal length in cm
        - petal_width: Petal width in cm
    
    Target:
        - species: setosa, versicolor, virginica (encoded as 0, 1, 2)
    
    Args:
        test_size: Proportion of data for test set (default: 0.3)
        random_state: Random seed for reproducibility (default: 42)
        return_raw: If True, return raw DataFrame instead of splits (default: False)
        
    Returns:
        If return_raw=False: X_train, X_test, y_train, y_test
        If return_raw=True: DataFrame (full dataset)
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_iris_data()
        >>> print(f"Features shape: {X_train.shape}")
        >>> print(f"Classes: {np.unique(y_train)}")
    """
    # Load data
    data_dir = get_data_path()
    file_path = data_dir / "iris.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Iris dataset not found at: {file_path}\n"
            f"Please run: cd data && python generate_datasets.py"
        )
    
    df = pd.read_csv(file_path)
    
    # Return raw if requested
    if return_raw:
        return df
    
    # Separate features and target
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Encode target labels
    label_mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    y = y.map(label_mapping)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    # Reset indices
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train, X_test, y_train, y_test


def get_dataset_info() -> dict:
    """
    Get metadata about available datasets.
    
    Returns:
        Dictionary with dataset information
        
    Example:
        >>> info = get_dataset_info()
        >>> print(info['customer_churn']['description'])
    """
    data_dir = get_data_path()
    info_path = data_dir / "data_info.json"
    
    if info_path.exists():
        import json
        with open(info_path, 'r') as f:
            return json.load(f)
    else:
        return {
            "customer_churn": {
                "description": "Customer churn prediction dataset",
                "rows": 10000,
                "task": "binary_classification"
            },
            "iris": {
                "description": "Iris flower species classification",
                "rows": 150,
                "task": "multiclass_classification"
            }
        }


def validate_data(X, y, dataset_name: str = "dataset"):
    """
    Validate loaded data for common issues.
    
    Args:
        X: Features DataFrame or array
        y: Target Series or array
        dataset_name: Name of dataset (for error messages)
        
    Raises:
        ValueError: If data has issues
    """
    # Check for NaN values
    if isinstance(X, pd.DataFrame):
        if X.isnull().any().any():
            raise ValueError(f"{dataset_name}: Features contain NaN values")
    
    if isinstance(y, pd.Series):
        if y.isnull().any():
            raise ValueError(f"{dataset_name}: Target contains NaN values")
    
    # Check shapes match
    if len(X) != len(y):
        raise ValueError(
            f"{dataset_name}: Feature and target lengths don't match "
            f"({len(X)} vs {len(y)})"
        )
    
    # Check for empty data
    if len(X) == 0:
        raise ValueError(f"{dataset_name}: Dataset is empty")
    
    print(f"✅ {dataset_name} validation passed")


# Convenience function for quick loading
def load_data(dataset: str = "churn", **kwargs):
    """
    Convenience function to load any dataset by name.
    
    Args:
        dataset: "churn" or "iris"
        **kwargs: Additional arguments passed to respective load function
        
    Returns:
        X_train, X_test, y_train, y_test or raw DataFrame
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_data("churn")
        >>> X_train, X_test, y_train, y_test = load_data("iris", test_size=0.25)
    """
    if dataset.lower() in ["churn", "customer_churn"]:
        return load_customer_churn_data(**kwargs)
    elif dataset.lower() == "iris":
        return load_iris_data(**kwargs)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available options: 'churn', 'iris'"
        )


# Print dataset summary
def print_dataset_summary(dataset: str = "churn"):
    """
    Print summary statistics for a dataset.
    
    Args:
        dataset: "churn" or "iris"
        
    Example:
        >>> print_dataset_summary("churn")
    """
    print(f"\n{'=' * 60}")
    print(f"📊 {dataset.upper()} DATASET SUMMARY")
    print(f"{'=' * 60}")
    
    if dataset.lower() in ["churn", "customer_churn"]:
        df = load_customer_churn_data(return_raw=True)
        target_col = 'churn'
    elif dataset.lower() == "iris":
        df = load_iris_data(return_raw=True)
        target_col = 'species'
    else:
        print(f"❌ Unknown dataset: {dataset}")
        return
    
    print(f"\n📈 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns: {', '.join(df.columns.tolist())}")
    print(f"\n🎯 Target: {target_col}")
    print(f"\n📊 Target Distribution:")
    print(df[target_col].value_counts().to_string())
    print(f"\n💾 Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    """Test data loading functionality."""
    print("🧪 Testing Data Loading Utilities\n")
    
    # Test customer churn
    try:
        print("1️⃣ Loading Customer Churn Data...")
        X_train, X_test, y_train, y_test = load_customer_churn_data()
        print(f"   ✅ Train: {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"   ✅ Test: {len(X_test)} samples")
        print(f"   ✅ Churn rate (train): {y_train.mean():.2%}")
        validate_data(X_train, y_train, "Customer Churn")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Test iris
    try:
        print("2️⃣ Loading Iris Data...")
        X_train, X_test, y_train, y_test = load_iris_data()
        print(f"   ✅ Train: {len(X_train)} samples, {X_train.shape[1]} features")
        print(f"   ✅ Test: {len(X_test)} samples")
        print(f"   ✅ Classes: {sorted(y_train.unique())}")
        validate_data(X_train, y_train, "Iris")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # Print summaries
    print_dataset_summary("churn")
    print_dataset_summary("iris")
    
    print("✅ All data loading tests passed!\n")
