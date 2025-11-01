"""
Data Preprocessing Utilities for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Provides preprocessing pipelines for feature scaling, encoding, and transformation.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Flexible data preprocessing pipeline for ML experiments.
    
    Supports:
    - Numeric feature scaling (StandardScaler, MinMaxScaler)
    - Categorical feature encoding (One-hot encoding)
    - Missing value handling
    - Feature selection
    
    Example:
        >>> preprocessor = DataPreprocessor(scaler_type='standard')
        >>> X_train_scaled = preprocessor.fit_transform(X_train)
        >>> X_test_scaled = preprocessor.transform(X_test)
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        handle_missing: bool = True,
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: 'standard', 'minmax', or 'none'
            handle_missing: Whether to handle missing values
            feature_names: List of feature names (optional)
        """
        self.scaler_type = scaler_type.lower()
        self.handle_missing = handle_missing
        self.feature_names = feature_names
        
        # Initialize scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError(
                f"Unknown scaler_type: {scaler_type}. "
                f"Use 'standard', 'minmax', or 'none'"
            )
        
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit the preprocessor on training data.
        
        Args:
            X: Training features (DataFrame or array)
            y: Training target (ignored, for sklearn compatibility)
            
        Returns:
            self
        """
        X = self._to_array(X)
        
        # Handle missing values
        if self.handle_missing:
            X = self._handle_missing(X)
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(X)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Features to transform (DataFrame or array)
            
        Returns:
            Transformed features (array)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X = self._to_array(X)
        
        # Handle missing values
        if self.handle_missing:
            X = self._handle_missing(X)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform in one step.
        
        Args:
            X: Training features (DataFrame or array)
            y: Training target (ignored)
            
        Returns:
            Transformed features (array)
        """
        return self.fit(X, y).transform(X)
    
    def _to_array(self, X):
        """Convert DataFrame to array if needed."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return X
    
    def _handle_missing(self, X):
        """Handle missing values by filling with column means."""
        if np.isnan(X).any():
            col_means = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_means, inds[1])
        return X
    
    def get_params(self) -> dict:
        """
        Get preprocessor parameters (useful for MLflow logging).
        
        Returns:
            Dictionary of parameters
        """
        return {
            'scaler_type': self.scaler_type,
            'handle_missing': self.handle_missing,
            'is_fitted': self.is_fitted
        }


def create_preprocessing_pipeline(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scaler_type: str = 'standard'
) -> ColumnTransformer:
    """
    Create a sklearn preprocessing pipeline for mixed data types.
    
    This is useful when you have both numeric and categorical features
    and want to apply different transformations to each.
    
    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        scaler_type: 'standard' or 'minmax' for numeric features
        
    Returns:
        ColumnTransformer pipeline
        
    Example:
        >>> pipeline = create_preprocessing_pipeline(
        ...     numeric_features=['age', 'income'],
        ...     categorical_features=['gender', 'city'],
        ...     scaler_type='standard'
        ... )
        >>> X_transformed = pipeline.fit_transform(X_train)
    """
    from sklearn.preprocessing import OneHotEncoder
    
    transformers = []
    
    # Numeric transformation
    if numeric_features:
        if scaler_type == 'standard':
            numeric_transformer = StandardScaler()
        elif scaler_type == 'minmax':
            numeric_transformer = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        transformers.append(('num', numeric_transformer, numeric_features))
    
    # Categorical transformation
    if categorical_features:
        categorical_transformer = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        )
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    # Create pipeline
    pipeline = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'  # Keep other columns as-is
    )
    
    return pipeline


def scale_features(
    X_train,
    X_test,
    method: str = 'standard'
) -> tuple:
    """
    Scale features using specified method.
    
    Args:
        X_train: Training features
        X_test: Test features
        method: 'standard', 'minmax', or 'none'
        
    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler)
        
    Example:
        >>> X_train_scaled, X_test_scaled, scaler = scale_features(
        ...     X_train, X_test, method='standard'
        ... )
    """
    if method == 'none':
        return X_train, X_test, None
    
    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'standard', 'minmax', or 'none'"
        )
    
    # Fit on train, transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Preserve DataFrame structure if input was DataFrame
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_test.columns,
            index=X_test.index
        )
    
    return X_train_scaled, X_test_scaled, scaler


def select_top_k_features(
    X_train,
    X_test,
    y_train,
    k: int = 10,
    method: str = 'mutual_info'
):
    """
    Select top K features based on feature importance.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        k: Number of features to select
        method: 'mutual_info' or 'f_classif'
        
    Returns:
        Tuple of (X_train_selected, X_test_selected, selected_features)
        
    Example:
        >>> X_train_top, X_test_top, features = select_top_k_features(
        ...     X_train, X_test, y_train, k=5
        ... )
        >>> print(f"Selected features: {features}")
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
    
    # Choose scoring function
    if method == 'mutual_info':
        score_func = mutual_info_classif
    elif method == 'f_classif':
        score_func = f_classif
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Select features
    selector = SelectKBest(score_func=score_func, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Get selected feature names
    if isinstance(X_train, pd.DataFrame):
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        # Convert to DataFrame
        X_train_selected = pd.DataFrame(
            X_train_selected,
            columns=selected_features,
            index=X_train.index
        )
        X_test_selected = pd.DataFrame(
            X_test_selected,
            columns=selected_features,
            index=X_test.index
        )
    else:
        selected_features = [f"feature_{i}" for i in range(k)]
    
    return X_train_selected, X_test_selected, selected_features


def get_preprocessing_config(config_name: str = 'default') -> dict:
    """
    Get pre-defined preprocessing configurations.
    
    Args:
        config_name: 'default', 'minmax', 'no_scaling', or 'feature_selection'
        
    Returns:
        Dictionary of preprocessing parameters
        
    Example:
        >>> config = get_preprocessing_config('minmax')
        >>> preprocessor = DataPreprocessor(**config)
    """
    configs = {
        'default': {
            'scaler_type': 'standard',
            'handle_missing': True
        },
        'minmax': {
            'scaler_type': 'minmax',
            'handle_missing': True
        },
        'no_scaling': {
            'scaler_type': 'none',
            'handle_missing': True
        },
        'robust': {
            'scaler_type': 'standard',
            'handle_missing': True
        }
    }
    
    if config_name not in configs:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[config_name]


def encode_categorical_features(
    X_train,
    X_test,
    categorical_columns: List[str],
    encoding_type: str = 'onehot',
    handle_unknown: str = 'ignore'
):
    """
    Encode categorical features for machine learning.
    
    Args:
        X_train: Training features (DataFrame)
        X_test: Test features (DataFrame)
        categorical_columns: List of categorical column names
        encoding_type: 'onehot' or 'label' (default: 'onehot')
        handle_unknown: How to handle unknown categories (default: 'ignore')
        
    Returns:
        Tuple of (X_train_encoded, X_test_encoded, encoder)
        
    Example:
        >>> categorical_cols = ['contract_type', 'payment_method']
        >>> X_train_enc, X_test_enc, encoder = encode_categorical_features(
        ...     X_train, X_test, categorical_cols, encoding_type='onehot'
        ... )
    """
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder
    
    if encoding_type == 'onehot':
        # One-hot encoding
        encoder = OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown=handle_unknown
        )
        
        # Fit on training data
        encoder.fit(X_train[categorical_columns])
        
        # Transform both sets
        X_train_cat_encoded = encoder.transform(X_train[categorical_columns])
        X_test_cat_encoded = encoder.transform(X_test[categorical_columns])
        
        # Get feature names
        feature_names = encoder.get_feature_names_out(categorical_columns)
        
        # Create DataFrames for encoded features
        X_train_cat_df = pd.DataFrame(
            X_train_cat_encoded,
            columns=feature_names,
            index=X_train.index
        )
        X_test_cat_df = pd.DataFrame(
            X_test_cat_encoded,
            columns=feature_names,
            index=X_test.index
        )
        
        # Combine with non-categorical columns
        non_categorical_cols = [col for col in X_train.columns if col not in categorical_columns]
        
        if non_categorical_cols:
            X_train_encoded = pd.concat([X_train[non_categorical_cols], X_train_cat_df], axis=1)
            X_test_encoded = pd.concat([X_test[non_categorical_cols], X_test_cat_df], axis=1)
        else:
            X_train_encoded = X_train_cat_df
            X_test_encoded = X_test_cat_df
        
        return X_train_encoded, X_test_encoded, encoder
    
    elif encoding_type == 'label':
        # Label encoding (not recommended for tree-based models)
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        encoders = {}
        
        for col in categorical_columns:
            encoder = LabelEncoder()
            
            # Fit on training data
            X_train_encoded[col] = encoder.fit_transform(X_train[col])
            
            # Transform test data, handle unknown categories
            try:
                X_test_encoded[col] = encoder.transform(X_test[col])
            except ValueError:
                # Handle unknown categories by assigning -1
                X_test_encoded[col] = X_test[col].map(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
            
            encoders[col] = encoder
        
        return X_train_encoded, X_test_encoded, encoders
    
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}. Use 'onehot' or 'label'")


def get_categorical_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically detect categorical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of categorical column names
        
    Example:
        >>> cat_cols = get_categorical_columns(X_train)
        >>> print(f"Found {len(cat_cols)} categorical columns")
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return categorical_cols


def compare_scaling_methods(X_train, X_test, y_train, y_test, model_func):
    """
    Compare different scaling methods by training models.
    
    This is useful for experiments to see which scaling works best.
    
    Args:
        X_train, X_test: Features
        y_train, y_test: Targets
        model_func: Function that takes (X_train, y_train) and returns trained model
        
    Returns:
        Dictionary with results for each scaling method
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> def train_rf(X, y):
        ...     return RandomForestClassifier().fit(X, y)
        >>> results = compare_scaling_methods(X_train, X_test, y_train, y_test, train_rf)
    """
    from sklearn.metrics import accuracy_score
    
    results = {}
    
    for method in ['none', 'standard', 'minmax']:
        # Scale features
        X_train_scaled, X_test_scaled, _ = scale_features(
            X_train, X_test, method=method
        )
        
        # Train model
        model = model_func(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[method] = {
            'accuracy': accuracy,
            'model': model
        }
        
        print(f"   {method:12s} → Accuracy: {accuracy:.4f}")
    
    return results


if __name__ == "__main__":
    """Test preprocessing utilities."""
    print("🧪 Testing Preprocessing Utilities\n")
    
    # Create synthetic data
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(30, 5)
    y_train = np.random.randint(0, 2, 100)
    
    # Test 1: DataPreprocessor
    print("1️⃣ Testing DataPreprocessor...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    print(f"   ✅ Train shape: {X_train_scaled.shape}")
    print(f"   ✅ Test shape: {X_test_scaled.shape}")
    print(f"   ✅ Train mean: {X_train_scaled.mean(axis=0).mean():.6f}")
    print(f"   ✅ Train std: {X_train_scaled.std(axis=0).mean():.6f}")
    
    print()
    
    # Test 2: scale_features
    print("2️⃣ Testing scale_features...")
    X_train_std, X_test_std, scaler = scale_features(X_train, X_test, 'standard')
    print(f"   ✅ StandardScaler applied")
    print(f"   ✅ Train mean: {X_train_std.mean():.6f}")
    
    print()
    
    # Test 3: Different configs
    print("3️⃣ Testing preprocessing configs...")
    for config_name in ['default', 'minmax', 'no_scaling']:
        config = get_preprocessing_config(config_name)
        print(f"   ✅ {config_name}: {config}")
    
    print("\n✅ All preprocessing tests passed!\n")
