"""
Model Configuration Files for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

Pre-defined hyperparameter configurations for all models.
Students can modify these dictionaries and run experiments.
"""

# =============================================================================
# RANDOM FOREST CONFIGURATIONS
# =============================================================================

RANDOM_FOREST_CONFIGS = {
    'rf_baseline': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'rf_shallow': {
        'n_estimators': 50,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'rf_deep': {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42
    },
    'rf_many_trees': {
        'n_estimators': 500,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'random_state': 42
    }
}

# =============================================================================
# LOGISTIC REGRESSION CONFIGURATIONS
# =============================================================================

LOGISTIC_REGRESSION_CONFIGS = {
    'lr_baseline': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    },
    'lr_strong_reg': {
        'C': 0.1,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    },
    'lr_weak_reg': {
        'C': 10.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    }
}

# =============================================================================
# SVM CONFIGURATIONS
# =============================================================================

SVM_CONFIGS = {
    'svm_rbf': {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': 42
    },
    'svm_linear': {
        'C': 1.0,
        'kernel': 'linear',
        'gamma': 'scale',
        'random_state': 42
    },
    'svm_poly': {
        'C': 1.0,
        'kernel': 'poly',
        'gamma': 'scale',
        'random_state': 42
    }
}

# =============================================================================
# GRADIENT BOOSTING CONFIGURATIONS
# =============================================================================

GRADIENT_BOOSTING_CONFIGS = {
    'gb_baseline': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 1.0,
        'random_state': 42
    },
    'gb_fast': {
        'n_estimators': 50,
        'learning_rate': 0.2,
        'max_depth': 3,
        'subsample': 0.8,
        'random_state': 42
    },
    'gb_accurate': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'subsample': 0.9,
        'random_state': 42
    }
}

# =============================================================================
# XGBOOST CONFIGURATIONS
# =============================================================================

XGBOOST_CONFIGS = {
    'xgb_baseline': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.3,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_state': 42
    },
    'xgb_regularized': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'random_state': 42
    },
    'xgb_deep': {
        'n_estimators': 150,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42
    }
}

# =============================================================================
# LIGHTGBM CONFIGURATIONS
# =============================================================================

LIGHTGBM_CONFIGS = {
    'lgbm_baseline': {
        'n_estimators': 100,
        'max_depth': -1,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_state': 42
    },
    'lgbm_fast': {
        'n_estimators': 50,
        'max_depth': -1,
        'learning_rate': 0.2,
        'num_leaves': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    },
    'lgbm_accurate': {
        'n_estimators': 200,
        'max_depth': -1,
        'learning_rate': 0.05,
        'num_leaves': 50,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'random_state': 42
    }
}

# =============================================================================
# KERAS NEURAL NETWORK CONFIGURATIONS
# =============================================================================

KERAS_NN_CONFIGS = {
    'nn_small': {
        'units': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'epochs': 30,
        'batch_size': 32,
        'random_state': 42
    },
    'nn_medium': {
        'units': 64,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32,
        'random_state': 42
    },
    'nn_large': {
        'units': 128,
        'dropout_rate': 0.4,
        'learning_rate': 0.0005,
        'epochs': 100,
        'batch_size': 64,
        'random_state': 42
    }
}

# =============================================================================
# GRID SEARCH PARAMETER GRIDS
# =============================================================================

GRID_SEARCH_PARAMS = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    'xgboost': {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'lightgbm': {
        'n_estimators': [50, 100, 200],
        'num_leaves': [20, 31, 50],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(model_type, config_name):
    """
    Get a specific model configuration.
    
    Args:
        model_type: 'rf', 'lr', 'svm', 'gb', 'xgb', 'lgbm', or 'nn'
        config_name: Name of the configuration
        
    Returns:
        Dictionary of hyperparameters
        
    Example:
        >>> config = get_config('rf', 'rf_baseline')
        >>> print(config)
    """
    config_maps = {
        'rf': RANDOM_FOREST_CONFIGS,
        'lr': LOGISTIC_REGRESSION_CONFIGS,
        'svm': SVM_CONFIGS,
        'gb': GRADIENT_BOOSTING_CONFIGS,
        'xgb': XGBOOST_CONFIGS,
        'lgbm': LIGHTGBM_CONFIGS,
        'nn': KERAS_NN_CONFIGS
    }
    
    if model_type not in config_maps:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if config_name not in config_maps[model_type]:
        raise ValueError(f"Unknown config_name: {config_name}")
    
    return config_maps[model_type][config_name]


def list_configs(model_type):
    """
    List all available configurations for a model type.
    
    Args:
        model_type: 'rf', 'lr', 'svm', 'gb', 'xgb', 'lgbm', or 'nn'
        
    Returns:
        List of configuration names
    """
    config_maps = {
        'rf': RANDOM_FOREST_CONFIGS,
        'lr': LOGISTIC_REGRESSION_CONFIGS,
        'svm': SVM_CONFIGS,
        'gb': GRADIENT_BOOSTING_CONFIGS,
        'xgb': XGBOOST_CONFIGS,
        'lgbm': LIGHTGBM_CONFIGS,
        'nn': KERAS_NN_CONFIGS
    }
    
    if model_type not in config_maps:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    return list(config_maps[model_type].keys())


if __name__ == "__main__":
    print("🧪 Testing Model Configurations\n")
    
    print("1️⃣ Random Forest Configurations:")
    for name in list_configs('rf'):
        print(f"   - {name}")
    
    print("\n2️⃣ Get Specific Config:")
    config = get_config('rf', 'rf_baseline')
    print(f"   rf_baseline: {config}")
    
    print("\n✅ All configurations loaded successfully!\n")
