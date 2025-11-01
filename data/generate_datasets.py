"""
Generate Synthetic Datasets for MLflow Tracking Starter Kit
Session 5: Model Tracking with MLflow (Intermediate)

This script generates:
1. customer_churn.csv - Main dataset for classification (10,000 rows)
2. iris.csv - Classic dataset for quick demos (150 rows)
3. data_info.json - Metadata about the datasets

Usage:
    python generate_datasets.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("🔄 Generating Synthetic Datasets for MLflow Starter Kit")
print("=" * 60)


def generate_customer_churn_data(n_samples=10000):
    """
    Generate synthetic customer churn dataset.
    
    Features realistic patterns:
    - Longer tenure reduces churn
    - Higher support calls increase churn
    - Month-to-month contracts have higher churn
    - Younger customers churn more
    
    Args:
        n_samples: Number of customers to generate
        
    Returns:
        DataFrame with customer features and churn target
    """
    print(f"\n📊 Generating Customer Churn Dataset ({n_samples:,} rows)...")
    
    # Customer demographics
    customer_id = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    age = np.random.normal(45, 15, n_samples).clip(18, 80).astype(int)
    
    # Account features
    tenure_months = np.random.exponential(24, n_samples).clip(0, 120).astype(int)
    
    # Charges (correlated with tenure)
    base_monthly_charge = np.random.uniform(20, 120, n_samples)
    monthly_charges = base_monthly_charge + (tenure_months * 0.5)
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 100, n_samples)
    total_charges = total_charges.clip(0)
    
    # Support interaction
    support_calls = np.random.poisson(2, n_samples)
    
    # Contract type (month-to-month, one year, two year)
    contract_probs = [0.5, 0.3, 0.2]
    contract_type = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        size=n_samples,
        p=contract_probs
    )
    
    # Payment method
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        size=n_samples,
        p=[0.35, 0.25, 0.25, 0.15]
    )
    
    # Internet service
    internet_service = np.random.choice(
        ['DSL', 'Fiber optic', 'No'],
        size=n_samples,
        p=[0.4, 0.4, 0.2]
    )
    
    # Generate churn with realistic patterns
    churn_probability = np.zeros(n_samples)
    
    # Base probability
    churn_probability += 0.2
    
    # Tenure effect (longer tenure = less likely to churn)
    churn_probability -= (tenure_months / 120) * 0.3
    
    # Age effect (younger customers churn more)
    churn_probability += ((80 - age) / 62) * 0.15
    
    # Support calls effect (more calls = more likely to churn)
    churn_probability += (support_calls / 10) * 0.25
    
    # Contract type effect
    contract_effect = {
        'Month-to-month': 0.25,
        'One year': 0.0,
        'Two year': -0.15
    }
    for i, ct in enumerate(contract_type):
        churn_probability[i] += contract_effect[ct]
    
    # Payment method effect
    payment_effect = {
        'Electronic check': 0.15,
        'Mailed check': 0.0,
        'Bank transfer': -0.05,
        'Credit card': -0.10
    }
    for i, pm in enumerate(payment_method):
        churn_probability[i] += payment_effect[pm]
    
    # Monthly charges effect (higher charges = more churn)
    churn_probability += ((monthly_charges - 50) / 100) * 0.1
    
    # Clip probabilities to valid range
    churn_probability = churn_probability.clip(0.05, 0.95)
    
    # Generate actual churn labels
    churn = (np.random.random(n_samples) < churn_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_id,
        'age': age,
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges.round(2),
        'total_charges': total_charges.round(2),
        'support_calls': support_calls,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'internet_service': internet_service,
        'churn': churn
    })
    
    # Print statistics
    print(f"   ✅ Generated {len(df):,} customers")
    print(f"   📈 Churn rate: {churn.mean():.2%}")
    print(f"   📊 Feature statistics:")
    print(f"      - Age: {age.min()}-{age.max()} years (mean: {age.mean():.1f})")
    print(f"      - Tenure: {tenure_months.min()}-{tenure_months.max()} months (mean: {tenure_months.mean():.1f})")
    print(f"      - Monthly charges: ${monthly_charges.min():.2f}-${monthly_charges.max():.2f}")
    print(f"      - Support calls: {support_calls.min()}-{support_calls.max()} (mean: {support_calls.mean():.1f})")
    
    return df


def generate_iris_data():
    """
    Generate the classic Iris dataset.
    
    Returns:
        DataFrame with iris features and species target
    """
    print(f"\n🌸 Generating Iris Dataset (150 rows)...")
    
    # Setosa
    setosa = pd.DataFrame({
        'sepal_length': np.random.normal(5.0, 0.35, 50),
        'sepal_width': np.random.normal(3.4, 0.38, 50),
        'petal_length': np.random.normal(1.5, 0.17, 50),
        'petal_width': np.random.normal(0.2, 0.10, 50),
        'species': ['setosa'] * 50
    })
    
    # Versicolor
    versicolor = pd.DataFrame({
        'sepal_length': np.random.normal(5.9, 0.52, 50),
        'sepal_width': np.random.normal(2.8, 0.31, 50),
        'petal_length': np.random.normal(4.3, 0.47, 50),
        'petal_width': np.random.normal(1.3, 0.20, 50),
        'species': ['versicolor'] * 50
    })
    
    # Virginica
    virginica = pd.DataFrame({
        'sepal_length': np.random.normal(6.5, 0.64, 50),
        'sepal_width': np.random.normal(3.0, 0.32, 50),
        'petal_length': np.random.normal(5.5, 0.55, 50),
        'petal_width': np.random.normal(2.0, 0.27, 50),
        'species': ['virginica'] * 50
    })
    
    # Combine
    df = pd.concat([setosa, versicolor, virginica], ignore_index=True)
    
    # Round to 1 decimal
    for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
        df[col] = df[col].round(1)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"   ✅ Generated {len(df)} iris samples")
    print(f"   🌸 Species distribution:")
    print(f"      - Setosa: 50")
    print(f"      - Versicolor: 50")
    print(f"      - Virginica: 50")
    
    return df


def create_data_info():
    """
    Create metadata file describing the datasets.
    
    Returns:
        Dictionary with dataset information
    """
    print(f"\n📝 Creating Dataset Metadata...")
    
    info = {
        "customer_churn": {
            "name": "Customer Churn Dataset",
            "description": "Synthetic telecom customer churn dataset with realistic patterns",
            "rows": 10000,
            "task": "binary_classification",
            "target": "churn",
            "target_values": {
                "0": "No churn (customer retained)",
                "1": "Churn (customer left)"
            },
            "features": {
                "customer_id": {
                    "type": "identifier",
                    "description": "Unique customer identifier"
                },
                "age": {
                    "type": "numeric",
                    "description": "Customer age in years",
                    "range": "18-80"
                },
                "tenure_months": {
                    "type": "numeric",
                    "description": "Number of months customer has been with company",
                    "range": "0-120"
                },
                "monthly_charges": {
                    "type": "numeric",
                    "description": "Current monthly service charges in dollars",
                    "range": "~20-180"
                },
                "total_charges": {
                    "type": "numeric",
                    "description": "Total amount charged to customer over lifetime",
                    "range": "0-20000+"
                },
                "support_calls": {
                    "type": "numeric",
                    "description": "Number of support calls made by customer",
                    "range": "0-10+"
                },
                "contract_type": {
                    "type": "categorical",
                    "description": "Type of customer contract",
                    "values": ["Month-to-month", "One year", "Two year"]
                },
                "payment_method": {
                    "type": "categorical",
                    "description": "How customer pays their bill",
                    "values": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
                },
                "internet_service": {
                    "type": "categorical",
                    "description": "Type of internet service",
                    "values": ["DSL", "Fiber optic", "No"]
                }
            },
            "usage_notes": [
                "Primary dataset for Session 5 experiments",
                "Balanced churn rate (~27%) for realistic modeling",
                "Contains both numeric and categorical features",
                "Realistic correlations (tenure reduces churn, support calls increase churn)",
                "Suitable for classification algorithms"
            ]
        },
        "iris": {
            "name": "Iris Flower Dataset",
            "description": "Classic iris species classification dataset",
            "rows": 150,
            "task": "multiclass_classification",
            "target": "species",
            "target_values": {
                "setosa": "Iris Setosa",
                "versicolor": "Iris Versicolor",
                "virginica": "Iris Virginica"
            },
            "features": {
                "sepal_length": {
                    "type": "numeric",
                    "description": "Sepal length in cm",
                    "range": "4.3-7.9"
                },
                "sepal_width": {
                    "type": "numeric",
                    "description": "Sepal width in cm",
                    "range": "2.0-4.4"
                },
                "petal_length": {
                    "type": "numeric",
                    "description": "Petal length in cm",
                    "range": "1.0-6.9"
                },
                "petal_width": {
                    "type": "numeric",
                    "description": "Petal width in cm",
                    "range": "0.1-2.5"
                }
            },
            "usage_notes": [
                "Secondary dataset for quick demos and testing",
                "Perfect for testing multiclass classification",
                "Small size good for fast iteration",
                "Well-separated classes (Setosa is linearly separable)"
            ]
        },
        "generation_info": {
            "script": "generate_datasets.py",
            "random_seed": 42,
            "generation_date": "2025-10-29",
            "purpose": "MLflow Tracking Starter Kit - Session 5"
        }
    }
    
    print(f"   ✅ Metadata created for 2 datasets")
    
    return info


def main():
    """Main execution function."""
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Generate datasets
    churn_df = generate_customer_churn_data(n_samples=10000)
    iris_df = generate_iris_data()
    data_info = create_data_info()
    
    # Save datasets
    print(f"\n💾 Saving Datasets...")
    
    churn_path = script_dir / "customer_churn.csv"
    iris_path = script_dir / "iris.csv"
    info_path = script_dir / "data_info.json"
    
    churn_df.to_csv(churn_path, index=False)
    print(f"   ✅ Saved: {churn_path}")
    
    iris_df.to_csv(iris_path, index=False)
    print(f"   ✅ Saved: {iris_path}")
    
    with open(info_path, 'w') as f:
        json.dump(data_info, f, indent=2)
    print(f"   ✅ Saved: {info_path}")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"✅ Dataset Generation Complete!")
    print(f"=" * 60)
    print(f"\n📁 Generated Files:")
    print(f"   1. {churn_path.name} ({len(churn_df):,} rows)")
    print(f"   2. {iris_path.name} ({len(iris_df)} rows)")
    print(f"   3. {info_path.name} (metadata)")
    print(f"\n🎯 Next Step: Run setup_mlflow.py")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
