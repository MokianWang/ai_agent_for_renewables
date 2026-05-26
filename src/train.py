"""LightGBM Training Script for Power Prediction"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import joblib

from data_loader import load_data, normalize, prepare_features, load_config
from utils import TASKS, evaluate_model, print_summary_table


def train_model(config, prediction_type, zone):
    """Train LightGBM model.
    
    Args:
        config: Configuration dictionary
        prediction_type: One of 'solar_power', 'wind_power', 'load_power'
        zone: Zone number (1-4)
    
    Returns:
        model: Trained LightGBM model
        val_rmse: Validation RMSE
    """
    print(f"\n{'#'*70}")
    print(f"# Training: {prediction_type} - Zone {zone}")
    print(f"{'#'*70}\n")
    
    # Load and prepare data
    data = load_data(zone, config)
    data = normalize(data)
    features, targets = prepare_features(data, prediction_type, config, "train")
    
    X = features.values
    y = targets.values.flatten()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # LightGBM parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.log_evaluation(50), lgb.early_stopping(20)]
    )
    
    # Evaluate on validation set
    val_pred = np.clip(model.predict(X_val), 0, None)
    val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
    
    print(f"\nVal RMSE: {val_rmse:.6f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = f"models/{prediction_type}_{zone}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved to {model_path}")
    
    return model, val_rmse


def main():
    """Main training function."""
    config = load_config("config.json")
    
    results = []
    
    for ptype, zone in TASKS:
        try:
            # Train model
            _, val_rmse = train_model(config, ptype, zone)
            
            # Evaluate on test set using shared utility
            test_rmse = evaluate_model(
                f"models/{ptype}_{zone}.joblib",
                ptype, zone, config, save_results=True
            )
            
            results.append((ptype, zone, val_rmse, test_rmse))
        except Exception as e:
            print(f"Error training {ptype} zone {zone}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if results:
        print_summary_table(results, "TRAINING SUMMARY")


if __name__ == "__main__":
    main()
