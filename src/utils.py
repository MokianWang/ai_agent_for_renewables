"""Shared utilities for model evaluation and visualization"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

from data_loader import load_data, normalize, prepare_features, load_config


# Common task definitions
TASKS = [
    ("solar_power", 1), ("solar_power", 2), ("solar_power", 3),
    ("wind_power", 3), ("wind_power", 4),
    ("load_power", 1), ("load_power", 2), ("load_power", 3), ("load_power", 4),
]


def evaluate_model(model_path, prediction_type, zone, config=None, save_results=True):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to the trained model file (.joblib)
        prediction_type: One of 'solar_power', 'wind_power', 'load_power'
        zone: Zone number (1-4)
        config: Configuration dict (loads from config.json if None)
        save_results: Whether to save CSV and plot results
    
    Returns:
        rmse: Root Mean Squared Error on test set
    """
    if config is None:
        config = load_config("config.json")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Test data
    data = load_data(zone, config)
    data = normalize(data)
    features, targets = prepare_features(data, prediction_type, config, "test")
    
    X_test = features.values
    y_test = targets.values.flatten()
    
    # Predict
    y_pred = np.clip(model.predict(X_test), 0, None)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n{prediction_type} - Zone {zone}")
    print(f"  Test RMSE: {rmse:.6f}")
    print(f"  Samples: {len(y_test)}")
    
    if save_results:
        # Save results
        os.makedirs('results', exist_ok=True)
        pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv(
            f"results/{prediction_type}_zone_{zone}.csv", index=False
        )
        
        # Plot
        os.makedirs('figures', exist_ok=True)
        plot_predictions(y_test, y_pred, prediction_type, zone, rmse)
    
    return rmse


def plot_predictions(y_true, y_pred, prediction_type, zone, rmse=None):
    """
    Plot actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        prediction_type: Type of prediction (for title/filename)
        zone: Zone number (for title/filename)
        rmse: Optional RMSE value to display in title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', linewidth=1.5)
    plt.plot(y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
    
    title = f"{prediction_type} - Zone {zone}"
    if rmse is not None:
        title += f"\nRMSE={rmse:.6f}"
    
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"figures/{prediction_type}_zone_{zone}.png"
    plt.savefig(filename, dpi=150)
    plt.close()


def print_summary_table(results, title="SUMMARY"):
    """
    Print a formatted summary table of results.
    
    Args:
        results: List of tuples (prediction_type, zone, metric_value, ...)
        title: Title for the summary table
    """
    if not results:
        print("No results to display.")
        return
    
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    
    # Determine columns based on result tuple length
    if len(results[0]) == 4:  # (type, zone, val_rmse, test_rmse)
        print(f"{'Type':<15} {'Zone':<5} {'Val RMSE':<12} {'Test RMSE':<12}")
        print(f"{'-'*70}")
        for ptype, zone, vr, tr in results:
            print(f"{ptype:<15} {zone:<5} {vr:<12.6f} {tr:<12.6f}")
    elif len(results[0]) == 3:  # (type, zone, rmse)
        print(f"{'Type':<15} {'Zone':<5} {'RMSE':<12}")
        print(f"{'-'*70}")
        for ptype, zone, rmse in sorted(results, key=lambda x: x[2]):
            print(f"{ptype:<15} {zone:<5} {rmse:<12.6f}")
    
    print(f"{'='*70}")
    
    # Calculate average if applicable
    if len(results[0]) == 3:
        avg_rmse = np.mean([r[2] for r in results])
        print(f"Average RMSE: {avg_rmse:.6f}")
