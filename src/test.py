"""Model Testing and Evaluation Script

This script evaluates all trained models and generates summary reports.
Uses shared utilities from utils.py for consistent evaluation logic.
"""

import os

from data_loader import load_config
from utils import TASKS, evaluate_model, print_summary_table


def test_all():
    """Test all trained models and generate summary report."""
    config = load_config("config.json")
    
    results = []
    missing_models = []
    
    print("="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    for ptype, zone in TASKS:
        model_path = f"models/{ptype}_{zone}.joblib"
        
        if os.path.exists(model_path):
            try:
                rmse = evaluate_model(
                    model_path, ptype, zone, config, save_results=True
                )
                results.append((ptype, zone, rmse))
            except Exception as e:
                print(f"\nError testing {ptype} zone {zone}: {e}")
                import traceback
                traceback.print_exc()
        else:
            missing_models.append((ptype, zone))
            print(f"\nModel not found: {model_path}")
    
    # Print summary table
    if results:
        print_summary_table(results, "TEST SUMMARY")
    
    # Report missing models
    if missing_models:
        print(f"\n{'='*70}")
        print("MISSING MODELS")
        print(f"{'='*70}")
        for ptype, zone in missing_models:
            print(f"  - {ptype} zone {zone}")
        print(f"{'='*70}")
        print(f"Total missing: {len(missing_models)} models")
    
    return results


if __name__ == "__main__":
    test_all()
