"""
Example: Evaluate trained Prophet models using the evaluate module.

This script demonstrates how to:
1. Load training results from train_forecasts.py
2. Compute comprehensive metrics (MAE, RMSE, MAPE, MASE, R2)
3. Perform residual diagnostics (ACF, Ljung-Box, KPSS tests)
4. Generate diagnostic plots
5. Compare models across all categories
"""

from __future__ import annotations

import sys
from pathlib import Path
import os

# Fix encoding for Windows terminals
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add ml directory to path
ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

import pandas as pd
import matplotlib.pyplot as plt
from train_forecasts import (
    load_sales_data, build_monthly_sales, train_all_categories, 
    prepare_prophet_frame, get_category_columns
)
from evaluate import ProphetEvaluator, ModelComparison, print_evaluation_report


def main() -> None:
    """Run evaluation on all trained models."""
    
    print("\n" + "=" * 70)
    print("PROPHET MODEL EVALUATION")
    print("=" * 70)
    
    # ==================== TRAIN OR LOAD MODELS ====================
    print("\n[1/4] Training models on all categories...")
    data = load_sales_data()
    monthly = build_monthly_sales(data)
    results = train_all_categories(cutoff="2018-06-30")
    print(f"[OK] Trained {len(results)} models")
    
    # ==================== EVALUATE EACH MODEL ====================
    print("\n[2/4] Evaluating forecasts...")
    evaluations = {}
    
    for category, training_result in results.items():
        forecast_df = training_result["forecast"]
        validation_df = training_result["validation_df"]
        
        evaluation = ProphetEvaluator.evaluate_forecast(
            forecast_df, 
            validation_df, 
            category=category
        )
        evaluations[category] = evaluation
        
        # Print summary for each category
        print(print_evaluation_report(evaluation, verbose=False))
    
    # ==================== COMPARE MODELS ====================
    print("\n[3/4] Comparing models...")
    comparison_table = ModelComparison.compare_categories(evaluations)
    print("\nModel Comparison Table (sorted by RMSE):")
    print(comparison_table.to_string(index=False))
    
    # Save comparison to CSV
    comparison_table.to_csv(ML_DIR / "comparison_report.csv", index=False)
    print(f"\n[OK] Saved comparison to {ML_DIR / 'comparison_report.csv'}")
    
    # ==================== GENERATE PLOTS ====================
    print("\n[4/4] Generating diagnostic plots...")
    plots_dir = ML_DIR / "evaluation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for category, evaluation in evaluations.items():
        forecast_df = results[category]["forecast"]
        validation_df = results[category]["validation_df"]
        
        # Generate and save plots
        eval_with_plots = ProphetEvaluator.evaluate_and_plot(
            forecast_df,
            validation_df,
            category=category,
            save_path=plots_dir
        )
        
        print(f"  [OK] {category}: diagnostics, actual vs fitted")
    
    print(f"\n[OK] Saved plots to {plots_dir}")
    
    # ==================== DETAILED DIAGNOSTICS FOR BEST MODEL ====================
    best_category = comparison_table.iloc[0]["Category"]
    print(f"\n[BONUS] Detailed diagnostics for best model: {best_category}")
    print(print_evaluation_report(evaluations[best_category], verbose=True))
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  • Comparison table: {ML_DIR / 'comparison_report.csv'}")
    print(f"  • Diagnostic plots: {plots_dir}")
    print(f"\nTo load results later:")
    print(f"  from evaluate import ProphetEvaluator")
    print(f"  from train_forecasts import load_trained_model, build_monthly_sales")


if __name__ == "__main__":
    main()
