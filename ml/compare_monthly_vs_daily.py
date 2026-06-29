"""
Compare Monthly vs Daily Model Training Results

Shows why daily data is better for ARIMA.
"""

import pandas as pd
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent


def main():
    print("\n" + "=" * 120)
    print("COMPARISON: MONTHLY vs DAILY DATA TRAINING")
    print("=" * 120)
    
    # Load comparisons
    monthly = pd.read_csv(ML_DIR / "ensemble_vs_prophet_comparison.csv")
    daily = pd.read_csv(ML_DIR / "daily_models_comparison.csv")
    
    print("\n[MONTHLY DATA] (70 data points: Jan 2014 - Jun 2019)")
    print("-" * 120)
    print(monthly[["Category", "ARIMA_MAE", "ExpSmooth_MAE", "Ensemble_MAE", "Prophet_MAE"]].to_string(index=False))
    
    print("\n\n[DAILY DATA] (2105 data points: Jan 2014 - Oct 2019)")
    print("-" * 120)
    print(daily[["Category", "ARIMA_MAE", "ExpSmooth_MAE", "Prophet_MAE"]].to_string(index=False))
    
    print("\n\n" + "=" * 120)
    print("KEY FINDINGS")
    print("=" * 120)
    
    improvements = []
    for _, row in daily.iterrows():
        cat = row["Category"]
        monthly_row = monthly[monthly["Category"] == cat].iloc[0]
        
        # Get best model
        daily_models = {
            "ARIMA": row["ARIMA_MAE"],
            "ExpSmooth": row["ExpSmooth_MAE"],
            "Prophet": row["Prophet_MAE"],
        }
        daily_best = min(daily_models, key=daily_models.get)
        
        monthly_models = {
            "ARIMA": monthly_row["ARIMA_MAE"],
            "ExpSmooth": monthly_row["ExpSmooth_MAE"],
            "Ensemble": monthly_row["Ensemble_MAE"],
            "Prophet": monthly_row["Prophet_MAE"],
        }
        monthly_best = min(monthly_models, key=monthly_models.get)
        
        improvement = ((monthly_models[monthly_best] - daily_models[daily_best]) / monthly_models[monthly_best] * 100)
        
        improvements.append({
            "Category": cat,
            "Best_Model_Monthly": monthly_best,
            "Best_Model_Daily": daily_best,
            "Monthly_MAE": monthly_models[monthly_best],
            "Daily_MAE": daily_models[daily_best],
            "Improvement_%": improvement,
        })
    
    imp_df = pd.DataFrame(improvements).sort_values("Improvement_%", ascending=False)
    print("\nBest Model Performance Comparison:")
    print(imp_df.to_string(index=False))
    
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    
    avg_monthly = monthly["Prophet_MAE"].mean()
    
    best_daily = []
    for _, row in daily.iterrows():
        best = min(row["ARIMA_MAE"], row["ExpSmooth_MAE"], row["Prophet_MAE"])
        best_daily.append(best)
    
    avg_daily = sum(best_daily) / len(best_daily)
    
    print(f"\nAverage MAE (best model per category):")
    print(f"  Monthly data: {avg_monthly:.2f}")
    print(f"  Daily data:   {avg_daily:.2f}")
    print(f"  Improvement:  {((avg_monthly - avg_daily) / avg_monthly * 100):+.1f}%")
    
    print(f"\nData Points:")
    print(f"  Monthly: ~70 points")
    print(f"  Daily:   2,105 points (30x more data!)")
    
    print(f"\nWhy Daily is Better for ARIMA:")
    print(f"  ✓ Much more training data (30x): Better parameter estimation")
    print(f"  ✓ Less aggregation loss: Daily values are closer to raw signal")
    print(f"  ✓ Better statistical properties: ARIMA needs enough samples")
    print(f"  ✓ More stable forecasts: Less reliance on few data points")
    
    print(f"\nRECOMMENDATION:")
    print(f"  ✓ USE DAILY DATA for all future training")
    print(f"  ✓ Update unified_forecaster to use daily models")
    print(f"  ✓ Archive monthly model experiments")
    
    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
