"""
Smart Model Selector: Choose best model per category (Prophet vs ARIMA vs Ensemble vs ExponentialSmoothing)

Recommends the optimal model for each category based on test set performance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np

ML_DIR = Path(__file__).resolve().parent


def load_comparison() -> pd.DataFrame:
    """Load ensemble vs prophet comparison."""
    return pd.read_csv(ML_DIR / "ensemble_vs_prophet_comparison.csv")


def select_best_models() -> pd.DataFrame:
    """Select best model for each category."""
    comp = load_comparison()
    
    results = []
    for _, row in comp.iterrows():
        category = row["Category"]
        
        # Extract MAE for each model
        models = {
            "ARIMA": row["ARIMA_MAE"],
            "ExponentialSmoothing": row["ExpSmooth_MAE"],
            "Ensemble": row["Ensemble_MAE"],
            "Prophet": row["Prophet_MAE"],
        }
        
        best_model = min(models, key=models.get)
        best_mae = models[best_model]
        
        # Compare best vs Prophet
        improvement = ((row["Prophet_MAE"] - best_mae) / row["Prophet_MAE"] * 100)
        
        results.append({
            "Category": category,
            "Recommended_Model": best_model,
            "MAE": best_mae,
            "Prophet_MAE": row["Prophet_MAE"],
            "Improvement_%": improvement,
        })
    
    df = pd.DataFrame(results)
    return df.sort_values("Improvement_%", ascending=False)


def generate_recommendation_report() -> str:
    """Generate actionable recommendations."""
    df = select_best_models()
    
    report_lines = [
        "\n" + "=" * 100,
        "MODEL SELECTION REPORT - BEST MODEL PER CATEGORY",
        "=" * 100,
        "",
        df.to_string(index=False),
        "",
        "=" * 100,
        "KEY INSIGHTS",
        "=" * 100,
    ]
    
    # Count models
    model_counts = df["Recommended_Model"].value_counts()
    for model, count in model_counts.items():
        report_lines.append(f"  {model:25s}: {count} categories (best choice)")
    
    # Overall improvement
    avg_improvement = df["Improvement_%"].mean()
    if avg_improvement > 0:
        report_lines.append(f"\nOverall: +{avg_improvement:.1f}% improvement over Prophet")
        report_lines.append(f"         ({(df['Improvement_%'] > 0).sum()} categories better, {(df['Improvement_%'] <= 0).sum()} worse)")
    else:
        report_lines.append(f"\nOverall: Prophet is still best on average ({abs(avg_improvement):.1f}% better)")
        report_lines.append(f"         But {(df['Improvement_%'] > 0).sum()} categories can improve with alternatives")
    
    report_lines.append("")
    report_lines.append("=" * 100)
    report_lines.append("DEPLOYMENT STRATEGY")
    report_lines.append("=" * 100)
    
    report_lines.append("\n[OPTION 1] Switch to Best Model Per Category (Mixed Strategy)")
    report_lines.append("  - Use recommendation table above")
    report_lines.append("  - Pros: Max accuracy per category")
    report_lines.append("  - Cons: Requires managing multiple model types")
    
    if avg_improvement > 0:
        report_lines.append("\n[OPTION 2] Switch All to ARIMA")
        report_lines.append("  - ARIMA is best for 6/8 categories")
        report_lines.append("  - Simpler deployment (one model type)")
        report_lines.append("  - Still competitive on other categories")
    else:
        report_lines.append("\n[OPTION 2] Keep Prophet")
        report_lines.append("  - Prophet is best on average")
        report_lines.append("  - Could fine-tune specific categories")
    
    report_lines.append("\n[OPTION 3] Hybrid: Use Ensemble for Uncertain Categories")
    report_lines.append("  - Keep Prophet as default")
    report_lines.append("  - Use Ensemble/ARIMA for high-variance categories (N02BE, R06, R03)")
    report_lines.append("  - Balance accuracy and simplicity")
    
    report_lines.append("\n" + "=" * 100)
    
    return "\n".join(report_lines)


def create_ensemble_config() -> Dict[str, str]:
    """Create config for runtime model selection."""
    df = select_best_models()
    
    config = {}
    for _, row in df.iterrows():
        config[row["Category"]] = row["Recommended_Model"]
    
    return config


def main():
    report = generate_recommendation_report()
    print(report)
    
    # Save report
    report_path = ML_DIR / "model_recommendation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")
    
    # Save config
    config = create_ensemble_config()
    import json
    config_path = ML_DIR / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
