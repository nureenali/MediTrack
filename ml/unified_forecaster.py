"""
Unified Forecast Interface: Generate forecasts using best model per category

Automatically selects Prophet, ARIMA, Ensemble, or ExponentialSmoothing based on
performance on validation set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from train_forecasts import load_trained_model as load_prophet_model
from train_arima_ensemble import load_ensemble

ML_DIR = Path(__file__).resolve().parent


class UnifiedForecaster:
    """Generate forecasts using optimal model per category."""
    
    def __init__(self):
        """Load model configuration and mapping."""
        config_path = ML_DIR / "model_config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)
    
    def forecast(self, category: str, periods: int = 12) -> Dict[str, object]:
        """
        Generate forecast for a category using best model.
        
        Args:
            category: Category name
            periods: Number of periods to forecast
        
        Returns:
            Dictionary with 'model_type', 'forecasts', 'ds'
        """
        model_type = self.config.get(category, "Prophet")
        
        if model_type == "Prophet":
            return self._forecast_prophet(category, periods)
        elif model_type == "ARIMA":
            return self._forecast_arima(category, periods)
        elif model_type == "ExponentialSmoothing":
            return self._forecast_exp_smoothing(category, periods)
        elif model_type == "Ensemble":
            return self._forecast_ensemble(category, periods)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _forecast_prophet(self, category: str, periods: int) -> Dict:
        """Forecast using Prophet."""
        from train_forecasts import generate_forecast
        
        forecast_df = generate_forecast(category, periods=periods)
        last_periods = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
        
        return {
            "model_type": "Prophet",
            "category": category,
            "periods": periods,
            "ds": last_periods["ds"].values,
            "point_forecast": last_periods["yhat"].values,
            "lower": last_periods["yhat_lower"].values,
            "upper": last_periods["yhat_upper"].values,
            "dataframe": last_periods,
        }
    
    def _forecast_arima(self, category: str, periods: int) -> Dict:
        """Forecast using ARIMA."""
        models = load_ensemble(category)
        arima_model = models.get("arima")
        
        if arima_model is None:
            raise ValueError(f"ARIMA model not found for {category}")
        
        pred = arima_model.predict(periods)
        ds = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq="MS")
        
        return {
            "model_type": "ARIMA",
            "category": category,
            "periods": periods,
            "ds": ds.values,
            "point_forecast": pred,
            "lower": None,
            "upper": None,
            "note": "ARIMA does not provide prediction intervals by default",
        }
    
    def _forecast_exp_smoothing(self, category: str, periods: int) -> Dict:
        """Forecast using ExponentialSmoothing."""
        models = load_ensemble(category)
        exp_model = models.get("exp_smoothing")
        
        if exp_model is None:
            raise ValueError(f"ExponentialSmoothing model not found for {category}")
        
        pred = exp_model.predict(periods)
        ds = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq="MS")
        
        return {
            "model_type": "ExponentialSmoothing",
            "category": category,
            "periods": periods,
            "ds": ds.values,
            "point_forecast": pred,
            "lower": None,
            "upper": None,
            "note": "ExponentialSmoothing does not provide prediction intervals by default",
        }
    
    def _forecast_ensemble(self, category: str, periods: int) -> Dict:
        """Forecast using Ensemble."""
        models = load_ensemble(category)
        ensemble_model = models.get("ensemble")
        
        if ensemble_model is None:
            raise ValueError(f"Ensemble model not found for {category}")
        
        pred = ensemble_model.predict(periods)
        ds = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq="MS")
        
        return {
            "model_type": "Ensemble",
            "category": category,
            "periods": periods,
            "ds": ds.values,
            "point_forecast": pred,
            "lower": None,
            "upper": None,
            "note": "Ensemble combines ARIMA(0.5) + ExponentialSmoothing(0.5)",
        }
    
    def forecast_all_categories(self, periods: int = 12) -> Dict[str, Dict]:
        """Forecast all categories."""
        from train_forecasts import get_category_columns
        
        results = {}
        for category in get_category_columns():
            try:
                results[category] = self.forecast(category, periods=periods)
            except Exception as e:
                print(f"Error forecasting {category}: {e}")
        
        return results
    
    def create_forecast_report(self, periods: int = 12) -> str:
        """Generate human-readable forecast report."""
        from train_forecasts import get_category_columns
        
        lines = [
            "\n" + "=" * 100,
            "FORECAST REPORT - USING BEST MODEL PER CATEGORY",
            "=" * 100,
        ]
        
        model_summary = {}
        
        for category in get_category_columns():
            try:
                forecast = self.forecast(category, periods=periods)
                model_type = forecast["model_type"]
                model_summary[model_type] = model_summary.get(model_type, 0) + 1
                
                lines.append(f"\n{category} ({model_type})")
                lines.append("-" * 50)
                
                if isinstance(forecast["dataframe"], pd.DataFrame):
                    df = forecast["dataframe"].copy()
                    df["ds"] = df["ds"].dt.strftime("%Y-%m-%d")
                    lines.append(df.to_string(index=False))
                else:
                    lines.append(f"  Point forecasts (next {periods} periods):")
                    for i, (d, val) in enumerate(zip(forecast["ds"], forecast["point_forecast"]), 1):
                        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
                        lines.append(f"    Period {i} ({date_str}): {val:.2f}")
            
            except Exception as e:
                lines.append(f"\n{category}: ERROR - {e}")
        
        lines.append("\n" + "=" * 100)
        lines.append("MODELS USED")
        lines.append("=" * 100)
        for model, count in sorted(model_summary.items(), key=lambda x: -x[1]):
            lines.append(f"  {model:25s}: {count} categories")
        
        lines.append("\n" + "=" * 100)
        
        return "\n".join(lines)


def main():
    """Example usage."""
    forecaster = UnifiedForecaster()
    
    # Show configuration
    print("\n" + "=" * 100)
    print("MODEL CONFIGURATION LOADED")
    print("=" * 100)
    for cat, model in sorted(forecaster.config.items()):
        print(f"  {cat:10s}: {model}")
    
    # Generate forecast for one category
    print("\n" + "=" * 100)
    print("EXAMPLE: Forecast for N05B (using Ensemble)")
    print("=" * 100)
    forecast = forecaster.forecast("N05B", periods=12)
    print(f"\nModel Type: {forecast['model_type']}")
    print(f"Next {forecast['periods']} periods:")
    for i, (d, val) in enumerate(zip(forecast["ds"], forecast["point_forecast"]), 1):
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        print(f"  Period {i:2d} ({date_str}): {val:8.2f}")
    
    # Full report
    report = forecaster.create_forecast_report(periods=6)
    print(report)
    
    # Save report
    report_path = ML_DIR / "unified_forecast_example.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
