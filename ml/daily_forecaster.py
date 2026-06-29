"""
Unified Daily Forecaster: Forecasts using best daily-trained models

Uses ARIMA / ExponentialSmoothing / Prophet trained on daily data (2105 points).
Much better accuracy than monthly models!
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib
from train_forecasts import get_category_columns

ML_DIR = Path(__file__).resolve().parent
DAILY_MODELS_DIR = ML_DIR / "daily_models"


def load_daily_arima(category: str):
    """Load ARIMA model trained on daily data."""
    try:
        return joblib.load(DAILY_MODELS_DIR / category / "arima.pkl")
    except:
        return None


def load_daily_exp_smoothing(category: str):
    """Load ExponentialSmoothing model trained on daily data."""
    try:
        return joblib.load(DAILY_MODELS_DIR / category / "exp_smoothing.pkl")
    except:
        return None


def load_daily_prophet(category: str):
    """Load Prophet model trained on daily data."""
    try:
        return joblib.load(DAILY_MODELS_DIR / category / "prophet.pkl")
    except:
        return None


class DailyForecaster:
    """Generate forecasts using daily-trained models."""
    
    def __init__(self):
        """Load comparison data to determine best model per category."""
        comp_path = ML_DIR / "daily_models_comparison.csv"
        if comp_path.exists():
            self.comparison = pd.read_csv(comp_path)
            self.model_selection = self._select_best_models()
        else:
            print("Warning: daily_models_comparison.csv not found")
            self.comparison = None
            self.model_selection = {}
    
    def _select_best_models(self) -> Dict[str, str]:
        """Select best model per category based on daily comparison."""
        selection = {}
        for _, row in self.comparison.iterrows():
            cat = row["Category"]
            models = {
                "ARIMA": row["ARIMA_MAE"],
                "ExponentialSmoothing": row["ExpSmooth_MAE"],
                "Prophet": row["Prophet_MAE"],
            }
            selection[cat] = min(models, key=models.get)
        return selection
    
    def forecast(self, category: str, periods: int = 30) -> Dict[str, object]:
        """
        Generate forecast using best daily-trained model.
        
        Args:
            category: Category name
            periods: Number of days to forecast
        
        Returns:
            Dictionary with forecast results
        """
        best_model = self.model_selection.get(category, "Prophet")
        
        if best_model == "ARIMA":
            return self._forecast_arima(category, periods)
        elif best_model == "ExponentialSmoothing":
            return self._forecast_exp_smoothing(category, periods)
        else:
            return self._forecast_prophet(category, periods)
    
    def _forecast_arima(self, category: str, periods: int) -> Dict:
        """Forecast using ARIMA."""
        model = load_daily_arima(category)
        if model is None:
            raise ValueError(f"ARIMA model not found for {category}")
        
        pred = model.get_forecast(steps=periods).predicted_mean.values
        ds = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq="D")
        
        return {
            "model_type": "ARIMA",
            "category": category,
            "periods": periods,
            "ds": ds,
            "forecast": pred,
            "data": pd.DataFrame({
                "ds": ds,
                "forecast": pred,
            })
        }
    
    def _forecast_exp_smoothing(self, category: str, periods: int) -> Dict:
        """Forecast using ExponentialSmoothing."""
        model = load_daily_exp_smoothing(category)
        if model is None:
            raise ValueError(f"ExponentialSmoothing model not found for {category}")
        
        pred = model.forecast(steps=periods).values
        ds = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq="D")
        
        return {
            "model_type": "ExponentialSmoothing",
            "category": category,
            "periods": periods,
            "ds": ds,
            "forecast": pred,
            "data": pd.DataFrame({
                "ds": ds,
                "forecast": pred,
            })
        }
    
    def _forecast_prophet(self, category: str, periods: int) -> Dict:
        """Forecast using Prophet."""
        model = load_daily_prophet(category)
        if model is None:
            raise ValueError(f"Prophet model not found for {category}")
        
        future = model.make_future_dataframe(periods=periods, freq="D")
        forecast = model.predict(future)
        
        tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
        
        return {
            "model_type": "Prophet",
            "category": category,
            "periods": periods,
            "ds": tail["ds"].values,
            "forecast": tail["yhat"].values,
            "lower": tail["yhat_lower"].values,
            "upper": tail["yhat_upper"].values,
            "data": tail,
        }
    
    def forecast_all(self, periods: int = 30) -> Dict[str, Dict]:
        """Forecast all categories."""
        results = {}
        for category in get_category_columns():
            try:
                results[category] = self.forecast(category, periods=periods)
            except Exception as e:
                print(f"Error forecasting {category}: {e}")
        return results
    
    def get_model_config(self) -> Dict[str, str]:
        """Get mapping of categories to best models."""
        return self.model_selection.copy()


def main():
    """Example usage."""
    print("\n" + "=" * 100)
    print("DAILY FORECASTER - Using Models Trained on Daily Aggregated Data")
    print("=" * 100)
    
    forecaster = DailyForecaster()
    
    # Show configuration
    print("\nBest Model Per Category (trained on 2,105 daily data points):")
    print("-" * 100)
    for cat, model in sorted(forecaster.model_selection.items()):
        print(f"  {cat:10s}: {model:25s}")
    
    # Example forecast
    print("\n" + "=" * 100)
    print("EXAMPLE: 30-Day Forecast for N05B (using ARIMA)")
    print("=" * 100)
    
    forecast = forecaster.forecast("N05B", periods=30)
    print(f"\nModel: {forecast['model_type']}")
    print(f"Next 10 days:")
    for i, (d, val) in enumerate(zip(forecast["ds"][:10], forecast["forecast"][:10]), 1):
        date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        print(f"  Day {i:2d} ({date_str}): {val:.2f}")
    
    # Full report
    print("\n" + "=" * 100)
    print("ALL CATEGORIES - 7-DAY FORECAST")
    print("=" * 100)
    
    all_forecasts = forecaster.forecast_all(periods=7)
    
    for category, fcast in sorted(all_forecasts.items()):
        print(f"\n{category} ({fcast['model_type']}):")
        for i, (d, val) in enumerate(zip(fcast["ds"][:7], fcast["forecast"][:7]), 1):
            date_str = pd.Timestamp(d).strftime("%Y-%m-%d")
            print(f"  {date_str}: {val:8.2f}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
