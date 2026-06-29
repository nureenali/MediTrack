"""
Train models on DAILY aggregated data from hourly sales records.

Aggregates 5 years of hourly data to daily, then trains ARIMA + Ensemble + Prophet.
Much better for ARIMA (2105 daily points vs 70 monthly points).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from train_forecasts import get_category_columns
from evaluate import ForecastMetrics

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
DAILY_MODELS_DIR = ML_DIR / "daily_models"
DAILY_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_hourly_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load hourly sales data."""
    path = data_path or DATA_DIR / "saleshourly.csv"
    df = pd.read_csv(path)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])
    return df.sort_values("datum").reset_index(drop=True)


def build_daily_sales(df: pd.DataFrame, category_columns: list = None) -> pd.DataFrame:
    """Aggregate hourly data to daily sums."""
    if category_columns is None:
        category_columns = get_category_columns()
    
    daily = (
        df.groupby(df["datum"].dt.date)[category_columns]
        .sum()
        .reset_index()
    )
    daily.columns = ["ds"] + category_columns
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)
    return daily


def train_daily_arima(y_train: np.ndarray, y_test: np.ndarray, category: str) -> Dict:
    """Train ARIMA on daily data."""
    try:
        model = ARIMA(pd.Series(y_train), order=(1, 1, 1))
        result = model.fit()
        y_pred = result.get_forecast(steps=len(y_test)).predicted_mean.values
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            "status": "OK",
            "model": result,
            "mae": mae,
            "rmse": rmse,
            "predictions": y_pred,
        }
    except Exception as e:
        return {
            "status": f"Failed: {str(e)[:50]}",
            "mae": np.nan,
            "rmse": np.nan,
        }


def train_daily_exp_smoothing(y_train: np.ndarray, y_test: np.ndarray, category: str) -> Dict:
    """Train ExponentialSmoothing on daily data."""
    try:
        model = ExponentialSmoothing(pd.Series(y_train), trend='add', seasonal=None, initialization_method='estimated')
        result = model.fit(optimized=True)
        y_pred = result.forecast(steps=len(y_test)).values
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            "status": "OK",
            "model": result,
            "mae": mae,
            "rmse": rmse,
            "predictions": y_pred,
        }
    except Exception as e:
        return {
            "status": f"Failed: {str(e)[:50]}",
            "mae": np.nan,
            "rmse": np.nan,
        }


def train_daily_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, category: str) -> Dict:
    """Train Prophet on daily data."""
    try:
        # Prepare data
        train_data = train_df[["ds", category]].copy()
        train_data.columns = ["ds", "y"]
        
        # Train
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(train_data)
        
        # Forecast
        future = model.make_future_dataframe(periods=len(test_df), freq="D")
        forecast = model.predict(future)
        
        # Get predictions for test period
        test_forecast = forecast[forecast["ds"].isin(test_df["ds"])][["ds", "yhat"]].copy()
        y_pred = test_forecast["yhat"].values
        y_test = test_df[category].values
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            "status": "OK",
            "model": model,
            "forecast": forecast,
            "mae": mae,
            "rmse": rmse,
            "predictions": y_pred,
        }
    except Exception as e:
        return {
            "status": f"Failed: {str(e)[:50]}",
            "mae": np.nan,
            "rmse": np.nan,
        }


def train_all_daily_models(cutoff_date: str = "2019-06-30") -> Dict[str, Dict]:
    """Train all models on daily aggregated data."""
    
    print("\n" + "=" * 100)
    print("TRAINING MODELS ON DAILY AGGREGATED DATA")
    print("=" * 100)
    
    # Load and aggregate data
    print("\n[1/4] Loading hourly data and aggregating to daily...")
    hourly_df = load_hourly_data()
    daily_df = build_daily_sales(hourly_df)
    
    print(f"  Daily data points: {len(daily_df)}")
    print(f"  Date range: {daily_df['ds'].min().date()} to {daily_df['ds'].max().date()}")
    print(f"  Duration: {(daily_df['ds'].max() - daily_df['ds'].min()).days} days")
    
    # Split
    cutoff = pd.Timestamp(cutoff_date)
    train_df = daily_df[daily_df["ds"] <= cutoff].copy()
    test_df = daily_df[daily_df["ds"] > cutoff].copy()
    
    print(f"  Training set: {len(train_df)} days")
    print(f"  Test set: {len(test_df)} days")
    
    # Train models for each category
    print("\n[2/4] Training models for each category...")
    results = {}
    
    for category in get_category_columns():
        print(f"\n  {category}:")
        
        y_train = train_df[category].values
        y_test = test_df[category].values
        
        # ARIMA
        arima_result = train_daily_arima(y_train, y_test, category)
        arima_mae = arima_result.get("mae", np.nan)
        print(f"    ARIMA:             MAE={arima_mae:8.2f}  {arima_result['status']}")
        
        # ExponentialSmoothing
        exp_result = train_daily_exp_smoothing(y_train, y_test, category)
        exp_mae = exp_result.get("mae", np.nan)
        print(f"    ExponentialSmooth: MAE={exp_mae:8.2f}  {exp_result['status']}")
        
        # Prophet
        prophet_result = train_daily_prophet(train_df, test_df, category)
        prophet_mae = prophet_result.get("mae", np.nan)
        print(f"    Prophet:           MAE={prophet_mae:8.2f}  {prophet_result['status']}")
        
        results[category] = {
            "category": category,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "arima": arima_result,
            "exp_smoothing": exp_result,
            "prophet": prophet_result,
        }
    
    # Save models
    print("\n[3/4] Saving models...")
    for category, result in results.items():
        cat_dir = DAILY_MODELS_DIR / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        
        if result["arima"].get("model") is not None:
            joblib.dump(result["arima"]["model"], cat_dir / "arima.pkl")
        if result["exp_smoothing"].get("model") is not None:
            joblib.dump(result["exp_smoothing"]["model"], cat_dir / "exp_smoothing.pkl")
        if result["prophet"].get("model") is not None:
            joblib.dump(result["prophet"]["model"], cat_dir / "prophet.pkl")
    
    print(f"  Models saved to: {DAILY_MODELS_DIR}")
    
    # Create comparison
    print("\n[4/4] Creating comparison table...")
    comparison_rows = []
    for category, result in results.items():
        row = {
            "Category": category,
            "ARIMA_MAE": result["arima"].get("mae", np.nan),
            "ARIMA_RMSE": result["arima"].get("rmse", np.nan),
            "ExpSmooth_MAE": result["exp_smoothing"].get("mae", np.nan),
            "ExpSmooth_RMSE": result["exp_smoothing"].get("rmse", np.nan),
            "Prophet_MAE": result["prophet"].get("mae", np.nan),
            "Prophet_RMSE": result["prophet"].get("rmse", np.nan),
        }
        comparison_rows.append(row)
    
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(ML_DIR / "daily_models_comparison.csv", index=False)
    
    print("\n" + "=" * 100)
    print("COMPARISON: Daily Data Models")
    print("=" * 100)
    print(comparison.to_string(index=False))
    
    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    best_models = {}
    for _, row in comparison.iterrows():
        cat = row["Category"]
        models = {
            "ARIMA": row["ARIMA_MAE"],
            "ExpSmooth": row["ExpSmooth_MAE"],
            "Prophet": row["Prophet_MAE"],
        }
        best = min(models, key=models.get)
        best_models[cat] = best
        improvement = ((row["Prophet_MAE"] - models[best]) / row["Prophet_MAE"] * 100)
        print(f"  {cat:10s}: {best:15s} (MAE={models[best]:.2f}, {improvement:+.1f}% vs Prophet)")
    
    print("\n" + "=" * 100)
    
    return results


def main():
    train_all_daily_models(cutoff_date="2019-06-30")


if __name__ == "__main__":
    main()
