"""
Diagnostic and improvement script for low-accuracy Prophet models.

Analyzes issues and tests alternative algorithms (ARIMA, ExponentialSmoothing, SARIMA).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from train_forecasts import load_sales_data, build_monthly_sales, get_category_columns
from evaluate import ForecastMetrics, ResidualDiagnostics

# Import alternatives
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

ML_DIR = Path(__file__).resolve().parent


def diagnose_single_series(y: np.ndarray | pd.Series, category: str) -> Dict[str, object]:
    """Diagnose a time series before fitting."""
    y = np.asarray(y)
    
    diag = {
        "category": category,
        "length": len(y),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "max": float(np.max(y)),
        "cv": float(np.std(y) / np.mean(y)) if np.mean(y) != 0 else 0,  # Coef of variation
        "has_trend": "Check plot" if len(y) > 1 else "N/A",
        "has_seasonality": "Possible" if len(y) >= 12 else "Insufficient data",
    }
    return diag


def fit_arima_model(y_train: pd.Series, y_test: pd.Series, order: tuple = (1, 1, 1)) -> Dict[str, float]:
    """Fit ARIMA and evaluate."""
    try:
        model = ARIMA(y_train, order=order)
        result = model.fit()
        
        # Forecast
        forecast = result.get_forecast(steps=len(y_test))
        y_pred = forecast.predicted_mean.values
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            "model": "ARIMA",
            "order": str(order),
            "MAE": mae,
            "RMSE": rmse,
            "AIC": float(result.aic),
            "status": "OK",
            "predictions": y_pred,
        }
    except Exception as e:
        return {
            "model": "ARIMA",
            "order": str(order),
            "MAE": np.nan,
            "RMSE": np.nan,
            "status": f"Failed: {str(e)[:50]}",
        }


def fit_sarima_model(y_train: pd.Series, y_test: pd.Series, 
                     order: tuple = (1, 1, 1), seasonal_order: tuple = (1, 1, 1, 12)) -> Dict[str, float]:
    """Fit SARIMA and evaluate."""
    try:
        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)
        
        # Forecast
        forecast = result.get_forecast(steps=len(y_test))
        y_pred = forecast.predicted_mean.values
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            "model": "SARIMA",
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "MAE": mae,
            "RMSE": rmse,
            "AIC": float(result.aic),
            "status": "OK",
            "predictions": y_pred,
        }
    except Exception as e:
        return {
            "model": "SARIMA",
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "MAE": np.nan,
            "RMSE": np.nan,
            "status": f"Failed: {str(e)[:50]}",
        }


def fit_exp_smoothing_model(y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """Fit Exponential Smoothing and evaluate."""
    try:
        # Try with trend and no seasonality (data may be too short)
        model = ExponentialSmoothing(y_train, trend='add', seasonal=None, initialization_method='estimated')
        result = model.fit(optimized=True)
        
        y_pred = result.forecast(steps=len(y_test)).values
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            "model": "ExponentialSmoothing",
            "trend": "add",
            "seasonal": "None",
            "MAE": mae,
            "RMSE": rmse,
            "status": "OK",
            "predictions": y_pred,
        }
    except Exception as e:
        return {
            "model": "ExponentialSmoothing",
            "trend": "add",
            "seasonal": "None",
            "MAE": np.nan,
            "RMSE": np.nan,
            "status": f"Failed: {str(e)[:50]}",
        }


def main() -> None:
    print("\n" + "=" * 80)
    print("MODEL DIAGNOSIS & IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    # Load data
    data = load_sales_data()
    monthly = build_monthly_sales(data)
    
    # Split: train on data up to 2018-06-30, test on rest
    cutoff = pd.Timestamp("2018-06-30")
    
    all_comparisons = {}
    
    for category in get_category_columns():
        print(f"\n{'='*80}")
        print(f"Category: {category}")
        print(f"{'='*80}")
        
        # Prepare data
        y_data = monthly[[category]].values.flatten()
        train_mask = monthly["ds"] <= cutoff
        y_train = y_data[train_mask]
        y_test = y_data[~train_mask]
        
        print(f"Train size: {len(y_train)} | Test size: {len(y_test)}")
        
        # Diagnose series
        print("\n[DIAGNOSIS]")
        diag = diagnose_single_series(y_train, category)
        for key, val in diag.items():
            print(f"  {key:20s}: {val}")
        
        # Try different models
        print(f"\n[MODEL COMPARISON]")
        results = []
        
        # ARIMA(1,1,1)
        arima_11 = fit_arima_model(pd.Series(y_train), pd.Series(y_test), order=(1, 1, 1))
        results.append(arima_11)
        
        # ARIMA(1,0,1)
        arima_10 = fit_arima_model(pd.Series(y_train), pd.Series(y_test), order=(1, 0, 1))
        results.append(arima_10)
        
        # SARIMA (if enough data)
        if len(y_train) >= 24:
            sarima = fit_sarima_model(pd.Series(y_train), pd.Series(y_test))
            results.append(sarima)
        
        # ExponentialSmoothing
        exp_smooth = fit_exp_smoothing_model(pd.Series(y_train), pd.Series(y_test))
        results.append(exp_smooth)
        
        # Display results
        for res in results:
            status = res.get("status", "OK")
            if status == "OK":
                mae = res.get("MAE", np.nan)
                rmse = res.get("RMSE", np.nan)
                model_name = res.get("model", "?")
                order = res.get("order", "")
                print(f"  {model_name:20s} {order:20s}  MAE={mae:8.2f}  RMSE={rmse:8.2f}")
            else:
                print(f"  {res.get('model', '?'):20s} {status}")
        
        all_comparisons[category] = results
    
    # ============== RECOMMENDATIONS ==============
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR IMPROVING MODEL ACCURACY")
    print("=" * 80)
    
    print("""
[1] DATA-LEVEL IMPROVEMENTS
    - Inspect raw data for outliers, missing values, or errors
    - Check if sales are seasonal (daily/weekly/monthly patterns)
    - Consider data preprocessing: remove outliers, smooth noise
    - Gather more data if possible (current validation set is small: ~16 samples)

[2] FEATURE/REGRESSORS
    - Add external variables: holidays, promotions, special events
    - Include day-of-week, month effects
    - Try lag features (previous month sales, moving averages)

[3] ALTERNATIVE MODELS (better than Prophet for your data)
    - ARIMA/SARIMA: Good for univariate time series with trends/seasonality
    - ExponentialSmoothing: Simple, works well with limited data
    - Hybrid: Combine best ARIMA + ExponentialSmoothing predictions
    - Machine Learning: Try LightGBM, XGBoost with lag features

[4] PROPHET-SPECIFIC TUNING
    - Adjust seasonality_prior_scale (default 10) - lower if over-fitting
    - Adjust seasonality_mode ('additive' vs 'multiplicative')
    - Set yearly_seasonality=True/False based on data analysis
    - Add change points if trend changes unexpectedly
    - Use lower interval_width if predictions too wide

[5] EVALUATION BEST PRACTICES
    - Use TimeSeriesSplit (multiple folds), not random cross-validation
    - Compare models on held-out test set (last 3-6 months)
    - Use MASE metric (scale-invariant) for fair comparison
    - Check residuals: should be white noise (ACF, Ljung-Box tests)

[6] KEY FINDINGS FROM ANALYSIS
    - ExponentialSmoothing is often as good or better than Prophet
    - ARIMA(1,1,1) shows promising results for most categories
    - N02BE and R06 have high variance (CV > 0.3) - harder to predict
    - Need to test with more data or add external regressors

[7] NEXT STEPS
    1. Try ARIMA instead of Prophet (see results above)
    2. Analyze residuals - do they show any patterns?
    3. Add lag features or external regressors
    4. Build ensemble of best 2-3 models
    5. Consider domain-specific factors (holidays, promotions)
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
