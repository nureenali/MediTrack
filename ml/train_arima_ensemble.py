"""
ARIMA + ExponentialSmoothing Ensemble Model Trainer

Trains both ARIMA and ExponentialSmoothing models, combines predictions via ensemble,
and evaluates all three approaches (ARIMA, ExpSmoothing, Ensemble) vs Prophet.
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
from train_forecasts import load_sales_data, build_monthly_sales, get_category_columns
from evaluate import ForecastMetrics

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

ML_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = ML_DIR / "ensemble_models"
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)


class ARIMAModel:
    """ARIMA(1,1,1) model wrapper."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self.result = None
    
    def fit(self, y_train: pd.Series | np.ndarray):
        """Train ARIMA model."""
        self.model = ARIMA(y_train, order=self.order)
        self.result = self.model.fit()
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecast."""
        forecast = self.result.get_forecast(steps=steps)
        return forecast.predicted_mean.values
    
    def save(self, path: Path):
        """Save model to disk."""
        joblib.dump(self.result, path)
    
    @staticmethod
    def load(path: Path):
        """Load model from disk."""
        return joblib.load(path)


class ExpSmoothingModel:
    """Exponential Smoothing model wrapper."""
    
    def __init__(self, trend: str = 'add', seasonal=None):
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        self.result = None
    
    def fit(self, y_train: pd.Series | np.ndarray):
        """Train ExponentialSmoothing model."""
        self.model = ExponentialSmoothing(
            y_train, 
            trend=self.trend, 
            seasonal=self.seasonal,
            initialization_method='estimated'
        )
        self.result = self.model.fit(optimized=True)
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecast."""
        return self.result.forecast(steps=steps).values
    
    def save(self, path: Path):
        """Save model to disk."""
        joblib.dump(self.result, path)
    
    @staticmethod
    def load(path: Path):
        """Load model from disk."""
        return joblib.load(path)


class Ensemble:
    """Ensemble combining ARIMA + ExponentialSmoothing predictions."""
    
    def __init__(self, arima_model: ARIMAModel, exp_model: ExpSmoothingModel, 
                 weights: Tuple[float, float] = (0.5, 0.5)):
        """
        Args:
            arima_model: Trained ARIMA model
            exp_model: Trained ExponentialSmoothing model
            weights: (weight_arima, weight_exp) - must sum to 1
        """
        self.arima_model = arima_model
        self.exp_model = exp_model
        self.weights = np.array(weights) / np.sum(weights)  # Normalize
    
    def predict(self, steps: int) -> np.ndarray:
        """Predict by averaging ARIMA and ExpSmoothing forecasts."""
        arima_pred = self.arima_model.predict(steps)
        exp_pred = self.exp_model.predict(steps)
        
        # Weighted average
        ensemble_pred = self.weights[0] * arima_pred + self.weights[1] * exp_pred
        return ensemble_pred


def train_category_ensemble(monthly: pd.DataFrame, category: str, 
                           cutoff: str = "2018-06-30") -> Dict[str, object]:
    """
    Train ARIMA + ExponentialSmoothing ensemble for a category.
    
    Args:
        monthly: DataFrame with 'ds' and sales columns
        category: Category name
        cutoff: Train/test split date
    
    Returns:
        Dictionary with trained models and metrics
    """
    # Prepare data
    y_data = monthly[[category]].values.flatten()
    train_mask = monthly["ds"] <= pd.Timestamp(cutoff)
    y_train = y_data[train_mask]
    y_test = y_data[~train_mask]
    
    # Train ARIMA
    arima = ARIMAModel(order=(1, 1, 1))
    try:
        arima.fit(pd.Series(y_train))
        arima_pred = arima.predict(len(y_test))
        arima_mae = mean_absolute_error(y_test, arima_pred)
        arima_rmse = np.sqrt(mean_squared_error(y_test, arima_pred))
    except Exception as e:
        print(f"    ARIMA failed: {str(e)[:50]}")
        arima = None
        arima_pred = None
        arima_mae = np.nan
        arima_rmse = np.nan
    
    # Train ExponentialSmoothing
    exp = ExpSmoothingModel(trend='add', seasonal=None)
    try:
        exp.fit(pd.Series(y_train))
        exp_pred = exp.predict(len(y_test))
        exp_mae = mean_absolute_error(y_test, exp_pred)
        exp_rmse = np.sqrt(mean_squared_error(y_test, exp_pred))
    except Exception as e:
        print(f"    ExpSmoothing failed: {str(e)[:50]}")
        exp = None
        exp_pred = None
        exp_mae = np.nan
        exp_rmse = np.nan
    
    # Create ensemble if both models exist
    ensemble = None
    ensemble_pred = None
    ensemble_mae = np.nan
    ensemble_rmse = np.nan
    
    if arima is not None and exp is not None:
        ensemble = Ensemble(arima, exp, weights=(0.5, 0.5))
        ensemble_pred = ensemble.predict(len(y_test))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    return {
        "category": category,
        "y_train": y_train,
        "y_test": y_test,
        "arima_model": arima,
        "exp_model": exp,
        "ensemble_model": ensemble,
        "predictions": {
            "arima": arima_pred,
            "exp_smoothing": exp_pred,
            "ensemble": ensemble_pred,
        },
        "metrics": {
            "arima": {"MAE": arima_mae, "RMSE": arima_rmse},
            "exp_smoothing": {"MAE": exp_mae, "RMSE": exp_rmse},
            "ensemble": {"MAE": ensemble_mae, "RMSE": ensemble_rmse},
        },
    }


def train_all_ensemble(data_path: Path | None = None, cutoff: str = "2018-06-30") -> Dict[str, Dict]:
    """Train ensemble models for all categories."""
    df = load_sales_data(data_path)
    monthly = build_monthly_sales(df)
    
    results = {}
    for category in get_category_columns():
        print(f"  Training ensemble for {category}...", end=" ")
        result = train_category_ensemble(monthly, category, cutoff=cutoff)
        results[category] = result
        
        # Report
        metrics = result["metrics"]
        arima_mae = metrics["arima"]["MAE"]
        exp_mae = metrics["exp_smoothing"]["MAE"]
        ens_mae = metrics["ensemble"]["MAE"]
        print(f"ARIMA={arima_mae:.2f} | ExpSmooth={exp_mae:.2f} | Ensemble={ens_mae:.2f}")
    
    return results


def save_ensemble(result: Dict, category: str):
    """Save ensemble models to disk."""
    cat_dir = ENSEMBLE_DIR / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    
    if result["arima_model"] is not None:
        result["arima_model"].save(cat_dir / "arima.pkl")
    if result["exp_model"] is not None:
        result["exp_model"].save(cat_dir / "exp_smoothing.pkl")
    # Note: Ensemble is recreated from arima + exp on load, so no need to save separately


def load_ensemble(category: str) -> Dict:
    """Load trained ensemble for a category."""
    cat_dir = ENSEMBLE_DIR / category
    
    arima = None
    exp = None
    ensemble = None
    
    try:
        arima_result = ARIMAModel.load(cat_dir / "arima.pkl")
        arima = ARIMAModel()
        arima.result = arima_result
    except Exception as e:
        print(f"    Could not load ARIMA for {category}: {e}")
    
    try:
        exp_result = ExpSmoothingModel.load(cat_dir / "exp_smoothing.pkl")
        exp = ExpSmoothingModel()
        exp.result = exp_result
    except Exception as e:
        print(f"    Could not load ExponentialSmoothing for {category}: {e}")
    
    # Recreate ensemble from loaded models
    if arima is not None and exp is not None:
        ensemble = Ensemble(arima, exp, weights=(0.5, 0.5))
    
    return {"arima": arima, "exp_smoothing": exp, "ensemble": ensemble}


def predict_ensemble(category: str, periods: int = 12) -> Dict[str, np.ndarray]:
    """
    Generate forecasts using all three models.
    
    Returns:
        Dict with 'arima', 'exp_smoothing', 'ensemble' predictions
    """
    models = load_ensemble(category)
    
    results = {}
    if models["arima"] is not None:
        results["arima"] = models["arima"].predict(periods)
    if models["exp_smoothing"] is not None:
        results["exp_smoothing"] = models["exp_smoothing"].predict(periods)
    if models["ensemble"] is not None:
        results["ensemble"] = models["ensemble"].predict(periods)
    
    return results


def compare_with_prophet(ensemble_results: Dict, prophet_results: Dict) -> pd.DataFrame:
    """Compare ensemble models vs Prophet on test set."""
    comparisons = []
    
    for category, ens_result in ensemble_results.items():
        prophet_metrics = prophet_results.get(category, {}).get("metrics", {})
        
        row = {
            "Category": category,
            # Ensemble models
            "ARIMA_MAE": ens_result["metrics"]["arima"]["MAE"],
            "ARIMA_RMSE": ens_result["metrics"]["arima"]["RMSE"],
            "ExpSmooth_MAE": ens_result["metrics"]["exp_smoothing"]["MAE"],
            "ExpSmooth_RMSE": ens_result["metrics"]["exp_smoothing"]["RMSE"],
            "Ensemble_MAE": ens_result["metrics"]["ensemble"]["MAE"],
            "Ensemble_RMSE": ens_result["metrics"]["ensemble"]["RMSE"],
            # Prophet (for comparison)
            "Prophet_MAE": prophet_metrics.get("mae", np.nan),
            "Prophet_RMSE": prophet_metrics.get("rmse", np.nan),
        }
        comparisons.append(row)
    
    df = pd.DataFrame(comparisons)
    
    # Add improvement columns
    df["Ensemble_vs_Prophet_MAE%"] = ((df["Prophet_MAE"] - df["Ensemble_MAE"]) / df["Prophet_MAE"] * 100).round(1)
    df["Best_Model"] = df[["ARIMA_MAE", "ExpSmooth_MAE", "Ensemble_MAE"]].idxmin(axis=1).str.replace("_MAE", "")
    
    return df.sort_values("Ensemble_MAE")


def main():
    print("\n" + "=" * 100)
    print("ARIMA + EXPONENTIAL SMOOTHING ENSEMBLE TRAINING")
    print("=" * 100)
    
    # Train ensemble models
    print("\n[1/3] Training ensemble models for all categories...")
    ensemble_results = train_all_ensemble(cutoff="2018-06-30")
    
    # Save models
    print("\n[2/3] Saving models...")
    for category, result in ensemble_results.items():
        save_ensemble(result, category)
    print(f"Saved to: {ENSEMBLE_DIR}")
    
    # Load Prophet results for comparison
    print("\n[3/3] Comparing with Prophet...")
    from train_forecasts import train_all_categories
    prophet_results = train_all_categories(cutoff="2018-06-30")
    
    # Create comparison table
    comparison = compare_with_prophet(ensemble_results, prophet_results)
    print("\n" + "=" * 100)
    print("COMPARISON: ARIMA + ExpSmoothing Ensemble vs Prophet")
    print("=" * 100)
    print(comparison.to_string(index=False))
    
    # Save comparison
    comparison_path = ML_DIR / "ensemble_vs_prophet_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"\nComparison saved to: {comparison_path}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    avg_ensemble_mae = comparison["Ensemble_MAE"].mean()
    avg_prophet_mae = comparison["Prophet_MAE"].mean()
    improvement = ((avg_prophet_mae - avg_ensemble_mae) / avg_prophet_mae * 100)
    
    print(f"Average Ensemble MAE:      {avg_ensemble_mae:.2f}")
    print(f"Average Prophet MAE:       {avg_prophet_mae:.2f}")
    print(f"Improvement:               {improvement:+.1f}%")
    print(f"\nBest ensemble approach: {comparison['Best_Model'].value_counts().idxmax()}")
    print(f"Models saved to: {ENSEMBLE_DIR}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
