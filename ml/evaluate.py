"""
Model evaluation and diagnostics for Prophet forecasts.

Provides metrics, residual diagnostics, model comparison, and cross-validation tools.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import kpss


class ForecastMetrics:
    """Compute standard forecast evaluation metrics."""

    @staticmethod
    def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
        """Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
        """Mean Absolute Percentage Error (%)."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100)

    @staticmethod
    def smape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
        """Symmetric Mean Absolute Percentage Error (%)."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred))
        mask = denominator != 0
        if not mask.any():
            return 0.0
        return float(100.0 * np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]))

    @staticmethod
    def r2(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
        """R-squared (Coefficient of Determination)."""
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def mase(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, 
             y_train: np.ndarray | pd.Series | None = None, m: int = 12) -> float:
        """
        Mean Absolute Scaled Error.
        
        Args:
            y_true: actual values
            y_pred: predicted values
            y_train: training set (for computing scale)
            m: seasonal period (default 12 for monthly data)
        
        Returns:
            MASE score (values < 1 indicate better than naive seasonal forecast)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if y_train is None:
            # Use naive seasonal forecast on y_true itself as fallback
            y_train = y_true
        
        y_train = np.asarray(y_train)
        scale = np.mean(np.abs(np.diff(y_train, n=m)))
        
        if scale == 0:
            # Handle edge case: constant training series
            return float(np.mean(np.abs(y_true - y_pred)))
        
        return float(np.mean(np.abs(y_true - y_pred)) / scale)

    @staticmethod
    def evaluate_all(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, 
                    y_train: np.ndarray | pd.Series | None = None) -> Dict[str, float]:
        """Compute all standard metrics at once."""
        return {
            "MAE": ForecastMetrics.mae(y_true, y_pred),
            "RMSE": ForecastMetrics.rmse(y_true, y_pred),
            "MAPE%": ForecastMetrics.mape(y_true, y_pred),
            "sMAPE%": ForecastMetrics.smape(y_true, y_pred),
            "R2": ForecastMetrics.r2(y_true, y_pred),
            "MASE": ForecastMetrics.mase(y_true, y_pred, y_train=y_train),
        }


class ResidualDiagnostics:
    """Analyze residuals and model fit quality."""

    @staticmethod
    def compute_residuals(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> np.ndarray:
        """Compute residuals: y_true - y_pred."""
        return np.asarray(y_true) - np.asarray(y_pred)

    @staticmethod
    def ljung_box_test(residuals: np.ndarray | pd.Series, lags: int = 10) -> pd.DataFrame:
        """
        Ljung-Box test for autocorrelation.
        H0: residuals are white noise (no autocorrelation).
        p-value > 0.05 suggests white noise (good fit).
        """
        return acorr_ljungbox(residuals, lags=[lags], return_df=True)

    @staticmethod
    def kpss_stationarity_test(residuals: np.ndarray | pd.Series) -> Tuple[float, float]:
        """
        KPSS stationarity test.
        H0: series is stationary.
        p-value > 0.05 suggests stationarity (good for residuals).
        """
        stat, pvalue, _, _ = kpss(residuals, regression='c', nlags='auto')
        return stat, pvalue

    @staticmethod
    def residual_summary(residuals: np.ndarray | pd.Series) -> Dict[str, float]:
        """Compute summary statistics for residuals."""
        res = np.asarray(residuals)
        return {
            "Mean": float(np.mean(res)),
            "Std": float(np.std(res)),
            "Min": float(np.min(res)),
            "Max": float(np.max(res)),
            "Skewness": float(pd.Series(res).skew()),
            "Kurtosis": float(pd.Series(res).kurtosis()),
        }

    @staticmethod
    def plot_diagnostics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, 
                        lags: int = 40, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Create a 2x2 diagnostic plot panel.
        
        Returns:
            matplotlib Figure with residual plots
        """
        residuals = ResidualDiagnostics.compute_residuals(y_true, y_pred)
        
        # Adjust lags based on residual length
        max_lags = min(lags, len(residuals) // 2 - 1)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Residuals over time
        axes[0, 0].plot(residuals, marker='o', linestyle='-', alpha=0.7)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Histogram of Residuals')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals vs fitted
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residuals vs Fitted Values')
        axes[1, 0].set_xlabel('Fitted')
        axes[1, 0].set_ylabel('Residual')
        axes[1, 0].grid(True, alpha=0.3)
        
        # ACF of residuals
        if max_lags > 0:
            plot_acf(residuals, lags=max_lags, ax=axes[1, 1])
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough data for ACF', ha='center', va='center')
        axes[1, 1].set_title('ACF of Residuals')
        
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_actual_vs_fitted(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, 
                             dates: pd.Series | None = None, 
                             figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """Plot actual vs predicted over time."""
        fig, ax = plt.subplots(figsize=figsize)
        
        x = range(len(y_true))
        if dates is not None:
            x = dates
        
        ax.plot(x, y_true, marker='o', linestyle='-', label='Actual', alpha=0.7)
        ax.plot(x, y_pred, marker='s', linestyle='--', label='Predicted', alpha=0.7)
        ax.fill_between(range(len(y_pred)), y_pred, y_true, alpha=0.2, color='gray')
        ax.set_title('Actual vs Predicted')
        ax.set_ylabel('Sales')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class ProphetEvaluator:
    """Convenience wrapper for evaluating Prophet models."""

    @staticmethod
    def evaluate_forecast(forecast_df: pd.DataFrame, actual_df: pd.DataFrame, 
                         category: str | None = None) -> Dict[str, object]:
        """
        Evaluate a Prophet forecast against actuals.
        
        Args:
            forecast_df: Prophet forecast output (with 'ds', 'yhat', 'yhat_lower', 'yhat_upper')
            actual_df: DataFrame with 'ds' and 'y' columns
            category: optional category name for reporting
        
        Returns:
            Dictionary with metrics, residuals, and diagnostics
        """
        # Merge forecast with actuals
        comparison = actual_df[['ds', 'y']].merge(
            forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
            on='ds', 
            how='inner'
        )
        
        if len(comparison) == 0:
            raise ValueError("No overlapping dates between forecast and actuals")
        
        y_true = comparison['y'].values
        y_pred = comparison['yhat'].values
        
        # Metrics
        metrics = ForecastMetrics.evaluate_all(y_true, y_pred)
        
        # Residuals and diagnostics
        residuals = ResidualDiagnostics.compute_residuals(y_true, y_pred)
        residual_summary = ResidualDiagnostics.residual_summary(residuals)
        
        # Ljung-Box test (white noise check)
        lb_result = ResidualDiagnostics.ljung_box_test(residuals, lags=10)
        lb_pvalue = float(lb_result['lb_pvalue'].values[0])
        
        # KPSS test (stationarity check)
        kpss_stat, kpss_pvalue = ResidualDiagnostics.kpss_stationarity_test(residuals)
        
        # Prediction interval coverage
        lower = comparison['yhat_lower'].values
        upper = comparison['yhat_upper'].values
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        
        return {
            "category": category,
            "n_samples": len(comparison),
            "metrics": metrics,
            "residuals": residuals,
            "residual_summary": residual_summary,
            "ljung_box_pvalue": lb_pvalue,
            "is_white_noise": lb_pvalue > 0.05,
            "kpss_pvalue": kpss_pvalue,
            "prediction_interval_coverage": float(coverage),
            "comparison": comparison,
        }

    @staticmethod
    def evaluate_and_plot(forecast_df: pd.DataFrame, actual_df: pd.DataFrame, 
                         category: str | None = None, save_path: Path | None = None) -> Dict[str, object]:
        """
        Evaluate forecast and generate diagnostic plots.
        
        Args:
            forecast_df: Prophet forecast output
            actual_df: DataFrame with 'ds' and 'y'
            category: category name for reporting
            save_path: optional path to save plots as PNG
        
        Returns:
            Dictionary with evaluation results and matplotlib figures
        """
        evaluation = ProphetEvaluator.evaluate_forecast(forecast_df, actual_df, category=category)
        
        y_true = evaluation['comparison']['y'].values
        y_pred = evaluation['comparison']['yhat'].values
        dates = evaluation['comparison']['ds'].values
        
        # Create diagnostic plots
        fig_diagnostics = ResidualDiagnostics.plot_diagnostics(y_true, y_pred)
        fig_actual_vs_fitted = ResidualDiagnostics.plot_actual_vs_fitted(y_true, y_pred, dates=dates)
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            category_str = category or "model"
            fig_diagnostics.savefig(save_path / f"{category_str}_diagnostics.png", dpi=100, bbox_inches='tight')
            fig_actual_vs_fitted.savefig(save_path / f"{category_str}_actual_vs_fitted.png", dpi=100, bbox_inches='tight')
        
        evaluation["figures"] = {
            "diagnostics": fig_diagnostics,
            "actual_vs_fitted": fig_actual_vs_fitted,
        }
        
        return evaluation


class ModelComparison:
    """Compare multiple models or forecasts."""

    @staticmethod
    def compare_categories(evaluations: Dict[str, Dict[str, object]]) -> pd.DataFrame:
        """
        Create a comparison table across multiple categories.
        
        Args:
            evaluations: dict mapping category name to evaluation result
        
        Returns:
            DataFrame with metrics for each category
        """
        rows = []
        for category, result in evaluations.items():
            row = {"Category": category}
            row.update(result.get("metrics", {}))
            row["Is_White_Noise"] = result.get("is_white_noise", False)
            row["PI_Coverage"] = result.get("prediction_interval_coverage", 0)
            rows.append(row)
        
        return pd.DataFrame(rows).sort_values("RMSE")

    @staticmethod
    def diebold_mariano_test(y_true: np.ndarray | pd.Series, 
                            forecast1: np.ndarray | pd.Series, 
                            forecast2: np.ndarray | pd.Series, 
                            loss: str = "mae") -> Dict[str, float]:
        """
        Diebold-Mariano test for comparing two forecasts.
        
        Args:
            y_true: actual values
            forecast1: predictions from model 1
            forecast2: predictions from model 2
            loss: 'mae' or 'mse' for loss function
        
        Returns:
            Dictionary with DM statistic and p-value.
            Positive DM suggests forecast2 is better.
        """
        y_true = np.asarray(y_true)
        forecast1 = np.asarray(forecast1)
        forecast2 = np.asarray(forecast2)
        
        if loss == "mae":
            e1 = np.abs(y_true - forecast1)
            e2 = np.abs(y_true - forecast2)
        elif loss == "mse":
            e1 = (y_true - forecast1) ** 2
            e2 = (y_true - forecast2) ** 2
        else:
            raise ValueError(f"Unknown loss function: {loss}")
        
        d = e1 - e2
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)
        
        if var_d == 0:
            return {"DM_statistic": 0.0, "p_value": 1.0}
        
        dm_stat = mean_d / np.sqrt(var_d / len(d))
        
        # Approximate p-value using normal distribution
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
        
        return {
            "DM_statistic": float(dm_stat),
            "p_value": float(p_value),
            "mean_error_diff": float(mean_d),
        }


def print_evaluation_report(evaluation: Dict[str, object], verbose: bool = True) -> str:
    """
    Format evaluation results as a text report.
    
    Args:
        evaluation: evaluation result from ProphetEvaluator
        verbose: include residual summary and tests
    
    Returns:
        Formatted string report
    """
    lines = []
    
    if evaluation.get("category"):
        lines.append(f"\n{'='*60}")
        lines.append(f"Model Evaluation: {evaluation['category']}")
        lines.append(f"{'='*60}")
    
    lines.append(f"\nSample Size: {evaluation['n_samples']}")
    
    # Metrics
    lines.append("\nForecast Metrics:")
    lines.append("-" * 40)
    for metric, value in evaluation.get("metrics", {}).items():
        lines.append(f"  {metric:12s}: {value:10.4f}")
    
    # Prediction interval coverage
    pi_coverage = evaluation.get("prediction_interval_coverage", 0)
    lines.append(f"\n  PI Coverage: {pi_coverage:10.1%}")
    
    # Residual summary
    if verbose:
        lines.append("\nResidual Summary:")
        lines.append("-" * 40)
        for stat, value in evaluation.get("residual_summary", {}).items():
            lines.append(f"  {stat:12s}: {value:10.4f}")
    
    # Diagnostic tests
    if verbose:
        lines.append("\nDiagnostic Tests:")
        lines.append("-" * 40)
        is_wn = evaluation.get("is_white_noise", False)
        lines.append(f"  Ljung-Box (p)  : {evaluation.get('ljung_box_pvalue', 0):.4f} {'✓' if is_wn else '✗'} (white noise)")
        kpss_p = evaluation.get("kpss_pvalue", 0)
        lines.append(f"  KPSS (p)       : {kpss_p:.4f} {'✓' if kpss_p > 0.05 else '✗'} (stationarity)")
    
    return "\n".join(lines)
