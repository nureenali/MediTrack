from __future__ import annotations

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "salesdaily.csv"
CATEGORIES = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]


def get_category_columns() -> List[str]:
    return CATEGORIES.copy()


def load_sales_data(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or DATA_FILE
    df = pd.read_csv(path)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df = df.dropna(subset=["datum"])
    return df.sort_values("datum").reset_index(drop=True)


def build_monthly_sales(df: pd.DataFrame, category_columns: List[str] | None = None) -> pd.DataFrame:
    categories = category_columns or get_category_columns()
    work_df = df.copy()
    work_df["datum"] = pd.to_datetime(work_df["datum"], errors="coerce")
    work_df = work_df.dropna(subset=["datum"]).sort_values("datum").reset_index(drop=True)

    monthly = (
        work_df.groupby([work_df["datum"].dt.to_period("M")])[categories]
        .sum()
        .reset_index()
    )
    monthly.columns = ["ds", *categories]
    monthly["ds"] = monthly["ds"].dt.to_timestamp()
    monthly = monthly.sort_values("ds").reset_index(drop=True)
    return monthly


def prepare_prophet_frame(monthly: pd.DataFrame, category: str) -> pd.DataFrame:
    frame = monthly[["ds", category]].copy()
    frame.columns = ["ds", "y"]
    return frame


def train_category_model(monthly: pd.DataFrame, category: str, cutoff: str = "2018-06-30") -> Dict[str, object]:
    frame = prepare_prophet_frame(monthly, category)
    frame["ds"] = pd.to_datetime(frame["ds"])
    frame = frame.sort_values("ds").reset_index(drop=True)

    train_end = pd.Timestamp(cutoff)
    train_df = frame[frame["ds"] <= train_end].copy()
    validation_df = frame[frame["ds"] > train_end].copy()

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(validation_df), freq="MS")
    forecast = model.predict(future)

    validation_forecast = forecast[forecast["ds"].isin(validation_df["ds"])][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    comparison = validation_df[["ds", "y"]].merge(validation_forecast, on="ds", how="left")

    mae = mean_absolute_error(comparison["y"], comparison["yhat"])
    rmse = np.sqrt(mean_squared_error(comparison["y"], comparison["yhat"]))
    mean_actual = comparison["y"].mean()
    error_pct = (mae / mean_actual) * 100 if mean_actual > 0 else 0.0

    return {
        "model": model,
        "forecast": forecast,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "error_pct": error_pct,
        },
        "train_df": train_df,
        "validation_df": validation_df,
    }


def train_all_categories(data_path: Path | None = None, model_dir: Path | None = None, cutoff: str = "2018-06-30") -> Dict[str, Dict[str, object]]:
    df = load_sales_data(data_path)
    monthly = build_monthly_sales(df)

    model_dir = model_dir or MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}
    for category in get_category_columns():
        category_result = train_category_model(monthly, category, cutoff=cutoff)
        results[category] = category_result

        model_path = model_dir / f"{category}_prophet.pkl"
        forecast_path = model_dir / f"{category}_forecast.csv"
        joblib.dump(category_result["model"], model_path)
        category_result["forecast"].to_csv(forecast_path, index=False)

    return results


def load_trained_model(category: str, model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    model_path = model_dir / f"{category}_prophet.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"No trained model found for {category} at {model_path}")
    return joblib.load(model_path)


def generate_forecast(category: str, periods: int = 12, model_dir: Path | None = None, model=None) -> pd.DataFrame:
    model = model or load_trained_model(category, model_dir=model_dir)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    return model.predict(future)


def build_forecast_payload(category: str, periods: int = 12, model_dir: Path | None = None) -> Dict[str, object]:
    forecast_df = generate_forecast(category, periods=periods, model_dir=model_dir)
    forecast_tail = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).copy()
    forecast_tail["ds"] = forecast_tail["ds"].dt.strftime("%Y-%m-%d")
    return {
        "category": category,
        "periods": periods,
        "forecast": [
            {
                "ds": row.ds,
                "yhat": float(row.yhat),
                "yhat_lower": float(row.yhat_lower),
                "yhat_upper": float(row.yhat_upper),
            }
            for row in forecast_tail.itertuples(index=False)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Prophet forecasts for pharmacy sales categories")
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Path to the sales daily CSV file")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="Directory to store trained models and forecasts")
    parser.add_argument("--cutoff", default="2018-06-30", help="Training cutoff date for validation")
    parser.add_argument("--category", choices=get_category_columns(), help="Train a single category instead of all categories")
    parser.add_argument("--forecast-only", action="store_true", help="Generate a forecast using an existing model without retraining")
    parser.add_argument("--periods", type=int, default=12, help="Number of months to forecast when using --forecast-only")
    parser.add_argument("--json", action="store_true", help="Print the forecast as JSON for API-style usage")
    args = parser.parse_args()

    if args.forecast_only:
        if not args.category:
            raise SystemExit("--category is required when using --forecast-only")
        if args.json:
            payload = build_forecast_payload(args.category, periods=args.periods, model_dir=args.model_dir)
            print(json.dumps(payload, indent=2))
            return

        forecast = generate_forecast(args.category, periods=args.periods, model_dir=args.model_dir)
        print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(args.periods))
        return

    if args.category:
        df = load_sales_data(args.data)
        monthly = build_monthly_sales(df)
        result = train_category_model(monthly, args.category, cutoff=args.cutoff)
        model_path = args.model_dir / f"{args.category}_prophet.pkl"
        forecast_path = args.model_dir / f"{args.category}_forecast.csv"
        args.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["model"], model_path)
        result["forecast"].to_csv(forecast_path, index=False)
        print(f"Trained and saved model for {args.category}")
        print(result["metrics"])
        return

    results = train_all_categories(data_path=args.data, model_dir=args.model_dir, cutoff=args.cutoff)
    print(f"Trained {len(results)} categories")
    for category, result in results.items():
        print(category, result["metrics"])


if __name__ == "__main__":
    main()
