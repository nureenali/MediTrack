from pathlib import Path

import pandas as pd

from ml.train_forecasts import build_monthly_sales, build_forecast_payload, get_category_columns


def test_build_monthly_sales_aggregates_and_resets_index():
    data_path = Path("ml/data/salesdaily.csv")
    df = pd.read_csv(data_path)

    monthly = build_monthly_sales(df)

    assert isinstance(monthly, pd.DataFrame)
    assert monthly.columns[0] == "ds"
    assert "M01AB" in monthly.columns
    assert pd.api.types.is_datetime64_any_dtype(monthly["ds"])
    assert monthly.shape[0] > 0


def test_get_category_columns_returns_expected_categories():
    categories = get_category_columns()

    assert categories == ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"]


def test_build_forecast_payload_returns_serializable_forecast_data():
    payload = build_forecast_payload("M01AB", periods=3, model_dir=Path("ml/models"))

    assert payload["category"] == "M01AB"
    assert payload["periods"] == 3
    assert len(payload["forecast"]) == 3
    assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(payload["forecast"][0].keys())
