# MediTrack

This repository contains pharmacy inventory and demand forecasting work.

## ML workflow

### Trained models
- Saved Prophet models: `ml/models/*_prophet.pkl`
- Saved forecast CSVs: `ml/models/*_forecast.csv`

### Training script
- `ml/train_forecasts.py`
- Supports:
  - full training: `python ml/train_forecasts.py`
  - single-category training: `python ml/train_forecasts.py --category M01AB`
  - forecast only from saved model: `python ml/train_forecasts.py --forecast-only --category M01AB --periods 12`
  - JSON output: `python ml/train_forecasts.py --forecast-only --category M01AB --periods 12 --json`

### API
- `ml/api.py` exposes a FastAPI service
- Run the API:
  - `python -m uvicorn ml.api:app --host 127.0.0.1 --port 8000`
- Available endpoints:
  - `GET /health`
  - `GET /categories`
  - `POST /forecast`

### Forecast request body
```json
{
  "category": "M01AB",
  "periods": 12
}
```

## Dependencies
- install with `pip install -r ml/requirements.txt`

## Notes
- The model training uses monthly aggregation from `ml/data/salesdaily.csv`
- The current categories are: `M01AB`, `M01AE`, `N02BA`, `N02BE`, `N05B`, `N05C`, `R03`, `R06`
