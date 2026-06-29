from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from ml.train_forecasts import build_forecast_payload, get_category_columns

app = FastAPI(title="MediTrack Forecast API", version="1.0.0")

MODEL_DIR = Path(__file__).resolve().parent / "models"


class ForecastRequest(BaseModel):
    category: str
    periods: int = 12


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/categories")
def categories() -> dict:
    return {"categories": get_category_columns()}


@app.post("/forecast")
def forecast(request: ForecastRequest) -> dict:
    if request.category not in get_category_columns():
        raise HTTPException(status_code=404, detail=f"Unknown category: {request.category}")

    if request.periods <= 0:
        raise HTTPException(status_code=400, detail="periods must be greater than 0")

    return build_forecast_payload(request.category, periods=request.periods, model_dir=MODEL_DIR)
