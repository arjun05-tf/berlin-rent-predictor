"""FastAPI application for BerlinRentML."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from berlinrentml.config import MODELS_DIR
from berlinrentml.modeling.artifacts import load_conformal, load_model_artifacts
from berlinrentml.inference import predict_rent

# Initialize FastAPI app
app = FastAPI(
    title="BerlinRentML API",
    description="Spatially Robust Berlin Rental Price Prediction API",
    version="0.1.0",
)

# Global variables for model artifacts
model = None
preprocessor = None
feature_names = None
model_loaded = False
interval = None
load_error = None

REQUEST_LOG = Path(__file__).resolve().parents[3] / "logs" / "requests.jsonl"


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    livingSpace: float = Field(..., gt=0, le=500, description="Living space in m²")
    rooms: float = Field(..., gt=0, le=20, description="Number of rooms")
    floor: Optional[int] = Field(None, ge=0, le=50, description="Floor number")
    yearConstructed: Optional[int] = Field(None, ge=1800, le=2030, description="Construction year")
    geo_plz: Optional[str] = Field(None, description="Postal code (PLZ)")
    geo_bln: Optional[str] = Field(None, description="Berlin district")
    heatingType: Optional[str] = Field(None, description="Heating type")
    condition: Optional[str] = Field(None, description="Property condition")
    hasKitchen: Optional[bool] = Field(None, description="Has kitchen")
    hasBalcony: Optional[bool] = Field(None, description="Has balcony")
    hasGarden: Optional[bool] = Field(None, description="Has garden")
    cellar: Optional[bool] = Field(None, description="Has cellar")

    @field_validator("livingSpace")
    def validate_living_space(cls, v):
        if v < 10:
            raise ValueError("Living space must be at least 10 m²")
        return v

    @field_validator("rooms")
    def validate_rooms(cls, v):
        if v < 0.5:
            raise ValueError("Rooms must be at least 0.5")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    predicted_rent: float = Field(..., description="Predicted monthly cold rent in €")
    interval_low: Optional[float] = Field(None, description="Lower bound of the 90% prediction interval in €")
    interval_high: Optional[float] = Field(None, description="Upper bound of the 90% prediction interval in €")
    model_name: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information."""

    model_type: str
    features: List[str]
    n_features: int
    loaded_at: str


def _log_request(data: dict, prediction: float) -> None:
    """Append request + prediction to a JSONL file for drift monitoring (best effort)."""
    try:
        REQUEST_LOG.parent.mkdir(exist_ok=True)
        with REQUEST_LOG.open("a", encoding="utf8") as f:
            f.write(json.dumps({**data, "prediction": prediction, "ts": datetime.utcnow().isoformat()}) + "\n")
    except OSError:
        pass


def load_model() -> None:
    """Load model artifacts into module globals."""
    global model, preprocessor, feature_names, model_loaded, interval, load_error

    try:
        model, preprocessor, feature_names = load_model_artifacts(
            model_name="final_model",
            model_dir=MODELS_DIR,
        )
        interval = load_conformal(MODELS_DIR)
        model_loaded = True
        print("Model loaded successfully")
    except Exception as e:
        load_error = f"{type(e).__name__}: {e}"[:300]
        print(f"Failed to load model: {e}")
        print("Run training script first: python scripts/train.py")
        model_loaded = False


@app.on_event("startup")
async def _startup():
    load_model()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BerlinRentML API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        **({"error": load_error} if load_error else {}),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_type=model.__class__.__name__,
        features=feature_names,
        n_features=len(feature_names),
        loaded_at=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a prediction."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        result = predict_rent(request.model_dump(), model, preprocessor, feature_names, interval)
        prediction, low, high = result if interval else (result, None, None)
        _log_request(request.model_dump(), prediction)

        return PredictionResponse(
            predicted_rent=prediction,
            interval_low=low,
            interval_high=high,
            model_name="LGBMRegressor",
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
