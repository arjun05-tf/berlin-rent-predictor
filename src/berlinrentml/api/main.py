"""FastAPI application for BerlinRentML."""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from berlinrentml.config import MODELS_DIR
from berlinrentml.modeling.training import load_model_artifacts
from berlinrentml.features import FeatureEngineer

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
    model_name: str = Field(..., description="Model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information."""

    model_type: str
    features: List[str]
    n_features: int
    loaded_at: str


@app.on_event("startup")
async def load_model():
    """Load model artifacts on startup."""
    global model, preprocessor, feature_names, model_loaded

    try:
        model, preprocessor, feature_names = load_model_artifacts(
            model_name="final_model",
            model_dir=MODELS_DIR,
        )
        model_loaded = True
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Run training script first: python scripts/train.py")
        model_loaded = False


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
        # Convert request to DataFrame
        import pandas as pd

        input_dict = request.model_dump()
        df = pd.DataFrame([input_dict])

        # Engineer features
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)

        # Ensure all required features are present
        for feat in feature_names:
            if feat not in df_features.columns:
                df_features[feat] = None  # Will be handled by preprocessor

        # Select only the features the model was trained on
        df_features = df_features[feature_names]

        # Preprocess
        X_processed = preprocessor.transform(df_features)

        # Predict
        prediction = model.predict(X_processed)[0]

        # Round to 2 decimal places
        prediction = round(float(prediction), 2)

        return PredictionResponse(
            predicted_rent=prediction,
            model_name=model.__class__.__name__,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print("Starting BerlinRentML API...")
    print("Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
