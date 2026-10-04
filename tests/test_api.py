"""API tests."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from berlinrentml.api.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_valid_request():
    """Test prediction with valid request."""
    # This test will fail if model is not loaded, which is expected
    request_data = {
        "livingSpace": 60.0,
        "rooms": 2.5,
        "floor": 3,
        "yearConstructed": 2000,
        "geo_plz": "10115",
        "hasKitchen": True,
        "hasBalcony": True,
    }

    response = client.post("/predict", json=request_data)

    # Either succeeds (if model loaded) or returns 503 (model not loaded)
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "predicted_rent" in data
        assert "model_name" in data
        assert isinstance(data["predicted_rent"], float)


def test_predict_invalid_living_space():
    """Test prediction with invalid living space."""
    request_data = {
        "livingSpace": 5.0,  # Too small
        "rooms": 2.0,
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422  # Validation error


def test_predict_invalid_rooms():
    """Test prediction with invalid rooms."""
    request_data = {
        "livingSpace": 60.0,
        "rooms": 0.0,  # Too small
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422


def test_predict_missing_required_field():
    """Test prediction with missing required field."""
    request_data = {
        "livingSpace": 60.0,
        # Missing rooms
    }

    response = client.post("/predict", json=request_data)
    assert response.status_code == 422


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")

    # Either succeeds (if model loaded) or returns 503
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "model_type" in data
        assert "features" in data
        assert "n_features" in data
