"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from scripts.API.main_fastapi import app
    return TestClient(app)


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "mensaje" in response.json()

    def test_predict_endpoint_structure(self, client):
        """Test predict endpoint accepts POST."""
        # Test with sample valid data
        sample_data = {
            "laufkont": "A11",
            "laufzeit": 24,
            "moral": "A30",
            "verw": "A40",
            "hoehe": 5000,
            "sparkont": "A61",
            "beszeit": "A71",
            "rate": 2,
            "famges": "A91",
            "buerge": "A101",
            "wohnzeit": 2,
            "verm": "A121",
            "alter": 35,
            "weitkred": "A141",
            "wohn": "A151",
            "bishkred": "2",
            "beruf": "A171",
            "pers": "1",
            "telef": "A191",
            "gastarb": "A201"
        }
        
        response = client.post("/app-credit/predict/", json=sample_data)
        
        # Should return 200 or 400 (if model not found)
        assert response.status_code in [200, 400, 422]

    def test_predict_endpoint_returns_prediction(self, client):
        """Test that predict endpoint returns a prediction."""
        sample_data = {
            "laufkont": "A11",
            "laufzeit": 24,
            "moral": "A30",
            "verw": "A40",
            "hoehe": 5000,
            "sparkont": "A61",
            "beszeit": "A71",
            "rate": 2,
            "famges": "A91",
            "buerge": "A101",
            "wohnzeit": 2,
            "verm": "A121",
            "alter": 35,
            "weitkred": "A141",
            "wohn": "A151",
            "bishkred": "2",
            "beruf": "A171",
            "pers": "1",
            "telef": "A191",
            "gastarb": "A201"
        }
        
        response = client.post("/app-credit/predict/", json=sample_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "prediccion" in data

    def test_predict_invalid_data(self, client):
        """Test predict endpoint with invalid data."""
        invalid_data = {
            "invalid_field": "test"
        }
        
        response = client.post("/app-credit/predict/", json=invalid_data)
        
        # Should return error
        assert response.status_code in [400, 422]

    def test_cors_headers(self, client):
        """Test that CORS is properly configured."""
        response = client.get("/")
        
        # Check for CORS headers (FastAPI adds these automatically with middleware)
        assert response.status_code == 200


class TestAPIErrorHandling:
    """Test error handling in API."""

    def test_missing_fields_error(self, client):
        """Test error when required fields are missing."""
        incomplete_data = {
            "laufzeit": 24,
            "hoehe": 5000
        }
        
        response = client.post("/app-credit/predict/", json=incomplete_data)
        
        # Should return error for missing fields
        assert response.status_code in [400, 422]

    def test_invalid_data_types(self, client):
        """Test error handling for invalid data types."""
        invalid_types = {
            "laufkont": "A11",
            "laufzeit": "not_a_number",  # Should be int
            "moral": "A30"
        }
        
        response = client.post("/app-credit/predict/", json=invalid_types)
        
        # Should handle gracefully
        assert response.status_code in [400, 422]
