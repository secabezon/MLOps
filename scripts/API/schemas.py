"""Pydantic models for API request/response validation."""
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Union


class CreditInput(BaseModel):
    """Input schema for credit risk prediction."""
    
    laufkont: str = Field(..., description="Status of existing checking account")
    laufzeit: int = Field(..., ge=1, le=100, description="Duration in months")
    moral: str = Field(..., description="Credit history")
    verw: str = Field(..., description="Purpose of credit")
    hoehe: int = Field(..., ge=0, description="Credit amount")
    sparkont: str = Field(..., description="Savings account/bonds")
    beszeit: str = Field(..., description="Present employment since")
    rate: int = Field(..., ge=1, le=4, description="Installment rate in percentage")
    famges: str = Field(..., description="Personal status and sex")
    buerge: str = Field(..., description="Other debtors / guarantors")
    wohnzeit: int = Field(..., ge=1, le=4, description="Present residence since")
    verm: str = Field(..., description="Property")
    alter: int = Field(..., ge=18, le=100, description="Age in years")
    weitkred: str = Field(..., description="Other installment plans")
    wohn: str = Field(..., description="Housing")
    bishkred: str = Field(..., description="Number of existing credits at this bank")
    beruf: str = Field(..., description="Job")
    pers: str = Field(..., description="Number of people being liable")
    telef: str = Field(..., description="Telephone")
    gastarb: str = Field(..., description="Foreign worker")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""
    
    prediccion: Union[int, float, str] = Field(..., description="Predicted credit risk (0=good, 1=bad)")
    confidence: float = Field(None, ge=0, le=1, description="Prediction confidence (optional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediccion": 0,
                "confidence": 0.85
            }
        }


class PredictionBatchResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predicciones: list = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicciones": [0, 1, 0],
                "total": 3
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="API version")
    model_loaded: bool = Field(default=True, description="Whether model is loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True
            }
        }
