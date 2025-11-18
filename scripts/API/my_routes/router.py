"""API routes for credit risk prediction."""
from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import pandas as pd
from scripts.main import main, main_predict
from scripts.API.schemas import CreditInput, PredictionResponse, PredictionBatchResponse


router = APIRouter()

ruta_actual = os.getcwd()

@router.get("/base")
def fun_ruta_actual():
    """Test endpoint to verify router is working."""
    return {"mensaje": "Probando el route, todo OK"}

@router.get("/ruta-actual")
def fun_ruta_actual_path():
    """Get current working directory."""
    return {"mensaje": ruta_actual}  # /usr/src/app


@router.post("/predict/", response_model=PredictionResponse)
def predict(data: CreditInput):
    """
    Predict credit risk for a single applicant.
    
    Args:
        data: Credit applicant data with 20 validated fields
        
    Returns:
        Prediction result (0=good credit, 1=bad credit)
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert Pydantic model to dict for main_predict
        body = data.model_dump()
        resultado = main_predict(body)
        return {"prediccion": resultado, "confidence": None}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@router.post("/predict-csv/", response_model=PredictionBatchResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Predict credit risk for multiple applicants from CSV file.
    
    Args:
        file: CSV file with applicant data
        
    Returns:
        List of predictions and total count
        
    Raises:
        HTTPException: If file is not CSV or prediction fails
    """
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    path = f"./{file.filename}"
    try:
        # Save uploaded file
        with open(path, "wb") as f:
            f.write(await file.read())
        
        # Make predictions
        resultado = main_predict(path)
        
        # Clean up
        os.remove(path)
        
        # Convert to list if needed
        if isinstance(resultado, (list, pd.Series)):
            predicciones = list(resultado)
        else:
            predicciones = [resultado]
        
        return {
            "predicciones": predicciones,
            "total": len(predicciones)
        }
    except Exception as e:
        # Clean up on error
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=400, detail=f"CSV processing error: {str(e)}")