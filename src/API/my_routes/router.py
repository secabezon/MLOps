from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import pandas as pd
from ....scripts.main import main_predict
from ....scripts.API.schemas import CreditInput, PredictionResponse, PredictionBatchResponse


router = APIRouter()

ruta_actual = os.getcwd()

@router.get("/base")
def fun_ruta_actual():
    return {"mensaje": "Probando el route, todo OK"}

@router.get("/ruta-actual")
def fun_ruta_actual():
    return {f"mensaje: {ruta_actual}"} #/usr/src/app


@router.post("/predict/", response_model=PredictionResponse)
def predict(data: CreditInput):
    """
    Predict credit risk for a single applicant.
    
    Args:
        data: Credit applicant information
        
    Returns:
        Prediction result (0=good credit, 1=bad credit)
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        # Convert Pydantic model to dict
        input_dict = data.model_dump()
        resultado = main_predict(input_dict)
        return {"prediccion": resultado}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@router.post("/predict-csv/", response_model=PredictionBatchResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Predict credit risk for multiple applicants from CSV file.
    
    Args:
        file: CSV file with applicant data
        
    Returns:
        Batch prediction results
        
    Raises:
        HTTPException: If file is invalid or prediction fails
    """
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    path = f"./{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        resultado = main_predict(path)
        os.remove(path)
        
        # Convert to list if needed
        if hasattr(resultado, 'tolist'):
            resultado = resultado.tolist()
        elif not isinstance(resultado, list):
            resultado = [resultado]
            
        return {
            "predicciones": resultado,
            "total": len(resultado)
        }
    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
