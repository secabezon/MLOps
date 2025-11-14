from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import pandas as pd
from scripts.main import main, main_predict



router = APIRouter()

ruta_actual = os.getcwd()

@router.get("/base")
def fun_ruta_actual():
    return {"mensaje": "Probando el route, todo OK"}

@router.get("/ruta-actual")
def fun_ruta_actual():
    return {f"mensaje: {ruta_actual}"} #/usr/src/app


router = APIRouter()

@router.post("/predict/")
def predict(body: dict):

    try:
        resultado = main_predict(body)
        return {"prediccion": resultado}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/predict-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    path = f"./{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    try:
        resultado = main_predict(path)
        os.remove(path)
        return {"predicciones": resultado}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))