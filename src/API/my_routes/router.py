from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import pandas as pd
from ....scripts.main import main


router = APIRouter()

ruta_actual = os.getcwd()

@router.get("/base")
def fun_ruta_actual():
    return {"mensaje": "Probando el route, todo OK"}

@router.get("/ruta-actual")
def fun_ruta_actual():
    return {f"mensaje: {ruta_actual}"} #/usr/src/app


@router.post("/predict/")
def predict(body: dict):
    new_client = body.get(['laufkont','laufzeit','moral','verw','hoehe','sparkont','beszeit','rate','famges','buerge','wohnzeit','verm','alter','weitkred','wohn','bishkred','beruf','pers','telef','gastarb','kredit','mixed_type_col'])

    if not new_client:
        raise HTTPException(status_code=400, detail="One of the fields is not in the json.")

    resultado=main(new_client)

    return {"message": "pred", "prediccion": resultado}

@router.post("/predict-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="El archivo debe ser un CSV")

    # Guardar el archivo en el servidor (opcional)
    file_location = f"{ruta_actual}/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    resultado=main(file_location)
    os.remove(file_location)

    return {"message": "pred", "prediccion": resultado}
