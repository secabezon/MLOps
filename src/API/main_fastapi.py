from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from . import my_routes

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(my_routes.router, prefix="/app-credit")

@app.get('/')
def health():
    return {
        "mensaje": "Bienvenido a mi Proyecto de del Tec de Monterrey"
    }