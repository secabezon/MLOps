from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from . import my_routes
from .schemas import HealthResponse

app = FastAPI(
    title="MLOps Credit Risk API",
    description="API para predicción de riesgo crediticio usando modelos de Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

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

app.include_router(my_routes.router, prefix="/app-credit", tags=["Predictions"])

@app.get('/', response_model=HealthResponse, tags=["Health"])
def health():
    """
    Health check endpoint.
    
    Returns:
        Service status and metadata
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": True
    }