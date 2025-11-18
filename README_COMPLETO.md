# MLOps - Proyecto de Riesgo Crediticio

## 📋 Descripción del Proyecto

Proyecto MLOps del Tec de Monterrey 2025 que implementa un pipeline completo de Machine Learning para predecir riesgo crediticio utilizando el dataset German Credit.

### Problemática
Evaluación automatizada de riesgo crediticio para mejorar la toma de decisiones en otorgamiento de créditos, reduciendo el riesgo de impago y optimizando la aprobación de solicitantes.

### Solución MLOps
Sistema de Machine Learning end-to-end con:
- Pipeline automatizado de datos y entrenamiento
- Servicio de predicciones vía API REST
- Monitoreo de data drift y performance
- Reproducibilidad y versionamiento con DVC/MLflow
- Despliegue containerizado con Docker

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐
│   Raw Data      │
│   (DVC)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing   │
│  Pipeline       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Model Training  │─────▶│   MLflow     │
│  (sklearn)      │      │  Tracking    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│ Model Artifact  │─────▶│     DVC      │
│   (.joblib)     │      │   Storage    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI        │
│  Service        │
│  (Docker)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predictions    │
└─────────────────┘
```

---

## 📦 Estructura del Proyecto

```
├── src/
│   ├── data/              # Scripts de procesamiento de datos
│   ├── features/          # Feature engineering (Preprocessor)
│   ├── models/            # Entrenamiento y predicción
│   │   └── artifacts/     # Modelos entrenados (.joblib)
│   ├── monitoring/        # Data drift y performance
│   └── pipelines/         # sklearn Pipelines
├── scripts/
│   ├── main.py            # CLI principal para entrenar/evaluar
│   ├── dvc_*.py           # Wrappers para DVC pipeline
│   └── API/
│       ├── main_fastapi.py  # Servidor FastAPI
│       ├── schemas.py       # Validación Pydantic
│       └── my_routes/       # Endpoints
├── tests/                 # Pruebas unitarias e integración
├── notebooks/             # Análisis exploratorio
├── docs/                  # Documentación adicional
├── Dockerfile             # Imagen Docker optimizada
├── dvc.yaml               # Pipeline DVC
├── requirements.txt       # Dependencias Python
└── README.md
```

---

## 🚀 Inicio Rápido

### Prerrequisitos
- Python 3.11+
- Docker (opcional, para containerización)
- Git

### Instalación

```powershell
# Clonar repositorio
git clone https://dagshub.com/Pamela-ruiz9/MLOps.git
cd MLOps

# Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar DVC (opcional, para datos versionados)
dvc remote add -d dagshub https://dagshub.com/Pamela-ruiz9/MLOps.dvc
dvc pull
```

---

## 🧪 Pruebas Automatizadas

### Ejecutar Todas las Pruebas

```powershell
# Activar entorno
.venv\Scripts\activate

# Ejecutar tests con pytest
pytest -v

# Con reporte de cobertura
pytest --cov=src --cov-report=html

# Solo tests rápidos
pytest -q
```

### Tipos de Pruebas Implementadas

- **Unitarias** (`tests/test_preprocessing.py`, `test_model.py`)
  - Preprocessor: fit, transform, manejo de missing values
  - Modelos: carga, predicciones, probabilidades
  - Métricas: accuracy, F1, ROC-AUC

- **Integración** (`tests/test_integration.py`)
  - Pipeline end-to-end: carga → preprocesamiento → predicción → evaluación
  - Reproducibilidad de resultados
  - Generación y detección de data drift

- **API** (`tests/test_api.py`)
  - Endpoints `/predict` y `/predict-csv`
  - Validación de entrada/salida
  - Manejo de errores

---

## 🔧 Uso del Modelo

### Entrenamiento Local

```powershell
# Entrenar modelo con configuración por defecto
python scripts/main.py --train

# Entrenar Random Forest con hiperparámetros
python scripts/main.py --train --model rf --model-param n_estimators=200 --model-param max_depth=10

# Dry-run (sin guardar)
python scripts/main.py --dry-run --model logreg
```

### Pipeline DVC (Reproducible)

```powershell
# Reproducir pipeline completo
dvc repro

# Ver DAG del pipeline
dvc dag

# Sincronizar artefactos con remoto
dvc push
```

### Predicciones

```python
import joblib
import pandas as pd

# Cargar modelo
model = joblib.load('src/models/artifacts/model.joblib')

# Preparar datos
data = pd.DataFrame({...})

# Predecir
predictions = model.predict(data)
```

---

## 🌐 API REST (FastAPI)

### Iniciar Servidor Localmente

```powershell
# Opción 1: uvicorn directo
uvicorn scripts.API.main_fastapi:app --host 0.0.0.0 --port 8001 --reload

# Opción 2: Con dependencias de API
pip install -r requirements_api.txt
uvicorn scripts.API.main_fastapi:app --port 8001
```

### Documentación Interactiva

Una vez iniciado el servidor, visita:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### Endpoints Disponibles

#### `GET /` - Health Check
```bash
curl http://localhost:8001/
```

Respuesta:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true
}
```

#### `POST /app-credit/predict/` - Predicción Individual

```bash
curl -X POST http://localhost:8001/app-credit/predict/ \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Respuesta:
```json
{
  "prediccion": 0
}
```

#### `POST /app-credit/predict-csv/` - Predicción Batch

```bash
curl -X POST http://localhost:8001/app-credit/predict-csv/ \
  -F "file=@datos.csv"
```

Respuesta:
```json
{
  "predicciones": [0, 1, 0, 1],
  "total": 4
}
```

### Schemas (Validación Pydantic)

La API valida automáticamente:
- **Tipos de datos**: int, str según campo
- **Rangos**: edad (18-100), laufzeit (1-100), rate (1-4)
- **Campos requeridos**: todos los 20 features del modelo

Ejemplo de error de validación:
```json
{
  "detail": [
    {
      "loc": ["body", "alter"],
      "msg": "ensure this value is greater than or equal to 18",
      "type": "value_error"
    }
  ]
}
```

---

## 🐳 Docker

### Construcción de Imagen

```powershell
# Build
docker build -t ml-service:latest .

# Build con tag versionado
docker build -t ml-service:v1.0.0 .
```

### Ejecución del Contenedor

```powershell
# Ejecutar en puerto 8001
docker run -p 8001:8001 ml-service:latest

# Con variables de entorno
docker run -p 8001:8001 \
  -e MLFLOW_TRACKING_URI=https://dagshub.com/Pamela-ruiz9/MLOps.mlflow \
  ml-service:latest
```

### Publicar en DockerHub

```powershell
# Login
docker login

# Tag para DockerHub
docker tag ml-service:latest <usuario>/mlops-credit:v1.0.0

# Push
docker push <usuario>/mlops-credit:v1.0.0
```

### Optimizaciones del Dockerfile

- Base image: `python:3.11-slim` (tamaño reducido)
- Multi-layer caching: copia requirements primero
- `.dockerignore`: excluye venv, notebooks, tests
- Dependencias mínimas: solo `requirements_api.txt`

---

## 📊 Monitoreo y Data Drift

### Generar Datos con Drift

```powershell
python src/monitoring/make_drift.py
```

Esto crea `src/data/drift/german_credit_drift.csv` con drift sintético (distribución numérica alterada ~5%).

### Detectar Drift y Alertas

```powershell
# Análisis estadístico (KS-test)
python src/monitoring/drfit_alerts.py
```

Salida ejemplo:
```
=== DATA DRIFT ALERTS ===

➡ Dataset drift: True
➡ Drift share: 0.45
➡ Columns with drift (9):
laufzeit, hoehe, rate, wohnzeit, alter

=== RECOMMENDATIONS ===
⚠️ Drift severo → retraining recomendado.
```

### Evaluación de Performance

```powershell
python src/monitoring/performance.py
```

Compara métricas (accuracy, F1, ROC-AUC) entre:
- Dataset de validación (baseline)
- Dataset con drift

Genera: `reports/performance_comparison.csv`

### Dashboard con Evidently

```powershell
python src/monitoring/compute_drift_metrics.py
```

Genera dashboard HTML interactivo en `reports/` con visualizaciones de drift por feature.

---

## 🔄 Reproducibilidad

### Semillas Aleatorias

Todas las operaciones con componentes aleatorios usan semillas fijas:

```python
# En scripts/main.py y entrenamiento
import numpy as np
import random

random.seed(42)
np.random.seed(42)
```

### Versionamiento de Artefactos

**Datos** (DVC):
```powershell
# Versionar dataset procesado
dvc add src/data/processed/german_credit_clean.csv
git add src/data/processed/german_credit_clean.csv.dvc
git commit -m "Version processed data"
dvc push
```

**Modelos** (MLflow + DVC):
- MLflow tracking: parámetros, métricas, artifacts automáticos
- DVC: modelo `.joblib` en `src/models/artifacts/`

Acceso a modelos versionados:
- **MLflow**: `https://dagshub.com/Pamela-ruiz9/MLOps.mlflow`
- **Modelo registrado**: `models:/credit-risk-model/Production` (MLflow Model Registry)

### Verificación en Entorno Limpio

```powershell
# En máquina/VM/contenedor nuevo
git clone <repo>
cd MLOps

# Instalar dependencias fijadas
pip install -r requirements.txt

# Descargar datos y modelo
dvc pull

# Reproducir pipeline
dvc repro

# Comparar métricas con referencia
python scripts/main.py --train
# Verificar que accuracy/F1/ROC-AUC coincidan ±0.01
```

---

## 📈 Experimentos MLflow

### Tracking Local

```powershell
# Ver experimentos localmente
mlflow ui

# Abrir en navegador
# http://localhost:5000
```

### Sincronizar con DagsHub

```powershell
# Configurar remote
$env:MLFLOW_TRACKING_URI="https://dagshub.com/Pamela-ruiz9/MLOps.mlflow"
$env:MLFLOW_TRACKING_USERNAME="<usuario>"
$env:MLFLOW_TRACKING_PASSWORD="<token>"

# Importar experimentos locales a remoto
python scripts/import_mlflow_to_remote.py
```

Ver experimentos en: https://dagshub.com/Pamela-ruiz9/MLOps.mlflow

---

## 📚 Documentación Adicional

- [`docs/dvc_pipeline.md`](docs/dvc_pipeline.md) - Guía completa de DVC
- [`docs/dataset_modifications.md`](docs/dataset_modifications.md) - Transformaciones de datos
- [DagsHub Repo](https://dagshub.com/Pamela-ruiz9/MLOps) - Código, datos, modelos

---

## 🛠️ Tecnologías Utilizadas

| Componente | Tecnología |
|-----------|-----------|
| ML Framework | scikit-learn, XGBoost |
| Pipeline | sklearn Pipeline, ColumnTransformer |
| Tracking | MLflow |
| Versionamiento | DVC, Git |
| API | FastAPI, Pydantic |
| Testing | pytest, pytest-cov |
| Containerización | Docker |
| Drift Detection | Evidently, scipy (KS-test) |
| Remote Storage | DagsHub |

---

## 👥 Equipo

**Proyecto MLOps - Equipo 5**  
Tec de Monterrey 2025

---

## 📄 Licencia

Ver archivo `LICENSE` para detalles.

---

## 🔗 Enlaces Útiles

- **Repositorio Git**: https://github.com/secabezon/MLOps
- **DagsHub (DVC + MLflow)**: https://dagshub.com/Pamela-ruiz9/MLOps
- **MLflow UI**: https://dagshub.com/Pamela-ruiz9/MLOps.mlflow
- **Documentación FastAPI**: https://fastapi.tiangolo.com/
- **DVC Docs**: https://dvc.org/doc
