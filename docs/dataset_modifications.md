# Registro de modificaciones al dataset

Este documento resume todas las modificaciones realizadas al dataset *German Credit* durante el EDA y la preparación para modelado. Incluye la configuración de versionado con DVC (usando Dagshub como remoto), las transformaciones aplicadas, motivos técnicos y enlaces a los artefactos generados.

## 0) Configuración de DVC y Dagshub (resumen)

Notas principales sobre cómo se configuró DVC para este proyecto (no incluye tokens/secretos):

- Preparar entorno y dependencias:
  - Activar el entorno virtual (ej.: `.venv\Scripts\activate` en Windows PowerShell).
  - Instalar DVC con soporte para Google Drive si se va a usar: `pip install "dvc[gdrive]"`.
  - Verificar la instalación: `dvc --version`.

- Conectar con Dagshub como remoto DVC:
  - Crear/usar un remote DVC apuntando al repositorio Dagshub del proyecto. Ejemplo:
    - `dvc remote add -d dagshub https://dagshub.com/Pamela-ruiz9/MLOps.dvc`
  - Autenticación/credenciales: usa el método seguro que prefieras (variables de entorno, login interactivo de DVC, o las integraciones de Dagshub). **No** guardar tokens en archivos de texto del repositorio.
  - Subir datos y metadatos con DVC:
    - `dvc add <ruta_al_dataset>` para trackear archivos grandes.
    - `git add <.dvc files> && git commit -m "track data with dvc"`.
    - `dvc push` para enviar los datos al remote.
  - Finalmente, empujar cambios Git a Dagshub/remote git: `git push dagshub main` (o con `--force-with-lease` si es necesario, pero con cuidado).

- En este proyecto se registraron instrucciones y un ejemplo de comandos en un archivo local (suministrado por el equipo). Si vas a reproducir, reemplaza `<usuario>`/`<repo>` por los tuyos y asegúrate de usar un método de autenticación seguro.

---

## 1) Resumen de artefactos y notebooks añadidos/modificados

Archivos de notebooks y artefactos que se crearon o editaron durante el EDA y la preparación de modelado:

- Notebooks:
  - `notebooks/EDA_german_credit.ipynb` —Notebook principal creado para el EDA completo: normalización, detección de missing tokens, KNN imputation, outlier capping, casteos y gráficos (se añadieron secciones de correlación y target analysis).
  - `notebooks/modeling_and_training.ipynb` — Nuevo notebook creado para preprocesado, selección de features y entrenamiento (incluye Logistic Regression, RandomForest y XGBoost).

- Artefactos de datos y reglas:
  - `data/processed/cleaning_rules.json` — Registro automatizado por columna con las reglas aplicadas y muestras de valores problemáticos.
  - `reports/figures/*` — Figuras generadas: histogramas, boxplots, heatmaps, etc.
  - `models/pipeline_logreg.joblib` — Pipeline de regresión logística guardado (si se ejecutó el notebook de modelado).
  - `models/pipeline_RandomFores.joblib` — Pipeline de RandomForest guardado (si se ejecutó el notebook de modelado).
  - `models/pipeline_XGBoost.joblib` — Pipeline de XGBoost guardado (si se ejecutó el notebook de modelado).

---

## 2) Lista cronológica y justificación de cambios aplicados al dataset

A continuación se documentan, por pasos lógicos, las transformaciones realizadas, con la razón técnica para cada una y los artefactos que las registran.

1) Normalización de nombres de columnas
- Qué se hizo:
  - Eliminación de espacios, conversión a minúsculas y reemplazo de espacios por guiones bajos (ej.: `df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()`).
- Por qué:
  - Estándarizar nombres evita errores posteriores (accesos por clave), mejora la reproducibilidad y facilita la escritura de scripts y pipelines.
- Artefactos:
  - Cambio en los notebooks; reflejado indirectamente en los CSV resultantes.

2) Eliminación de columnas extras y columnas mixed-type
- Qué se hizo:
  - Se detectaron y eliminaron columnas que no existían en `df_orig` (posibles columnas añadidas accidentalmente) y columnas de tipo `object` con mezcla de valores numéricos y textuales que causarían problemas en pasos numéricos.
- Por qué:
  - Columnas extra pueden venir de salidas parciales o metadatos inconsistentes. Las columnas mixed-type pueden romper conversiones numéricas o modelos (strings en columnas numéricas). Es preferible auditar y eliminar o aislar estas columnas antes del preprocesado.
- Artefactos:
  - Registro en `data/processed/cleaning_rules.json` con anotaciones `dropped_mixed_type`.

3) Normalización y reemplazo de tokens de missing
- Qué se hizo:
  - Se identificaron tokens comunes que representaban missing (`n/a`, `na`, `null`, `?`, `unknown`, `error`, `invalid`, `none`, `''`) y se reemplazaron por `np.nan`.
  - Limpieza de strings (strip, normalización de espacios, eliminación de caracteres invisibles).
- Por qué:
  - En datasets reales, hay múltiples representaciones de missing; estandarizarlas permite conteos correctos, imputaciones consistentes y evita convertir estas cadenas en categorías útiles erroneamente.
- Artefactos:
  - `data/processed/cleaning_rules.json` registra que se aplicó `str_strip_and_normalize` y `replace_common_missing_tokens` por columna.

4) Coerción de columnas numéricas (coerce to numeric)
- Qué se hizo:
  - Para columnas que `df_orig` marcaba como numéricas, se intentó forzar a numérico con `pd.to_numeric(errors='coerce')`. Valores no convertibles pasaron a `NaN`.
- Por qué:
  - Garantizar tipo numérico es necesario para análisis estadístico, imputación y modelado. Coerción con `errors='coerce'` es segura porque registra en `NaN` las celdas no-convertibles para revisión posterior.
- Artefactos:
  - `rows_with_problematic_numeric_values_sample.csv` (si se detectaron) y `cleaning_rules.json` con conteos antes/después.

5) Identificación de filas extra (multiplicity-aware)
- Qué se hizo:
  - Se compararon `df` y `df_orig` de manera awareness de multiplicidad (conteo por fila/valores) para detectar las 20 filas extra que no pertenecen al `df_orig`.
  - Se exportaron estas filas a `data/raw/extra_rows_identified.csv` para auditoría.
- Por qué:
  - Mantener trazabilidad: no eliminar silenciosamente datos sin registrar. Identificar filas extra permite discutir si deben borrarse o corregirse.
- Artefactos:
  - `data/raw/extra_rows_identified.csv` (si se ejecutó la celda). Registro en `cleaning_rules.json` con anotación sobre filas extra.

6) Reemplazo de imputación previa y uso de KNNImputer (decisión del usuario)
- Qué se hizo:
  - Se implementó primero una opción de imputación median/mode, pero el usuario pidió explícitamente usar sólo KNNImputer. Se dejó la celda final que aplica `KNNImputer(n_neighbors=5)` sólo a columnas numéricas.
  - Si alguna columna numérica era completamente NaN, se rellenó temporalmente con mediana de `df_orig` (o 0) para que KNN pueda ejecutarse, y luego se registró la operación.
- Por qué:
  - KNNImputer aprovecha la estructura multivariada para imputar valores plausibles. El usuario solicitó KNN por razones de robustez en este caso.
- Artefactos:
  - `data/raw/german_credit_clean_v1_knn_imputed.csv` (nombre de ejemplo generado por notebook) y `cleaning_rules.json` con `imputation` info.

7) Detección y capping de outliers (IQR)
- Qué se hizo:
  - Se detectaron outliers usando el criterio IQR (k=1.5) por columna numérica.
  - Se aplicó capping: valores por debajo del límite inferior se reemplazaron por el mínimo no outlier; valores por encima del límite superior se reemplazaron por el máximo no outlier.
  - Se creó una copia `df_capped` y se guardó como CSV (`*_capped.csv`).
- Por qué:
  - El capping reduce el efecto de outliers extremos en modelos sensibles y en estadísticas. Se eligió capping por límite no-outlier para mantener valores realistas en el rango observado.
- Artefactos:
  - `data/raw/german_credit_clean_v1_capped.csv` (ejemplo), `reports/figures/*` con histogramas/boxplots y `cleaning_rules.json` con límite y conteos de outliers por columna.

8) Conversión a `Int64` (dtype nullable)
- Qué se hizo:
  - Se ejecutó un cast directo a `Int64` (dtype nullable de pandas) para columnas donde había valores numéricos detectables. Fracciones se redondearon a entero antes de convertir.
- Por qué:
  - El user pidió explícitamente que las columnas estén en un formato entero. `Int64` (nullable) permite representar NA junto a enteros sin forzar a float.
- Riesgo / Nota:
  - No todas las columnas categóricas deben convertirse a enteros sin una codificación explícita. Se aplicó cast solo cuando había valores numéricos detectables.
- Artefactos:
  - Cambios en `df` en memoria y guardados posteriores; registro en `cleaning_rules.json`.

9) Guardado de reglas y artefactos de auditoría
- Qué se hizo:
  - Se guardó un archivo `data/processed/cleaning_rules.json` con un historial de las acciones por columna (coerciones, imputaciones, capping, casts, drops), además de muestras problemáticas.
  - Se generaron CSVs y figuras a `reports/` y `data/raw/` para revisión manual.
- Por qué:
  - Trazabilidad y reproducibilidad: es crítico poder decir qué se hizo y por qué, y recuperar las filas problemáticas si es necesario.

10) Preparación para modelado
- Qué se hizo:
  - Se creó el notebook `notebooks/modeling_and_training.ipynb` que:
    - Fija `target_col = 'kredit'`.
    - Convierte las columnas categóricas (según la documentación suministrada) a `category`.
    - Construye un `ColumnTransformer` que aplica imputación + escalado a numéricas y OneHot/Ordinal a categóricas según cardinalidad.
    - Aplica `SelectKBest` si hay target para seleccionar un subconjunto de features.
    - Entrena dos modelos de referencia: LogisticRegression y RandomForest, y salva ambos pipelines (`pipeline_logreg.joblib`, `pipeline_model.joblib`).
- Por qué:
  - Preparar un pipeline reproducible que incluya transformación y modelo en un solo objeto facilita despliegue y evaluación.
- Artefactos:
  - `models/pipeline_logreg.joblib`, `models/pipeline_model.joblib`, reportes de métricas en la salida del notebook.

---

## 3) Buenas prácticas y decisiones conservadoras tomadas

- No sobrescribir `df_orig`: toda limpieza se hace sobre una copia `df` y los artefactos y reglas se almacenan separadamente para auditoría.
- Mantener backups de filas problemáticas y filas extra en CSV antes de cualquier borrado o reemplazo.
- Usar `Int64` (nullable) cuando se requieren enteros con NA en lugar de float.
- Registrar cada operación en `cleaning_rules.json` para reproducibilidad.
- Evitar compartir tokens/secretos en repositorios (usar variables de entorno o secrets manager).

---

## 4) Archivos a revisar y comandos útiles

- Revisa estos paths en el repo para confirmar artefactos:
  - `notebooks/EDA_full.ipynb`, `notebooks/EDA_german_credit.ipynb`, `notebooks/modeling_and_training.ipynb`
  - `data/processed/cleaning_rules.json`
  - `data/raw/*_knn_imputed.csv`, `data/raw/*_capped.csv`, `data/raw/extra_rows_identified.csv`
  - `reports/figures/` (gráficos generados)
  - `models/pipeline_logreg.joblib`, `models/pipeline_model.joblib`

Comandos DVC/Git básicos (ejemplos):

```powershell
# activar entorno (Windows PowerShell)
.venv\Scripts\activate

# instalar dvc si es necesario
pip install "dvc[gdrive]"

dvc --version

# agregar dataset a dvc y push (cuando corresponda)
dvc add data/raw/german_credit_modified.csv
git add data/raw/german_credit_modified.csv.dvc .gitignore
git commit -m "track raw german_credit with dvc"
dvc push

# configurar remote dagshub (ejemplo, reemplazar usuario/repo)
dvc remote add -d dagshub https://dagshub.com/<usuario>/<repo>.dvc
# autenticar con el método preferido (no en texto plano)

git push dagshub main
```

---