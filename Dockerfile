# ------------------------------------------
# BASE IMAGE
# ------------------------------------------
    FROM python:3.11-slim

    # ------------------------------------------
    # SYSTEM DEPENDENCIES
    # ------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*
    
    # ------------------------------------------
    # WORKDIR
    # ------------------------------------------
    WORKDIR /app
    
    # ------------------------------------------
    # COPY ONLY API REQUIREMENTS
    # ------------------------------------------
    COPY requirements_api.txt ./requirements_api.txt
    
    RUN pip install --no-cache-dir -r requirements_api.txt
    
    # ------------------------------------------
    # COPY ONLY WHAT THE API NEEDS
    # ------------------------------------------
    COPY scripts ./scripts
    COPY src ./src
    
    # *Si tu modelo está dentro de src/models/* ya queda copiado.
    
    # ------------------------------------------
    # Expose FastAPI Port
    # ------------------------------------------
    EXPOSE 8001
    
    # ------------------------------------------
    # Start API
    # ------------------------------------------
    CMD ["uvicorn", "scripts.API.main_fastapi:app", "--host", "0.0.0.0", "--port", "8001"]
    