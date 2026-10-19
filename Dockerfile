FROM python:3.12-slim
WORKDIR /app

# XGBoost necesita libgomp (OpenMP), que no viene en la imagen slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./app.py .
COPY ./model/diabetes-v1.joblib ./model/diabetes-v1.joblib

# Render define la variable PORT; en local usa 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
