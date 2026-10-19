# Diabetes Prediction API

API con FastAPI + XGBoost que devuelve la probabilidad de diabetes de un paciente.

## Local
```bash
python -m venv ml-env
source ml-env/Scripts/activate      # Git Bash en Windows
pip install -r requirements.txt
python model/train.py               # opcional: reentrena y regenera el .joblib
uvicorn app:app --reload            # http://127.0.0.1:8000/docs
python client.py                    # en otra terminal
```

## Docker
```bash
docker build -t diabetes-api .
docker run -p 8000:8000 diabetes-api
```

## Endpoints
- `GET /`: estado del servicio
- `POST /score`: predicción (ver `/docs` para el esquema)
