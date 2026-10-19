from typing import Literal
import pathlib

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel, Field

THRESHOLD = 0.5   # probabilidad mínima para clasificar como "con diabetes"

app = FastAPI(title='Diabetes Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

model = load(pathlib.Path('model/diabetes-v1.joblib'))


class InputData(BaseModel):
    gender: Literal['Female', 'Male'] = 'Female'
    age: float = Field(54.0, ge=0, le=120)
    hypertension: Literal[0, 1] = 0
    heart_disease: Literal[0, 1] = 0
    smoking_history: Literal['never', 'former', 'current', 'not current',
                             'ever', 'No Info'] = 'never'
    bmi: float = Field(27.3, ge=10, le=100)
    HbA1c_level: float = Field(6.6, ge=3, le=15)
    blood_glucose_level: int = Field(140, ge=50, le=500)


class OutputData(BaseModel):
    score: float = 0.15
    prediction: int = 0
    label: str = 'No diabetes'


@app.get('/')
def health():
    return {'status': 'ok', 'service': 'Diabetes Prediction'}


@app.post('/score', response_model=OutputData)
def score(data: InputData):
    model_input = pd.DataFrame([data.model_dump()])
    proba = float(model.predict_proba(model_input)[0, 1])
    pred = int(proba >= THRESHOLD)
    return {'score': proba,
            'prediction': pred,
            'label': 'Diabetes' if pred else 'No diabetes'}
