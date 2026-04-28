from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add models path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from predict import predict_cardio

app = FastAPI()

class PatientData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def home():
    return {"message": "Cardiology API Running"}

@app.post("/predict")
def predict(data: PatientData):
    features = [
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal
    ]

    result = predict_cardio(features)
    return result
