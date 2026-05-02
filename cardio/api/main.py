from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add models path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag")))

from predict import predict_cardio
from cardio_pipeline import full_cardio_analysis

try:
    from cardio_llm import generate_cardio_answer
except ImportError:
    generate_cardio_answer = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Cardiology API Running"}

@app.post("/chat")
def chat(request: ChatRequest):
    if generate_cardio_answer is None:
        return {"answer": "Llama chatbot integration is not available."}
    try:
        answer = generate_cardio_answer(request.query)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"Error communicating with Llama: {str(e)}"}

@app.post("/predict")
def predict(data: PatientData):
    features = [
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal
    ]

    result = predict_cardio(features)
    return result

def to_features(data: PatientData):
    return [
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal
    ]

@app.post("/full-analysis")
def full_analysis(data: PatientData):
    result = full_cardio_analysis(to_features(data))

    return {
        "profile": result["patient_profile"],
        "risk": result["risk_prediction"],
        "top_features": result["top_features"],
        "xai_summary": result["xai_summary"],
        "explanation": result["personalized_explanation"]
    }

@app.get("/validation-summary")
def validation_summary():
    return {
        "dataset": "UCI Heart Disease - Cleveland processed dataset",
        "model": "Random Forest Classifier",
        "metrics_used": [
            "Accuracy",
            "Precision",
            "Recall/Sensitivity",
            "F1 Score",
            "ROC-AUC",
            "Confusion Matrix",
        ],
        "validation_status": "Validated on public benchmark dataset",
        "clinical_note": "Academic prototype only. Not a replacement for cardiologist diagnosis.",
    }
