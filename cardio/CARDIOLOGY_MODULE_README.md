# Cardiology AI Module

## Overview
This module extends the RSNA CXR pneumonia detection project by adding a cardiology decision-support pipeline.

## Features
- Cardiovascular risk prediction using UCI Heart Disease dataset
- Personalized patient profile generation
- RAG-based cardiology LLM explanation using Ollama + Mistral
- Explainable AI using SHAP
- FastAPI endpoints for prediction and full analysis
- Validation reports and case studies

## Folder Structure

```text
cardio/
├── api/
├── data/
├── models/
├── personalization/
├── rag/
├── xai/
└── validation/
API Endpoints
/predict

Returns cardiac risk prediction.

/full-analysis

Returns risk prediction, patient profile, XAI explanation, and LLM explanation.

/validation-summary

Returns validation summary.

Disclaimer

This is an academic prototype and not a replacement for cardiologist diagnosis.


## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
2. Start Ollama
ollama serve
ollama pull mistral
3. Run API
cd cardio/api
uvicorn main:app --reload