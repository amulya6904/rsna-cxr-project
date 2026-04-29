# Multimodal Clinical Decision Support Assistant

**Chest X-Ray Pneumonia Detection + Cardiology Risk Analysis + Explainable AI + Retrieval-Augmented Clinical Context**

---

## Overview

This project is an end-to-end **AI-powered clinical decision-support prototype** with two connected healthcare AI workflows:

1. **Radiology module** for pneumonia detection from chest X-ray images using the RSNA Pneumonia Detection Challenge dataset.
2. **Cardiology module** for cardiovascular risk prediction, personalized patient summaries, explainable feature attribution, and retrieval-augmented cardiology explanations.

The system emphasizes **interpretability, modularity, reproducibility, and safety-aware clinical reasoning**. It is intended for research and educational use only.

> Warning: This project is an academic prototype and is not a clinical diagnostic tool. Outputs must not replace review by qualified clinicians.

---

## Core Capabilities

### Radiology: Pneumonia Detection from Chest X-Ray

The radiology pipeline detects pneumonia from chest X-ray images using a DenseNet-121 image classifier trained on the **RSNA Pneumonia Detection Challenge dataset**.

Key features:

* Binary classification: Normal vs Pneumonia
* DICOM-to-PNG preprocessing
* Train/validation split generation
* Grad-CAM heatmaps for visual explainability
* Local RAG over radiology notes and red-flag guidance
* Multimodal report generation using model probability, clinical text, and retrieved context
* Quantitative validation using ROC-AUC, F1, sensitivity, specificity, and confusion matrix

Example radiology validation results:

* AUC approximately 0.88
* F1 approximately 0.61
* Sensitivity approximately 0.54
* Specificity approximately 0.93

Generated radiology outputs:

```text
outputs/gradcam/
outputs/validation/
```

---

### Cardiology: Cardiovascular Risk Analysis

The cardiology module extends the project beyond image-based pneumonia detection by adding a structured cardiovascular decision-support pipeline.

Key features:

* Cardiovascular risk prediction using the UCI Heart Disease dataset
* Random Forest classifier saved as `cardio/models/cardio_model.pkl`
* Patient profile generation from structured clinical features
* SHAP-based explainability for top contributing risk factors
* RAG over cardiology guideline notes
* Ollama-backed local LLM explanations
* FastAPI endpoints for prediction, full analysis, and validation summary
* Validation reports, ROC curve artifact, and case-study support

Cardiology input features:

```text
age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal
```

Cardiology output includes:

* Risk prediction
* Risk probability
* Risk level: Low Risk, Medium Risk, or High Risk
* Patient profile summary
* Top SHAP feature impacts
* Explainability summary
* Personalized clinical explanation with cardiologist-review disclaimer

---

## Architecture

### Radiology Model

* Backbone: **DenseNet-121**
* Input: 224 x 224 RGB chest X-ray
* Output: Binary classification
* Loss: Cross-Entropy
* Training platform: Kaggle GPU
* Inference: CPU compatible
* Explainability: Grad-CAM

### Cardiology Model

* Model: **Random Forest Classifier**
* Dataset: UCI Heart Disease / Cleveland processed dataset
* Input: Structured tabular cardiac risk features
* Output: Binary risk prediction and risk probability
* Explainability: SHAP feature attribution
* Personalization: Template-based patient profile plus retrieved guideline context
* LLM layer: Local Ollama model, configurable with `CARDIO_OLLAMA_MODEL`

---

## Repository Structure

```text
rsna-cxr-project/
|
|-- backend/
|   |-- app.py
|
|-- frontend/
|   |-- index.html
|   |-- script.js
|   |-- styles.css
|
|-- scripts/
|   |-- convert_dicom_to_png.py
|   |-- create_binary_labels.py
|   |-- make_splits.py
|   |-- gradcam_generate.py
|   |-- make_clinical_text.py
|   |-- rag_utils.py
|   |-- rag_retrieve.py
|   |-- multimodal_infer.py
|   |-- evaluate_model.py
|
|-- kb/
|   |-- pneumonia_notes.txt
|   |-- red_flags.txt
|
|-- cardio/
|   |-- api/
|   |   |-- main.py
|   |
|   |-- data/
|   |   |-- heart.csv
|   |
|   |-- models/
|   |   |-- train_model.py
|   |   |-- predict.py
|   |   |-- cardio_model.pkl
|   |
|   |-- personalization/
|   |   |-- patient_profile.py
|   |
|   |-- rag/
|   |   |-- docs/
|   |   |   |-- cardio_guidelines.txt
|   |   |-- build_vectorstore.py
|   |   |-- query_rag.py
|   |   |-- cardio_llm.py
|   |   |-- cardio_pipeline.py
|   |
|   |-- xai/
|   |   |-- shap_explainer.py
|   |   |-- xai_interpreter.py
|   |
|   |-- validation/
|       |-- evaluate_cardio_model.py
|       |-- plot_roc_curve.py
|       |-- case_studies.py
|       |-- cardio_validation_report.txt
|       |-- cardio_case_studies.txt
|       |-- cardio_roc_curve.png
|
|-- data/              # local datasets, ignored in git
|-- models/            # local trained radiology weights, ignored in git
|-- outputs/           # generated radiology outputs
|-- vectorstore/        # generated retrieval artifacts
|
|-- main.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

Large datasets, generated artifacts, and trained weights may be excluded from version control depending on `.gitignore`.

---

## Installation

### 1. Create Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install Base Dependencies

```powershell
pip install numpy pandas scikit-learn pillow matplotlib tqdm
pip install torch torchvision
pip install opencv-python
pip install pydicom pylibjpeg pylibjpeg-libjpeg
```

### 3. Install Cardiology Dependencies

```powershell
pip install -r cardio/requirements.txt
```

The cardiology RAG and LLM explanation path also expects Ollama to be installed locally.

```powershell
ollama serve
ollama pull llama3.2
```

You can use another local Ollama model by setting:

```powershell
$env:CARDIO_OLLAMA_MODEL="mistral"
```

---

## Radiology Dataset Preparation

Download the **RSNA Pneumonia Detection Challenge dataset** from Kaggle.

Place the files locally:

```text
data/dicom/
data/stage_2_train_labels.csv
```

Convert DICOM to PNG:

```powershell
python scripts/convert_dicom_to_png.py
```

Create labels and splits:

```powershell
python scripts/create_binary_labels.py
python scripts/make_splits.py
```

---

## Running the Radiology Pipeline

### Generate Multimodal Clinical Report

```powershell
python scripts/make_clinical_text.py
python scripts/multimodal_infer.py
```

### Generate Grad-CAM Heatmaps

```powershell
python scripts/gradcam_generate.py
```

### Evaluate Pneumonia Model

```powershell
python scripts/evaluate_model.py
```

### Query Radiology Knowledge Base

```powershell
python scripts/rag_retrieve.py
```

---

## Running the Cardiology Module

### Train or Refresh the Cardiology Model

```powershell
cd cardio/models
python train_model.py
```

### Run a Direct Prediction Test

```powershell
python cardio/models/predict.py
```

### Run Full Cardiology Analysis

```powershell
python cardio/rag/cardio_pipeline.py
```

The full analysis combines:

* Random Forest risk prediction
* Patient profile generation
* SHAP explainability
* Retrieved cardiology guideline context
* Local LLM explanation through Ollama

### Start the Cardiology API

From the project root:

```powershell
uvicorn main:app --reload
```

Or from the cardiology API directory:

```powershell
cd cardio/api
uvicorn main:app --reload
```

Available endpoints:

```text
GET  /
POST /predict
POST /full-analysis
GET  /validation-summary
```

Example `/predict` request body:

```json
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
```

---

## Validation

### Radiology Validation

The pneumonia detection pipeline supports standard classification metrics and Grad-CAM-based qualitative review.

Metrics include:

* ROC-AUC
* F1 Score
* Sensitivity
* Specificity
* Confusion Matrix

### Cardiology Validation

The cardiology module includes:

* Model evaluation script
* ROC curve generation
* Validation summary endpoint
* Case-study generation support

Relevant files:

```text
cardio/validation/evaluate_cardio_model.py
cardio/validation/plot_roc_curve.py
cardio/validation/case_studies.py
cardio/validation/cardio_validation_report.txt
cardio/validation/cardio_case_studies.txt
cardio/validation/cardio_roc_curve.png
```

---

## Key Design Principles

* Local-first implementation
* No paid API dependency required
* Modular radiology and cardiology workflows
* Explainability-first model outputs
* Retrieval-augmented clinical context
* Human-in-the-loop safety framing
* Reproducible preprocessing and validation scripts

---

## Limitations

* Research prototype only
* Not externally validated for clinical deployment
* Pneumonia model uses a single public radiology dataset
* Cardiology model uses a small public benchmark dataset
* LLM explanations depend on local Ollama availability and model quality
* Clinical text and patient profiles are simplified and template-driven
* No final diagnosis should be made from this system alone

---

## Future Work

* Integrate the radiology and cardiology outputs into a unified frontend workflow
* Add calibrated probability thresholds for both modules
* Validate against external datasets such as MIMIC-CXR and larger cardiology cohorts
* Add authentication and audit logging for deployment-style demos
* Improve RAG source tracking and citation display
* Add model cards for the radiology and cardiology components
* Expand clinician review forms and human-in-the-loop validation
