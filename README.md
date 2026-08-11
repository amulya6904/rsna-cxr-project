# Multimodal Clinical Decision Support Assistant

> **An end-to-end multimodal healthcare AI prototype combining chest X-ray pneumonia detection, cardiovascular risk prediction, Explainable AI, Retrieval-Augmented Generation, and local LLM-based clinical context generation.**

---

## Overview

The **Multimodal Clinical Decision Support Assistant** is a research-oriented AI system designed to demonstrate how multiple forms of healthcare data can be analyzed within a unified decision-support workflow.

The project contains two major AI pipelines:

1. **Radiology Module** — Pneumonia detection from chest X-ray images using deep learning.
2. **Cardiology Module** — Cardiovascular risk prediction from structured patient data.

The system combines:

- Deep Learning
- Machine Learning
- Explainable AI
- Retrieval-Augmented Generation
- Local Large Language Models
- REST APIs
- Clinical safety-oriented output framing

The goal is not to replace clinicians, but to explore how interpretable AI systems can assist with clinical information analysis.

> **Disclaimer:** This project is an academic and research prototype. It is not a medical diagnostic system and must not be used as a substitute for evaluation by qualified healthcare professionals.

---

# Key Results

## Radiology — Pneumonia Detection

| Metric | Result |
|---|---:|
| ROC-AUC | ~0.88 |
| F1 Score | ~0.61 |
| Sensitivity | ~0.54 |
| Specificity | ~0.93 |

## Cardiology — Cardiovascular Risk Prediction

| Metric | Result |
|---|---:|
| Accuracy | **88.33%** |
| Precision | **84.00%** |
| Sensitivity / Recall | **87.50%** |
| F1 Score | **85.71%** |
| ROC-AUC | **94.79%** |

The cardiology model was evaluated on a held-out test set from the UCI Heart Disease dataset.

---

# System Architecture

```text
                     MULTIMODAL CLINICAL AI SYSTEM
                                  |
                 +----------------+----------------+
                 |                                 |
            RADIOLOGY                         CARDIOLOGY
                 |                                 |
          Chest X-Ray                         Patient Data
                 |                                 |
                 v                                 v
          DenseNet-121                     Random Forest
                 |                                 |
          +------+-------+                  +------+------+
          |              |                  |             |
          v              v                  v             v
     Prediction       Grad-CAM          Risk Score       SHAP
          |                                 |
          +---------------+-----------------+
                          |
                          v
                 Clinical Context / RAG
                          |
                          v
                    Local LLM Layer
                          |
                          v
               Explainable Clinical Output
                          |
                          v
                   Human Review Required
```

---

# Core Capabilities

## Radiology Module

The radiology pipeline detects pneumonia from chest X-ray images using a **DenseNet-121 convolutional neural network** trained using the RSNA Pneumonia Detection Challenge dataset.

### Features

- Binary chest X-ray classification
- Normal vs Pneumonia prediction
- DICOM medical image preprocessing
- DICOM-to-PNG conversion
- Automated train/validation split generation
- DenseNet-121 inference pipeline
- Grad-CAM visual explainability
- Local retrieval over radiology notes
- Red-flag clinical context retrieval
- Multimodal report generation
- Model evaluation using standard classification metrics

### Radiology Outputs

```text
outputs/
├── gradcam/
└── validation/
```

---

## Cardiology Module

The cardiology pipeline predicts possible cardiovascular disease risk from structured clinical features.

The pipeline combines a machine-learning classifier with explainability, retrieval, and local LLM-generated context.

### Features

- Random Forest cardiovascular risk prediction
- Risk probability estimation
- Patient profile generation
- SHAP-based feature attribution
- Personalized risk interpretation
- Retrieval-Augmented Generation over cardiology knowledge
- Local Ollama-backed LLM explanations
- FastAPI endpoints
- ROC curve generation
- Validation report generation
- Case-study support

### Input Features

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

### Example Output

The cardiology analysis can generate:

```text
Prediction:
Possible cardiovascular disease risk

Risk Probability:
0.82

Risk Level:
High Risk

Top Contributing Factors:
- Exercise-induced angina
- Maximum heart rate
- ST depression
- Chest pain type

Explanation:
Generated using model prediction,
SHAP attribution, retrieved clinical
context, and a local LLM.

Recommendation:
Clinical review required.
```

---

# Explainable AI

Healthcare AI systems should not produce predictions without providing insight into how those predictions were generated.

This project therefore includes two explainability mechanisms.

## Grad-CAM

Grad-CAM is used for the radiology pipeline to generate heatmaps highlighting image regions that contributed to the pneumonia prediction.

```text
Chest X-Ray
     |
     v
DenseNet-121
     |
     +------> Pneumonia Probability
     |
     +------> Grad-CAM Heatmap
```

This allows qualitative inspection of whether the image model is focusing on clinically relevant regions.

---

## SHAP

The cardiology model uses **SHAP feature attribution** to explain the contribution of individual patient features.

Examples include:

- Age
- Cholesterol
- Resting blood pressure
- Maximum heart rate
- Chest pain type
- Exercise-induced angina
- ST depression

Instead of returning only a prediction, the pipeline provides insight into which factors contributed most strongly to the model output.

---

# Retrieval-Augmented Generation

The project includes local Retrieval-Augmented Generation pipelines for both clinical domains.

Instead of asking a language model to generate unrestricted clinical information, relevant context is first retrieved from a controlled local knowledge base.

```text
User / Patient Context
        |
        v
   Retrieval System
        |
        v
Relevant Clinical Notes
        |
        v
   Local LLM
        |
        v
Context-Aware Explanation
```

This design improves grounding while reducing reliance on unconstrained LLM generation.

---

# Local LLM Integration

The generative AI layer runs locally using **Ollama**.

This provides several advantages:

- No paid API dependency
- Local inference
- Better privacy for development experiments
- Configurable language model
- Offline-friendly architecture

The default model can be changed using:

```powershell
$env:CARDIO_OLLAMA_MODEL="mistral"
```

Example Ollama setup:

```powershell
ollama serve
ollama pull llama3.2
```

---

# Technology Stack

## AI / Machine Learning

- Python
- PyTorch
- TorchVision
- DenseNet-121
- Scikit-learn
- Random Forest
- NumPy
- Pandas

## Explainable AI

- Grad-CAM
- SHAP

## Generative AI / RAG

- Ollama
- LangChain
- Sentence Transformers
- ChromaDB
- Retrieval-Augmented Generation

## Medical Data Processing

- DICOM
- Pydicom
- OpenCV
- Pillow

## Backend

- FastAPI
- Uvicorn
- Python

## Frontend

- HTML
- CSS
- JavaScript

## Visualization

- Matplotlib
- ROC curves
- Confusion matrices
- Grad-CAM heatmaps

---

# Datasets

## RSNA Pneumonia Detection Challenge

Used for the radiology module.

The dataset contains chest X-ray images and pneumonia annotations.

The project performs:

- DICOM preprocessing
- Binary label generation
- Train/validation splitting
- Model evaluation

Due to dataset size, the raw RSNA dataset is not stored in this repository.

---

## UCI Heart Disease Dataset

Used for cardiovascular risk prediction.

The cardiology pipeline uses the Cleveland processed Heart Disease dataset and structured clinical attributes such as:

- Age
- Sex
- Blood pressure
- Cholesterol
- Chest pain type
- ECG-related features
- Exercise response

---

# Repository Structure

```text
rsna-cxr-project/
|
├── backend/
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
│
├── scripts/
│   ├── convert_dicom_to_png.py
│   ├── create_binary_labels.py
│   ├── make_splits.py
│   ├── gradcam_generate.py
│   ├── make_clinical_text.py
│   ├── rag_utils.py
│   ├── rag_retrieve.py
│   ├── multimodal_infer.py
│   └── evaluate_model.py
│
├── kb/
│   ├── pneumonia_notes.txt
│   └── red_flags.txt
│
├── cardio/
│   |
│   ├── api/
│   │   └── main.py
│   |
│   ├── data/
│   │   └── heart.csv
│   |
│   ├── models/
│   │   ├── train_model.py
│   │   ├── predict.py
│   │   └── cardio_model.pkl
│   |
│   ├── personalization/
│   │   └── patient_profile.py
│   |
│   ├── rag/
│   │   ├── docs/
│   │   │   └── cardio_guidelines.txt
│   │   ├── build_vectorstore.py
│   │   ├── query_rag.py
│   │   ├── cardio_llm.py
│   │   └── cardio_pipeline.py
│   |
│   ├── xai/
│   │   ├── shap_explainer.py
│   │   └── xai_interpreter.py
│   |
│   └── validation/
│       ├── evaluate_cardio_model.py
│       ├── plot_roc_curve.py
│       ├── case_studies.py
│       ├── cardio_validation_report.txt
│       ├── cardio_case_studies.txt
│       └── cardio_roc_curve.png
│
├── data/
├── models/
├── outputs/
├── vectorstore/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

Large datasets, trained radiology weights, and generated artifacts may be excluded from Git depending on `.gitignore`.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/amulya6904/rsna-cxr-project.git
cd rsna-cxr-project
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Core Dependencies

```bash
pip install numpy pandas scikit-learn pillow matplotlib tqdm
pip install torch torchvision
pip install opencv-python
pip install pydicom pylibjpeg pylibjpeg-libjpeg
```

Install project RAG dependencies:

```bash
pip install -r requirements.txt
```

Install cardiology-specific dependencies if required:

```bash
pip install -r cardio/requirements.txt
```

---

# Radiology Dataset Preparation

Download the **RSNA Pneumonia Detection Challenge dataset** from Kaggle.

Place the required data locally using the following structure:

```text
data/
├── dicom/
└── stage_2_train_labels.csv
```

Convert DICOM files to PNG:

```bash
python scripts/convert_dicom_to_png.py
```

Generate binary labels:

```bash
python scripts/create_binary_labels.py
```

Generate train/validation splits:

```bash
python scripts/make_splits.py
```

---

# Running the Radiology Pipeline

## Generate Clinical Text

```bash
python scripts/make_clinical_text.py
```

## Run Multimodal Inference

```bash
python scripts/multimodal_infer.py
```

## Generate Grad-CAM Heatmaps

```bash
python scripts/gradcam_generate.py
```

## Evaluate the Pneumonia Model

```bash
python scripts/evaluate_model.py
```

## Query the Radiology Knowledge Base

```bash
python scripts/rag_retrieve.py
```

---

# Running the Cardiology Pipeline

## Train the Model

```bash
python cardio/models/train_model.py
```

---

## Run a Direct Prediction

```bash
python cardio/models/predict.py
```

---

## Run the Complete Analysis Pipeline

```bash
python cardio/rag/cardio_pipeline.py
```

The full pipeline combines:

```text
Patient Features
      |
      v
Random Forest Prediction
      |
      +------> Risk Probability
      |
      +------> SHAP Explanation
      |
      v
Patient Profile
      |
      v
Clinical RAG
      |
      v
Local Ollama LLM
      |
      v
Personalized Decision-Support Output
```

---

# API

The cardiology module exposes REST endpoints through FastAPI.

Start the API from the project root:

```bash
uvicorn main:app --reload
```

Alternatively:

```bash
cd cardio/api
uvicorn main:app --reload
```

## Available Endpoints

```text
GET  /
POST /predict
POST /full-analysis
GET  /validation-summary
```

---

# Example API Request

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

# Validation

## Radiology Validation

The chest X-ray classifier is evaluated using:

- ROC-AUC
- F1 Score
- Sensitivity
- Specificity
- Confusion Matrix
- Grad-CAM qualitative inspection

Current experimental results:

```text
ROC-AUC:     ~0.88
F1 Score:    ~0.61
Sensitivity: ~0.54
Specificity: ~0.93
```

---

## Cardiology Validation

The Random Forest cardiovascular model achieved:

```text
Accuracy:              88.33%
Precision:             84.00%
Recall / Sensitivity:  87.50%
F1 Score:              85.71%
ROC-AUC:               94.79%
```

Confusion matrix:

```text
[[32  4]
 [ 3 21]]
```

Validation artifacts are available in:

```text
cardio/validation/
```

---

# Why Multimodal Clinical AI?

Healthcare decisions are rarely based on a single data source.

A clinician may need to interpret:

- Medical imaging
- Patient history
- Structured clinical measurements
- Symptoms
- Guidelines
- Risk factors

This project explores how different AI approaches can work together instead of operating as isolated models.

The architecture therefore combines:

```text
Computer Vision
      +
Tabular Machine Learning
      +
Explainable AI
      +
Retrieval
      +
Large Language Models
      +
Human Oversight
```

The result is a more complete decision-support prototype than a standalone classifier.

---

# Design Principles

The system was developed around several key principles.

### Explainability First

Predictions should be accompanied by interpretable evidence whenever possible.

### Local-First AI

The generative AI pipeline can operate locally through Ollama without requiring paid external APIs.

### Modular Architecture

Radiology, cardiology, retrieval, explainability, frontend, and API functionality are separated into modular components.

### Reproducibility

Dataset preprocessing, model evaluation, validation, and visualization are implemented as repeatable scripts.

### Human-in-the-Loop

The system is intended to assist clinical interpretation rather than autonomously generate medical diagnoses.

### Safety-Aware Output

Clinical predictions are accompanied by disclaimers and human-review requirements.

---

# Limitations

This project remains an academic prototype and has several important limitations.

- It is not clinically validated for real-world deployment.
- The pneumonia classifier is trained using a single public radiology dataset.
- The cardiology model uses a relatively small public benchmark dataset.
- No prospective clinical validation has been performed.
- Radiology sensitivity remains an area for improvement.
- Local LLM explanation quality depends on the selected Ollama model.
- Retrieved clinical knowledge is limited to the local knowledge base.
- Patient summaries are simplified representations of real clinical workflows.
- Model probabilities should not be interpreted as medical diagnoses.
- All outputs require review by qualified medical professionals.

---

# Future Work

Future improvements could include:

- Unified radiology and cardiology dashboard
- Improved pneumonia sensitivity
- Probability calibration
- External validation using additional medical datasets
- MIMIC-CXR evaluation
- Larger cardiology cohorts
- Improved RAG source attribution
- Citation display for retrieved medical context
- Authentication and user management
- Clinical audit logging
- Model monitoring
- Model cards
- Expanded explainability reports
- Clinician feedback interface
- Human-in-the-loop validation workflows
- Containerized deployment
- Cloud deployment
- FHIR-compatible healthcare data integration

---

# Potential Real-World Workflow

```text
Patient Data / Imaging
        |
        v
AI Analysis
        |
        v
Risk / Prediction
        |
        v
Explainability
        |
        v
Clinical Knowledge Retrieval
        |
        v
Contextual Explanation
        |
        v
Qualified Clinician Review
        |
        v
Final Clinical Decision
```

The AI component is intentionally positioned as a **decision-support layer**, not the final decision-maker.

---

# Project Highlights

- End-to-end multimodal healthcare AI architecture
- Deep-learning chest X-ray analysis
- DenseNet-121 pneumonia classification
- Grad-CAM visual explainability
- Cardiovascular risk prediction
- Random Forest classification
- SHAP feature attribution
- Retrieval-Augmented Generation
- Local LLM integration using Ollama
- FastAPI backend
- REST API endpoints
- Frontend implementation
- Quantitative model validation
- Safety-aware medical AI framing
- Modular and reproducible implementation

---

# Research and Educational Use

This repository is intended for:

- Machine Learning experimentation
- Medical AI research
- Explainable AI exploration
- Retrieval-Augmented Generation experimentation
- Multimodal system design
- Academic demonstrations
- Portfolio and educational purposes

It should **not** be used for real-world medical diagnosis or treatment decisions.

---

# Author

**Amulya Anutej**

Computer Science Engineering  
Ramaiah Institute of Technology

GitHub: `amulya6904`

---

# Acknowledgements

This project uses publicly available datasets and open-source tools, including:

- RSNA Pneumonia Detection Challenge
- UCI Heart Disease Dataset
- PyTorch
- Scikit-learn
- FastAPI
- SHAP
- LangChain
- ChromaDB
- Ollama
- Sentence Transformers

---

## Important Medical Disclaimer

This software is provided solely for **academic, research, and educational purposes**.

It is not approved as a medical device and must not be used for:

- Diagnosis
- Treatment planning
- Emergency decision-making
- Medication decisions
- Replacement of professional medical evaluation

All model outputs require review and interpretation by qualified healthcare professionals.