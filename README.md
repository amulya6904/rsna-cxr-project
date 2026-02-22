# Multimodal Chest X-Ray Pneumonia Detection Assistant

**DenseNet121 + Grad-CAM + Knowledge Retrieval (RAG) + Clinical Validation Framework**

---

## Overview

This project presents an end-to-end **AI-powered radiology decision support prototype** for detecting pneumonia from chest X-ray images using the **RSNA Pneumonia Detection Challenge dataset**.

The system integrates:

* Deep learning–based image classification
* Explainable AI (Grad-CAM heatmaps)
* Multimodal reasoning (image + clinical context)
* Expandable medical knowledge retrieval (local RAG system)
* Quantitative benchmarking and structured clinical review framework

The design emphasizes **interpretability, safety, modularity, and reproducibility**.

> ⚠ This project is a research/educational prototype and not a clinical diagnostic tool.

---

## Project Objectives

### Objective 1 — Expandable Radiology Knowledge Base (RAG)

A local Retrieval-Augmented Generation (RAG) module enables integration of medical guidelines and notes stored in the `kb/` directory.
The system retrieves relevant medical context without retraining the model.

Key features:

* No paid APIs
* Fully local TF-IDF retrieval
* Easy knowledge updates via `.txt` files

---

### Objective 2 — Multimodal Clinical Reasoning

The system combines:

* Image model probability
* Structured clinical text (symptoms, vitals)
* Retrieved guideline context

The final output produces a clinically interpretable summary including:

* Pneumonia probability
* Risk interpretation
* Suggested next steps

---

### Objective 3 — Explainable AI (Grad-CAM)

Grad-CAM overlays highlight image regions contributing to predictions.

Benefits:

* Improves transparency
* Enables qualitative clinical validation
* Reduces “black box” perception

Generated outputs:

```
outputs/gradcam/
```

---

### Objective 4 — Quantitative & Qualitative Validation

The model is evaluated using:

* ROC Curve
* AUC
* F1 Score
* Sensitivity
* Specificity
* Confusion Matrix

Example evaluation results:

* AUC ≈ 0.88
* F1 ≈ 0.61
* Sensitivity ≈ 0.54
* Specificity ≈ 0.93

Validation artifacts:

```
outputs/validation/
```

Additionally, a structured clinical review form and interview framework are provided to support human-in-the-loop validation.

---

## Model Architecture

* Backbone: **DenseNet-121**
* Input: 224 × 224 RGB chest X-ray
* Output: Binary classification (Normal vs Pneumonia)
* Loss: Cross-Entropy
* Training platform: Kaggle GPU (Tesla T4)
* Inference: CPU compatible

DenseNet was selected due to:

* Efficient gradient flow
* Strong performance in medical imaging
* Feature reuse across layers

---

## Repository Structure

```
rsna-cxr-project/
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
├── data/              # (ignored in git)
├── models/            # (ignored in git)
├── outputs/           # generated locally
│
├── README.md
└── .gitignore
```

Large datasets and trained weights are intentionally excluded from the repository.

---

## Installation

### Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Install Dependencies

```powershell
pip install numpy pandas scikit-learn pillow matplotlib tqdm
pip install torch torchvision
pip install opencv-python
pip install pydicom pylibjpeg pylibjpeg-libjpeg
```

---

## Dataset Preparation

Download the **RSNA Pneumonia Detection Challenge dataset** from Kaggle.

Place locally:

```
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

## Running the System

### 1. Multimodal Clinical Report

```powershell
python scripts/make_clinical_text.py
python scripts/multimodal_infer.py
```

### 2. Generate Grad-CAM Heatmaps

```powershell
python scripts/gradcam_generate.py
```

### 3. Evaluate Model

```powershell
python scripts/evaluate_model.py
```

### 4. Query Knowledge Base

```powershell
python scripts/rag_retrieve.py
```

---

## Key Design Principles

* Modular architecture
* Human-in-the-loop safety
* Interpretability-first approach
* Reproducible evaluation
* No paid API dependencies

---

## Limitations

* Single-institution dataset
* Binary classification only
* No external validation cohort
* Clinical text is template-generated

---

## Future Work

* Transformer-based multimodal fusion
* Calibration optimization for improved sensitivity
* External dataset validation (e.g., MIMIC-CXR)
* Lightweight web interface for deployment
* Clinical workflow simulation study

---

## Author

Amulya Anutej
B.E. Computer Science & Engineering

---

## License

Educational / Research Use Only
