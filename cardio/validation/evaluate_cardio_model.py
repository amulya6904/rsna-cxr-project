import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "validation", "cardio_validation_report.txt")

FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

ALL_COLUMNS = FEATURE_COLUMNS + ["target"]

df = pd.read_csv(DATA_PATH, header=None)
df.columns = ALL_COLUMNS

df.replace("?", np.nan, inplace=True)
df = df.dropna()
df = df.astype(float)

df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

X = df[FEATURE_COLUMNS]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

output = f"""
Cardiology Risk Prediction Validation Report
===========================================

Dataset:
UCI Heart Disease - Cleveland processed dataset

Model:
Random Forest Classifier

Metrics:
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall / Sensitivity: {recall:.4f}
F1 Score: {f1:.4f}
ROC-AUC: {auc:.4f}

Confusion Matrix:
{cm}

Classification Report:
{report}

Interpretation:
The model predicts whether a patient is at possible cardiovascular disease risk
based on structured clinical features such as age, cholesterol, blood pressure,
ECG-related values, chest pain type, and exercise-induced symptoms.

Clinical Note:
This model is not a replacement for cardiologist diagnosis. It is intended as
a decision-support and academic research prototype.
"""

print(output)

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(output)

print(f"Report saved to: {REPORT_PATH}")
