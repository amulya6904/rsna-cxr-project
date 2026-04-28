import os
import joblib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_model.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "validation", "cardio_roc_curve.png")

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

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = joblib.load(MODEL_PATH)

y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Cardiology Risk Prediction ROC Curve")
plt.legend(loc="lower right")
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

print(f"ROC curve saved to: {OUTPUT_PATH}")
