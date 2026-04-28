import shap
import joblib
import pandas as pd
import os
import numpy as np

# Load model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cardio_model.pkl")

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

def explain_prediction(data):
    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    shap_values = explainer.shap_values(df)
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[0] == len(df):
            class_values = shap_values[0, :, -1]
        else:
            class_values = shap_values[-1, 0, :]
    elif shap_values.ndim == 2:
        class_values = shap_values[0]
    else:
        class_values = shap_values

    feature_importance = dict(zip(FEATURE_COLUMNS, class_values))

    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return sorted_features

if __name__ == "__main__":
    sample = [55, 1, 2, 140, 240, 0, 1, 150, 0, 1.5, 1, 0, 2]
    explanation = explain_prediction(sample)

    print("Top contributing features:")
    for feature, value in explanation[:5]:
        print(f"{feature}: {round(value, 3)}")
