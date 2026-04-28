import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(BASE_DIR, "models"))
sys.path.append(os.path.join(BASE_DIR, "rag"))
sys.path.append(os.path.join(BASE_DIR, "personalization"))
sys.path.append(os.path.join(BASE_DIR, "xai"))

from predict import predict_cardio
from cardio_llm import generate_cardio_answer
from patient_profile import build_patient_profile
from xai_interpreter import generate_xai_summary
from shap_explainer import explain_prediction

def full_cardio_analysis(patient_data):
    risk_result = predict_cardio(patient_data)
    patient_profile = build_patient_profile(patient_data)

    xai_summary = generate_xai_summary(patient_data)
    shap_features = explain_prediction(patient_data)

    top_features = [
        {
            "feature": feature,
            "impact": float(round(value, 4))
        }
        for feature, value in shap_features[:5]
    ]

    query = f"""
{patient_profile}

Model Risk Result:
- Prediction: {risk_result['prediction']}
- Risk Probability: {risk_result['risk_probability']}
- Risk Level: {risk_result['risk_level']}

Explainability Summary:
{xai_summary}

Give a personalized explanation and suggest safe next steps.
Do not give final medical diagnosis.
Mention that cardiologist review is required.
"""

    explanation = generate_cardio_answer(query)

    return {
        "patient_profile": patient_profile,
        "risk_prediction": risk_result,
        "top_features": top_features,
        "xai_summary": xai_summary,
        "personalized_explanation": explanation
    }

if __name__ == "__main__":
    sample = [55, 1, 2, 140, 240, 0, 1, 150, 0, 1.5, 1, 0, 2]
    result = full_cardio_analysis(sample)

    print("Final Explainable Cardio Analysis")
    print("---------------------------------")
    print(result)
