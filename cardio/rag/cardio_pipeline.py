import sys
import os

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag")))

from predict import predict_cardio
from cardio_llm import generate_cardio_answer

def full_cardio_analysis(patient_data):
    # Step 1: Risk prediction
    risk_result = predict_cardio(patient_data)

    # Step 2: Build query for LLM
    query = f"""
Patient details:
Age: {patient_data[0]}
BP: {patient_data[3]}
Cholesterol: {patient_data[4]}

Risk Level: {risk_result['risk_level']}

Explain the risk and suggest next steps.
"""

    # Step 3: LLM explanation
    explanation = generate_cardio_answer(query)

    return {
        "risk_prediction": risk_result,
        "llm_explanation": explanation
    }

if __name__ == "__main__":
    sample = [55, 1, 2, 140, 240, 0, 1, 150, 0, 1.5, 1, 0, 2]
    
    result = full_cardio_analysis(sample)

    print("Final Cardio Analysis")
    print("---------------------")
    print(result)