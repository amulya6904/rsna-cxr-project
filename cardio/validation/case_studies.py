import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(BASE_DIR, "rag"))

from cardio_pipeline import full_cardio_analysis

cases = [
    {
        "case_id": "Case 1 - Low Risk",
        "data": [35, 0, 1, 118, 180, 0, 0, 170, 0, 0.2, 1, 0, 2],
    },
    {
        "case_id": "Case 2 - Medium Risk",
        "data": [55, 1, 2, 140, 240, 0, 1, 150, 0, 1.5, 1, 0, 2],
    },
    {
        "case_id": "Case 3 - High Risk",
        "data": [67, 1, 4, 160, 290, 1, 2, 110, 1, 3.2, 2, 2, 3],
    },
]

for case in cases:
    print("=" * 60)
    print(case["case_id"])
    print("=" * 60)

    result = full_cardio_analysis(case["data"])

    print("Risk Prediction:")
    print(result["risk_prediction"])

    print("\nTop Features:")
    print(result["top_features"])

    print("\nXAI Summary:")
    print(result["xai_summary"])

    print("\nPersonalized Explanation:")
    print(result["personalized_explanation"])
    print()
