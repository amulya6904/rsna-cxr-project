from shap_explainer import explain_prediction

def generate_xai_summary(patient_data):
    explanation = explain_prediction(patient_data)
    top_features = explanation[:5]

    summary_lines = []

    for feature, value in top_features:
        direction = "increased" if value > 0 else "decreased"

        meanings = {
            "chol": "cholesterol level",
            "trestbps": "resting blood pressure",
            "age": "age",
            "thalach": "maximum heart rate",
            "oldpeak": "ST depression value",
            "cp": "chest pain type",
            "exang": "exercise-induced angina",
            "ca": "number of major vessels",
            "thal": "thalassemia result"
        }

        meaning = meanings.get(feature, feature)

        summary_lines.append(
            f"- {meaning} {direction} the predicted cardiac risk."
        )

    return "\n".join(summary_lines)

if __name__ == "__main__":
    sample = [55, 1, 2, 140, 240, 0, 1, 150, 0, 1.5, 1, 0, 2]

    print("XAI Explanation")
    print("---------------")
    print(generate_xai_summary(sample))
