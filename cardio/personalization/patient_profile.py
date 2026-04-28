def build_patient_profile(patient_data):
    age = patient_data[0]
    sex = "Male" if patient_data[1] == 1 else "Female"
    chest_pain_type = patient_data[2]
    bp = patient_data[3]
    cholesterol = patient_data[4]
    fasting_sugar = patient_data[5]
    resting_ecg = patient_data[6]
    max_heart_rate = patient_data[7]
    exercise_angina = patient_data[8]
    oldpeak = patient_data[9]

    profile = f"""
Patient Profile:
- Age: {age}
- Sex: {sex}
- Chest Pain Type: {chest_pain_type}
- Resting Blood Pressure: {bp}
- Cholesterol: {cholesterol}
- Fasting Blood Sugar > 120 mg/dl: {"Yes" if fasting_sugar == 1 else "No"}
- Resting ECG Result: {resting_ecg}
- Maximum Heart Rate: {max_heart_rate}
- Exercise Induced Angina: {"Yes" if exercise_angina == 1 else "No"}
- ST Depression Oldpeak: {oldpeak}
"""
    return profile