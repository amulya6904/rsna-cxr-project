import os
import random
import pandas as pd

SPLITS_CSV = "data/splits.csv"
OUT_PATH = "outputs/reports/clinical_text.csv"
os.makedirs("outputs/reports", exist_ok=True)

random.seed(42)

# Simple template pool
SYMPTOMS = [
    "fever and cough", "shortness of breath", "productive cough",
    "chest pain on breathing", "fatigue and malaise", "no respiratory symptoms"
]
HISTORY = [
    "no significant past history", "history of asthma", "diabetes mellitus",
    "smoker", "recent viral illness", "hypertension"
]
VITALS = [
    ("SpO2 88%", "tachypnea"), ("SpO2 92%", "mild tachycardia"),
    ("SpO2 96%", "stable vitals"), ("SpO2 90%", "tachycardia"),
    ("SpO2 98%", "stable vitals")
]

df = pd.read_csv(SPLITS_CSV)

rows = []
for _, r in df.iterrows():
    pid = r["patientId"]
    label = r["label"]
    split = r["split"]

    # Make "clinical text" slightly correlated with label for realism
    if label == "Pneumonia":
        symptom = random.choice(SYMPTOMS[:-1])  # avoid "no symptoms"
        spo2, other = random.choice(VITALS[:4]) # lower oxygen more likely
    else:
        symptom = random.choice(SYMPTOMS)
        spo2, other = random.choice(VITALS[2:]) # more stable more likely

    history = random.choice(HISTORY)

    note = f"Patient presents with {symptom}. Past history: {history}. Vitals: {spo2}, {other}."

    rows.append({"patientId": pid, "split": split, "label": label, "clinical_text": note})

out = pd.DataFrame(rows)
out.to_csv(OUT_PATH, index=False)

print("✅ Saved:", OUT_PATH)
print(out.head(3))
