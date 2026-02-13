import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from rag_utils import retrieve

IMG_DIR = "data/images"
MODEL_PATH = "models/vision/densenet121_best.pt"
CLINICAL_CSV = "outputs/reports/clinical_text.csv"
SPLITS_CSV = "data/splits.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# Load model
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

clinical = pd.read_csv(CLINICAL_CSV)
splits = pd.read_csv(SPLITS_CSV)

# pick 1 random test sample
row = splits[splits["split"] == "test"].sample(1, random_state=42).iloc[0]
pid = row["patientId"]
true_label = row["label"]

text = clinical[clinical["patientId"] == pid]["clinical_text"].values[0]

img_path = os.path.join(IMG_DIR, f"{pid}.png")
img = Image.open(img_path).convert("RGB")
x = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(x)
    prob_pneu = torch.softmax(logits, dim=1)[0,1].item()

decision = "Likely Pneumonia" if prob_pneu >= 0.5 else "Likely Normal"
risk = "HIGH" if prob_pneu >= 0.8 else ("MEDIUM" if prob_pneu >= 0.5 else "LOW")

# RAG query based on both image + text
query = f"pneumonia chest x-ray guidance next steps {text} risk {risk}"
hits = retrieve(query, top_k=2)

rag_block = ""
for h in hits:
    snippet = h["text"].strip().replace("\n", " ")
    rag_block += f"- Source: {h['doc']} (score={h['score']:.3f})\n  Snippet: {snippet[:300]}...\n"

summary = f"""
=== Multimodal + RAG Radiology Assistant Report ===
PatientId: {pid}
True Label (for eval): {true_label}

[Clinical Text]
{text}

[Image Model Prediction]
Pneumonia probability: {prob_pneu:.3f}
Decision: {decision} (Risk: {risk})

[Retrieved Knowledge (Expandable KB)]
{rag_block}

[Final Suggested Next Steps]
- Use prediction + clinical context + retrieved guidance together.
- If red flags (e.g., SpO2 < 90% or severe distress): escalate care urgently.
- Document differential diagnoses and follow-up plan.
"""

print(summary)
