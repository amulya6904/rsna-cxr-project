import base64
import io
import os
from datetime import datetime
from typing import List, Optional
import json

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from torchvision import models, transforms

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import cv2
except Exception:
    cv2 = None

APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(APP_ROOT, "models", "vision", "densenet121_best.pt")
KB_DIR = os.path.join(APP_ROOT, "kb")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Nuvexa Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Model + Grad-CAM
# -------------------
model = None
target_layer = None
activations = None
gradients = None

tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def disable_inplace_relu(m):
    if isinstance(m, torch.nn.ReLU):
        m.inplace = False


def forward_hook(module, inp, out):
    global activations
    activations = out


def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]


def load_model():
    global model, target_layer
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.apply(disable_inplace_relu)
    model.eval()

    target_layer = model.features[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)


def read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    upload.file.seek(0)
    name = (upload.filename or "").lower()

    if name.endswith(".dcm"):
        if pydicom is None:
            raise ValueError("pydicom is not installed for DICOM support.")
        ds = pydicom.dcmread(io.BytesIO(data))
        img = ds.pixel_array.astype(np.float32)
        img -= img.min()
        mx = img.max()
        if mx > 0:
            img /= mx
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img).convert("RGB")

    return Image.open(io.BytesIO(data)).convert("RGB")


def make_gradcam(pil_img: Image.Image):
    global activations, gradients
    activations = None
    gradients = None

    x = tf(pil_img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = int(np.argmax(probs))
    pred_prob = float(probs[pred_class])
    pneu_prob = float(probs[1]) if probs.shape[0] > 1 else pred_prob

    model.zero_grad(set_to_none=True)
    score = logits[0, pred_class]
    score.backward()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)
    cam = torch.relu(cam).squeeze(0)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    return cam, pred_class, pred_prob, pneu_prob


def overlay_cam_on_image(pil_img: Image.Image, cam: np.ndarray) -> Optional[Image.Image]:
    if cv2 is None:
        return None

    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (224, 224))

    cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def load_kb_snippets(limit: int = 2):
    snippets = []
    if not os.path.isdir(KB_DIR):
        return snippets
    for fname in os.listdir(KB_DIR):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(KB_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip().replace("\n", " ")
            snippets.append(
                {
                    "source": fname,
                    "score": 0.5,
                    "snippet": text[:300],
                }
            )
        except Exception:
            continue
    return snippets[:limit]


def build_next_steps(spo2: Optional[float], risk: str) -> List[str]:
    steps = [
        "Correlate radiographic findings with clinical symptoms and vitals.",
        "Review the heatmap for focal attention prior to escalation.",
        "Document differential diagnoses and follow-up plan.",
    ]
    if spo2 is not None and spo2 < 90:
        steps.insert(0, "Escalate care urgently due to hypoxia (SpO2 < 90%).")
    if risk == "high":
        steps.append("Consider empiric antibiotics per institutional guidelines.")
    return steps


def build_red_flags(spo2: Optional[float]) -> List[str]:
    flags = [
        "Severe respiratory distress or altered sensorium",
        "Rapid progression of symptoms",
        "Hypotension or signs of sepsis",
    ]
    if spo2 is not None and spo2 < 90:
        flags.insert(0, "SpO2 < 90% on room air")
    return flags


@app.on_event("startup")
def startup():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    age: Optional[int] = Form(None),
    sex: Optional[str] = Form(None),
    spo2: Optional[float] = Form(None),
    symptoms: Optional[str] = Form(None),
    history: Optional[str] = Form(None),
    urgency: Optional[str] = Form(None),
):
    pil_img = read_image(image)

    cam, pred_class, pred_prob, pneu_prob = make_gradcam(pil_img)
    overlay = overlay_cam_on_image(pil_img, cam)

    # Calibrate pneumonia probability with clinical context (does not change model logits)
    symptom_list = []
    if symptoms:
        try:
            symptom_list = json.loads(symptoms)
        except Exception:
            symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]

    symptom_set = {s.lower() for s in symptom_list}
    has_key_symptoms = any(s in symptom_set for s in ["fever", "cough", "shortness of breath", "chest pain"])
    severe_hypoxia = spo2 is not None and spo2 < 90
    urgent_flag = urgency is not None and urgency.lower() == "urgent"

    adjusted_prob = pneu_prob
    if severe_hypoxia:
        adjusted_prob = max(adjusted_prob, 0.85)
    elif has_key_symptoms:
        adjusted_prob = max(adjusted_prob, 0.65)
    if len(symptom_set) >= 2:
        adjusted_prob = max(adjusted_prob, 0.7)
    if urgent_flag:
        adjusted_prob = max(adjusted_prob, 0.75)

    pneu_prob = float(min(1.0, max(0.0, adjusted_prob)))

    risk = "high" if pneu_prob >= 0.7 else "moderate" if pneu_prob >= 0.4 else "low"

    model_summary = (
        "DenseNet-121 inference completed. Review heatmap and clinical context before final decision."
    )

    return {
        "id": f"CXR-{int(datetime.utcnow().timestamp())}",
        "created_at": datetime.utcnow().isoformat(),
        "probability": pneu_prob,
        "risk": risk,
        "model_summary": model_summary,
        "confidence": pred_prob,
        "image_base64": image_to_base64(pil_img.resize((224, 224))),
        "gradcam_base64": image_to_base64(overlay) if overlay is not None else None,
        "knowledge": load_kb_snippets(),
        "next_steps": build_next_steps(spo2, risk),
        "red_flags": build_red_flags(spo2),
        "clinical": {
            "age": age,
            "sex": sex,
            "spo2": spo2,
            "symptoms": symptoms,
            "history": history,
            "urgency": urgency,
        },
    }

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: ChatRequest):
    import requests
    try:
        # Simple fallback to ollama if available, otherwise mock response
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2:latest",
            "prompt": f"You are a Pulmonology AI. Answer: {request.query}",
            "stream": False
        }, timeout=5)
        if res.status_code == 200:
            return {"answer": res.json().get("response", "No response from Llama")}
    except Exception:
        pass
    
    return {"answer": f"Pulmonology Assistant: I see you asked about '{request.query}'. Please ensure Ollama is running locally for full dynamic responses."}
