import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------
# Paths
# -------------------
IMG_DIR = "data/images"
SPLITS_CSV = "data/splits.csv"
MODEL_PATH = "models/vision/densenet121_best.pt"
OUT_DIR = "outputs/gradcam"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# -------------------
# Load model
# -------------------
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)

# Disable in-place ReLU to avoid backward-hook inplace/view errors
def disable_inplace_relu(m):
    if isinstance(m, torch.nn.ReLU):
        m.inplace = False

model.apply(disable_inplace_relu)
model.eval()

# Target layer for Grad-CAM
target_layer = model.features[-1]

# -------------------
# Preprocessing
# -------------------
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -------------------
# Grad-CAM hooks
# -------------------
activations = None
gradients = None

def forward_hook(module, inp, out):
    global activations
    activations = out

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer.register_forward_hook(forward_hook)
# Warning is okay; works for our purpose
target_layer.register_backward_hook(backward_hook)

# -------------------
# Grad-CAM functions
# -------------------
def make_gradcam(pil_img: Image.Image):
    global activations, gradients

    pil_img = pil_img.convert("RGB")

    x = tf(pil_img).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = int(np.argmax(probs))
    pred_prob = float(probs[pred_class])

    model.zero_grad(set_to_none=True)
    score = logits[0, pred_class]
    score.backward()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    # activations/gradients: [1, C, H, W] (H,W often 7x7)
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * activations).sum(dim=1)            # [1, H, W]
    cam = torch.relu(cam).squeeze(0)

    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)  # 0..1

    return cam, pred_class, pred_prob

def overlay_cam_on_image(pil_img: Image.Image, cam: np.ndarray):
    # Base image
    pil_img = pil_img.convert("RGB")
    img = np.array(pil_img)
    img = cv2.resize(img, (224, 224))

    # Resize CAM to match image size (fix for 7x7 CAM)
    cam_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    return overlay

# -------------------
# Choose samples
# -------------------
splits = pd.read_csv(SPLITS_CSV)
test_df = splits[splits["split"] == "test"].copy()

normals = test_df[test_df["label"] == "Normal"]["patientId"].tolist()
pneums  = test_df[test_df["label"] == "Pneumonia"]["patientId"].tolist()

random.seed(42)
sample_normals = random.sample(normals, 5)
sample_pneums  = random.sample(pneums, 5)

samples = [(pid, "Normal") for pid in sample_normals] + [(pid, "Pneumonia") for pid in sample_pneums]

print("Generating Grad-CAM for 10 samples...")

# -------------------
# Generate + save
# -------------------
saved = 0
for pid, true_label in samples:
    img_path = os.path.join(IMG_DIR, f"{pid}.png")
    if not os.path.exists(img_path):
        print("Missing image:", img_path)
        continue

    pil_img = Image.open(img_path).convert("RGB")

    cam, pred_class, pred_prob = make_gradcam(pil_img)
    overlay = overlay_cam_on_image(pil_img, cam)

    pred_name = "Pneumonia" if pred_class == 1 else "Normal"
    out_name = f"{pid}_true-{true_label}_pred-{pred_name}_{pred_prob:.2f}.png"
    out_path = os.path.join(OUT_DIR, out_name)

    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    saved += 1

print(f"✅ Saved {saved} overlays to: {OUT_DIR}")
