import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

IMG_DIR = "data/images"
SPLITS_CSV = "data/splits.csv"
MODEL_PATH = "models/vision/densenet121_best.pt"
OUT_DIR = "outputs/validation"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

label_map = {"Normal": 0, "Pneumonia": 1}

class CXRDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row["patientId"]
        y = label_map[row["label"]]
        img = Image.open(os.path.join(IMG_DIR, pid + ".png")).convert("RGB")
        x = self.transform(img)
        return x, y, pid

tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

splits = pd.read_csv(SPLITS_CSV)
test_df = splits[splits["split"] == "test"].copy()

test_loader = DataLoader(CXRDataset(test_df, tf), batch_size=32, shuffle=False, num_workers=0)


# Load model
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

ys, ps = [], []
with torch.no_grad():
    for x, y, _ in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        ps.extend(prob.tolist())
        ys.extend(y)

ys = np.array(ys)
ps = np.array(ps)
pred = (ps >= 0.5).astype(int)

auc = roc_auc_score(ys, ps)
f1 = f1_score(ys, pred)
cm = confusion_matrix(ys, pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn + 1e-9)
specificity = tn / (tn + fp + 1e-9)

print("AUC:", auc)
print("F1:", f1)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("CM:\n", cm)

# Save text report
report_txt = os.path.join(OUT_DIR, "test_report.txt")
with open(report_txt, "w") as f:
    f.write(f"AUC={auc:.4f}\n")
    f.write(f"F1={f1:.4f}\n")
    f.write(f"Sensitivity={sensitivity:.4f}\n")
    f.write(f"Specificity={specificity:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write(classification_report(ys, pred, target_names=["Normal","Pneumonia"]))
print("Saved:", report_txt)

# ROC plot
fpr, tpr, _ = roc_curve(ys, ps)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC={auc:.3f})")
roc_path = os.path.join(OUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", roc_path)

# Confusion matrix plot
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1], ["Normal","Pneumonia"])
plt.yticks([0,1], ["Normal","Pneumonia"])
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close()
print("Saved:", cm_path)
