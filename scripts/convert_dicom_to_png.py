import os
import pydicom
import cv2
import numpy as np
from tqdm import tqdm

DICOM_DIR = "data/dicom"
OUT_DIR = "data/images"
os.makedirs(OUT_DIR, exist_ok=True)

files = [f for f in os.listdir(DICOM_DIR) if f.lower().endswith(".dcm")]
print(f"Found {len(files)} DICOM files in {DICOM_DIR}")

for f in tqdm(files, desc="Converting DICOM to PNG"):
    dcm_path = os.path.join(DICOM_DIR, f)
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array.astype(np.float32)

    # Normalize to 0–255
    img -= img.min()
    mx = img.max()
    if mx > 0:
        img /= mx
    img = (img * 255).astype(np.uint8)

    out_path = os.path.join(OUT_DIR, os.path.splitext(f)[0] + ".png")
    cv2.imwrite(out_path, img)

print("Done! PNG images saved to data/images")
