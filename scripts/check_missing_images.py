import os
import pandas as pd

img_dir = "data/images"
splits = pd.read_csv("data/splits.csv")

missing = []
for pid in splits["patientId"]:
    if not os.path.exists(os.path.join(img_dir, f"{pid}.png")):
        missing.append(pid)

print("Total rows in splits:", len(splits))
print("Missing PNGs:", len(missing))
if missing:
    print("First 10 missing:", missing[:10])
