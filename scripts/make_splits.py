import pandas as pd
from sklearn.model_selection import train_test_split

labels = pd.read_csv("data/labels.csv")

# 70% train, 15% val, 15% test (stratified)
train_df, temp_df = train_test_split(
    labels, test_size=0.30, random_state=42, stratify=labels["label"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

splits = pd.concat([train_df, val_df, test_df], ignore_index=True)
splits.to_csv("data/splits.csv", index=False)

print("Saved: data/splits.csv")
print("\nSplit counts:")
print(splits["split"].value_counts())
print("\nSplit × Label counts:")
print(splits.groupby(["split", "label"]).size())
