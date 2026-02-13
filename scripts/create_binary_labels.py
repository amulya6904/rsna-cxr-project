import pandas as pd

df = pd.read_csv("data/stage_2_train_labels.csv")

# One label per patientId
labels = df.groupby("patientId")["Target"].max().reset_index()

labels["label"] = labels["Target"].map({0: "Normal", 1: "Pneumonia"})

labels = labels[["patientId", "label"]]
labels.to_csv("data/labels.csv", index=False)

print("Saved: data/labels.csv")
print(labels["label"].value_counts())
