import os
import shutil
import pandas as pd

BASE = os.path.join("..", "A. RFMiD_All_Classes_Dataset")
OUT = os.path.join("..", "rfmid_binary")

SPLITS = {
    "train": {
        "img_dir": os.path.join(BASE, "1. Original Images", "a. Training Set"),
        "csv": os.path.join(BASE, "2. Groundtruths", "a. RFMiD_Training_Labels.csv"),
    },
    "val": {
        "img_dir": os.path.join(BASE, "1. Original Images", "b. Validation Set"),
        "csv": os.path.join(BASE, "2. Groundtruths", "b. RFMiD_Validation_Labels.csv"),
    },
    "test": {
        "img_dir": os.path.join(BASE, "1. Original Images", "c. Testing Set"),
        "csv": os.path.join(BASE, "2. Groundtruths", "c. RFMiD_Testing_Labels.csv"),
    },
}

for split in SPLITS:
    os.makedirs(os.path.join(OUT, split, "0"), exist_ok=True)
    os.makedirs(os.path.join(OUT, split, "1"), exist_ok=True)

def find_image(img_dir, image_id):
    for ext in [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]:
        path = os.path.join(img_dir, str(image_id) + ext)
        if os.path.exists(path):
            return path
    return None

for split, cfg in SPLITS.items():
    df = pd.read_csv(cfg["csv"])
    cols = df.columns.tolist()

    id_col = "ID" if "ID" in cols else cols[0]

    if "Disease_Risk" in cols:
        df["binary_label"] = df["Disease_Risk"].astype(int)
    else:
        disease_cols = [c for c in cols if c != id_col]
        if "Normal" in disease_cols:
            disease_cols.remove("Normal")
        df["binary_label"] = (df[disease_cols].sum(axis=1) > 0).astype(int)

    for _, row in df.iterrows():
        image_id = row[id_col]
        label = str(int(row["binary_label"]))
        src = find_image(cfg["img_dir"], image_id)
        if src is None:
            print(f"[WARN] imagem não encontrada: {image_id}")
            continue

        dst = os.path.join(OUT, split, label, os.path.basename(src))
        shutil.copy2(src, dst)

print("Dataset binário criado em:", OUT)