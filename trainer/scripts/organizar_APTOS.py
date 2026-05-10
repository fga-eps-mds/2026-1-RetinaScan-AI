import os
import shutil
import pandas as pd

BASE = os.path.join("..", "APTOS_2019")
OUT = os.path.join("..", "aptos_binary")

SPLITS = {
    "train": {
        "img_dir": os.path.join(BASE, "train_images", "train_images"),
        "csv": os.path.join(BASE, "train_1.csv"),
    },
    "val": {
        "img_dir": os.path.join(BASE, "val_images", "val_images"),
        "csv": os.path.join(BASE, "valid.csv"),
    },
    "test": {
        "img_dir": os.path.join(BASE, "test_images", "test_images"),
        "csv": os.path.join(BASE, "test.csv"),
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
        
        # APTOS usa id_code_*.png
        import glob
        matches = glob.glob(os.path.join(img_dir, f"{image_id}_*.{ext}"))
        if matches:
            return matches[0]
    return None

for split, cfg in SPLITS.items():
    df = pd.read_csv(cfg["csv"])
    cols = df.columns.tolist()

    id_col = "id_code"  # APTOS padrão
    df["binary_label"] = (df["diagnosis"] > 0).astype(int)

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