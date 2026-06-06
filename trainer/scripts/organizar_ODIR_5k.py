import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.join("..", "ODIR-5K")
OUT = os.path.join("..", "odir_binary")

IMG_DIRS = [
    os.path.join(BASE, "Training Images"),
    os.path.join(BASE, "Testing Images"),
]

CSV_PATH = os.path.join(BASE, "data.xlsx")

# =========================
# cria estrutura
# =========================
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUT, split, "0"), exist_ok=True)
    os.makedirs(os.path.join(OUT, split, "1"), exist_ok=True)

# =========================
# find image (procura em ambas pastas)
# =========================
def find_image(image_name):
    for img_dir in IMG_DIRS:
        path = os.path.join(img_dir, image_name)
        if os.path.exists(path):
            return path
    return None

# =========================
# label binária
# =========================
def get_label(keyword):
    if pd.isna(keyword):
        return 1
    keyword = keyword.lower()
    if "normal fundus" in keyword:
        return 0
    return 1

# =========================
# load dataset
# =========================
df = pd.read_excel(CSV_PATH)

# =========================
# EXPANDE left/right (igual RFMiD mas adaptado)
# =========================
rows = []

for _, row in df.iterrows():
    patient_id = row["ID"]

    # LEFT
    if pd.notna(row["Left-Fundus"]):
        rows.append({
            "image": row["Left-Fundus"],
            "label": get_label(row["Left-Diagnostic Keywords"]),
            "patient_id": patient_id
        })

    # RIGHT
    if pd.notna(row["Right-Fundus"]):
        rows.append({
            "image": row["Right-Fundus"],
            "label": get_label(row["Right-Diagnostic Keywords"]),
            "patient_id": patient_id
        })

df_expanded = pd.DataFrame(rows)

# =========================
# SPLIT (🔥 CORRETO: por paciente)
# =========================
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(test_size=0.3, random_state=42)

train_idx, temp_idx = next(
    gss.split(df_expanded, df_expanded["label"], groups=df_expanded["patient_id"])
)

train_df = df_expanded.iloc[train_idx]
temp_df  = df_expanded.iloc[temp_idx]

gss2 = GroupShuffleSplit(test_size=0.5, random_state=42)

val_idx, test_idx = next(
    gss2.split(temp_df, temp_df["label"], groups=temp_df["patient_id"])
)

val_df  = temp_df.iloc[val_idx]
test_df = temp_df.iloc[test_idx]

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df,
}

# =========================
# COPY (igual RFMiD)
# =========================
for split, df_split in splits.items():
    for _, row in df_split.iterrows():
        image_name = row["image"]
        label = str(int(row["label"]))

        src = find_image(image_name)
        if src is None:
            print(f"[WARN] imagem não encontrada: {image_name}")
            continue

        dst = os.path.join(OUT, split, label, image_name)
        shutil.copy2(src, dst)

print("Dataset ODIR binário criado em:", OUT)