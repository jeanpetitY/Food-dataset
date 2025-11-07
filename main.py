import os
import shutil
import pandas as pd

# === CONFIGURATION ===
TXT_FILE = "category.txt"         # ton fichier texte
SOURCE_DIR = "data/UECFOOD256"            # dossier contenant les sous-dossiers 1, 2, 3, ...
DEST_DIR = "uecfood256_sampled"      # dossier de sortie

# Crée le dossier de sortie s'il n'existe pas
os.makedirs(DEST_DIR, exist_ok=True)

# === 1️⃣ Charger le fichier des labels ===
df = pd.read_csv(TXT_FILE, sep="\t") if "\t" in open(TXT_FILE).read() else pd.read_csv(TXT_FILE, sep=",")

# Normaliser les noms (remplacer espaces et caractères spéciaux)
def clean_name(name):
    name = name.strip().lower()
    name = name.replace(" ", "_").replace("'", "").replace('"', "")
    name = name.replace("/", "_").replace("&", "and").replace("-", "_")
    return name

# === 2️⃣ Parcourir chaque ligne ===
for _, row in df.iterrows():
    food_id = str(row["id"])
    food_name = clean_name(str(row["name"]))
    
    folder_path = os.path.join(SOURCE_DIR, food_id)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found for ID {food_id}")
        continue
    
    # Trouver une image (jpg, png, jpeg)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"⚠️ No image found in folder {folder_path}")
        continue
    
    # Prendre la première image
    selected_image = images[0]
    src_path = os.path.join(folder_path, selected_image)
    
    # Créer le nom de sortie
    dest_name = f"{food_name}.jpg"
    dest_path = os.path.join(DEST_DIR, dest_name)
    
    # Copier et renommer
    shutil.copy(src_path, dest_path)
    print(f"✅ Copied {selected_image} → {dest_name}")

print("\n🎉 Extraction terminée — toutes les images renommées sont dans :", DEST_DIR)
