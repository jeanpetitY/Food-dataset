import json
import os

# === CONFIGURATION ===
INPUT_JSON = "json/old/fruitveg81_usda_enriched.json"  # ton fichier d’entrée
OUTPUT_JSON = "json/new/usda_fruitveg81.json"  # fichier de sortie
BASE_URL = "https://api.tsotsa.org/other_dataset/fruitveg81/"

# === Fonction pour nettoyer les noms ===
def clean_name(name: str) -> str:
    name = name.lower().strip()
    name = name.replace("'", "")  # enlever les apostrophes
    name = name.replace('"', "")  # enlever les guillemets
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = name.replace(",", "")
    name = name.replace("&", "and")
    name = name.replace("/", "_")
    return name

# === Charger le fichier JSON ===
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# === Ajouter la propriété "image" à chaque entrée ===
for item in data:
    if "class" in item:
        clean = clean_name(item["class"])
        image_url = f"{BASE_URL}{clean}.jpg"
        item["image"] = image_url
    else:
        item["image"] = None  # Sécurité au cas où le champ 'name' manque

# === Sauvegarder le nouveau fichier ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Fichier enrichi sauvegardé sous : {OUTPUT_JSON}")
