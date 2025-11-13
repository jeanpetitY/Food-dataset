#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build a multimodal dataset linking food images to nutrient information.
Each image in the train dataset is associated with its corresponding nutrient data
for fine-tuning a Vision-Language Model (VLM) such as Qwen2.5-VL or LLaVA.

Author: Jean Petit BIKIM
"""

import os
import json
from tqdm import tqdm
from typing import List, Dict, Any


# --------------------------- Utility Functions --------------------------- #

import re

import re

def clean_label(label: str) -> str:
    """
    Clean and normalize class labels while preserving uppercase and underscores.

    ✅ Ce que cette version fait :
    - Garde les majuscules
    - Conserve les underscores existants
    - Remplace chaque espace ou tiret par un underscore (même si cela crée plusieurs underscores)
    - Supprime les autres caractères spéciaux (&, ', ., etc.)
    - Ne supprime pas les underscores initiaux/finals
    """
    if not isinstance(label, str):
        return ""

    # Remplacer certains symboles spécifiques
    label = label.replace("&", "and")
    label = label.replace("'n'", "n")
    label = label.replace("'n", "n")
    label = label.replace("n'", "n")

    # Étape 1 : Remplacer chaque espace ou tiret par UN underscore (1:1)
    label = re.sub(r"[-\s]", "_", label)

    # Étape 2 : Supprimer tous les caractères spéciaux sauf underscore
    label = re.sub(r"[^A-Za-z0-9_]", "", label)

    # Étape 3 : Retourner tel quel (sans supprimer les underscores multiples)
    return label



def safe_dir_name(name: str) -> str:
    """Make a filesystem-safe directory name from a label."""
    name = name.strip()
    name = re.sub(r"[^\w\s\-\.\(\)&]", " ", name)  # remove exotic chars
    name = re.sub(r"\s+", " ", name)               # collapse spaces
    return name.strip()



def load_nutrient_json(json_path: str) -> List[Dict[str, Any]]:
    """Load the JSON file containing food nutrient data."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} food entries from {json_path}")
    return data


def build_components_list(nutrients: List[Dict[str, Any]]) -> List[str]:
    """
    Convert a list of nutrient dicts into readable text entries.
    Example: [{"name": "Protein", "value": 1.7, "unit": "g"}]
             → ["Protein (1.7 g)"]
    """
    components = []
    for n in nutrients:
        if "name" in n and "value" in n and "unit" in n:
            value = n['value'] or 0
            components.append(f"{n['name']} ({value} {n['unit']})")
    return components


def build_instruction() -> str:
    """
    Return a consistent instruction for the multimodal model.
    """
    return (
        "Given this food image, identify its main nutrients "
        "and their corresponding nutritional values in grams, milligrams, or kcal."
    )


# --------------------------- Main Dataset Builder --------------------------- #

def build_multimodal_dataset(
    json_source: str,
    train_dir: str,
    output_file: str
) -> None:
    """
    Build a multimodal JSON dataset linking image paths to nutrient data.

    Args:
        json_source (str): Path to JSON file containing nutrient info.
        train_dir (str): Path to training dataset (with class subfolders).
        output_file (str): Path to save the final multimodal JSON.
    """
    data = load_nutrient_json(json_source)
    instruction = build_instruction()
    records = []

    for item in tqdm(data, desc="🔗 Linking images to nutrients"):
        # Normalize class name
        class_name = clean_label(item.get("class", ""))
        if class_name == "chicken_n_egg_on_rice":
            class_name = "chicken__n__egg_on_rice"
        elif class_name == "salt_and_pepper_fried_shrimp_with_shell":
            class_name = "salt_&_pepper_fried_shrimp_with_shell"
        if not class_name:
            continue

        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Skipping: {class_dir} not found.")
            continue

        components = build_components_list(item.get("nutrients", []))

        # Loop through each image in the class directory
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, img_name)

            records.append({
                "image": img_path,
                "label": class_name,
                "instruction": instruction,
                "components": components
            })

    # Save the result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(records)} multimodal samples to {output_file}")
    

import json
from typing import List, Dict
import json
from typing import List, Dict

def merge_json_files(file1: str, file2: str, file3: str, output_file: str) -> None:
    """
    Fusionne trois fichiers JSON (listes d’objets) avec priorité :
      file1 > file2 > file3
    En vérifiant l’unicité sur la combinaison (label + image).
    """
    def load_json(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def make_key(item: Dict) -> str:
        """Construit une clé unique à partir du label + image."""
        label = item.get("label", "").strip()
        image = item.get("image", "").strip()
        return f"{label}|{image}"  # Séparateur sûr

    # Charger les fichiers
    data1 = load_json(file1)
    data2 = load_json(file2)
    data3 = load_json(file3)

    # Dictionnaire pour suivre les clés uniques
    merged_data = {}

    # Fusion en respectant la priorité (file1 > file2 > file3)
    for source, name in zip([data3, data2, data1], ["file3", "file2", "file1"]):
        for item in source:
            key = make_key(item)
            if not key.strip("|"):  # Ignore les entrées sans label ou image
                continue

            if key not in merged_data:
                merged_data[key] = item
            else:
                # Le couple (label+image) existe déjà dans un fichier prioritaire
                print(f"[INFO] Entrée ignorée ({name}) — déjà présente : {key}")

    # Convertir en liste finale
    merged_list = list(merged_data.values())

    # Sauvegarde
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=4, ensure_ascii=False)

    print(f"✅ Fusion terminée : {len(merged_list)} éléments sauvegardés dans '{output_file}'.")



# --------------------------- Main Entry Point --------------------------- #

def main():
    # === Config ===
    json_source = "json/new/usda_fruitveg81.json"       # Input JSON with nutrients
    train_dir = "dataset/merged_dataset/test"         # Folder containing image subdirs
    output_file = "dataset/multimodal/not_merged/test/fruitveg81_vlm_test.json"        # Output JSON file

    build_multimodal_dataset(json_source, train_dir, output_file)


if __name__ == "__main__":
    # main()
    merge_json_files(
        "dataset/multimodal/not_merged/test/food101_vlm_test.json",
        "dataset/multimodal/not_merged/test/fruitveg81_vlm_test.json",
        "dataset/multimodal/not_merged/test/uecfood256_vlm_test.json",
        "dataset/multimodal/merged/test_final.json"
    )
