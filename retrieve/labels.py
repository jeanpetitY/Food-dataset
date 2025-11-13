import os
import re
from difflib import SequenceMatcher
import csv
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from PIL import Image


food101_labels = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
    'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
    'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros',
    'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich',
    'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette',
    'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 'panna_cotta',
    'peking_duck', 'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto', 'samosa',
    'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits', 'spaghetti_bolognese',
    'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake', 'sushi',
    'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
]

def load_food_classes(metadata_path: str):
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            rows = [row for row in reader if len(row) >= 2]

        # Création du DataFrame à partir du contenu lu
        df = pd.DataFrame(rows[1:], columns=rows[0])  # première ligne = header

        # Nettoyage et conversion des colonnes
        df["id"] = df["id"].astype(int)
        df["name"] = df["name"].astype(str).str.strip()

        # Conversion en liste de tuples
        data = list(df.itertuples(index=False, name=None))

        print(f"✅ {len(data)} classes chargées avec succès.")
        return data

    except FileNotFoundError:
        print(f"❌ Erreur : le fichier '{metadata_path}' est introuvable.")
        return []
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du fichier : {e}")
        return []


uecfood256_labels = load_food_classes("dataset/uecfood256/uecfood255_categories.txt")
# print(uecfood256_labels)

def prepare_food101_dataset(output_dir: str = "food101", train_per_class=750, test_per_class=250):
    """
    Downloads the Food101 dataset from Hugging Face and saves it locally
    with a structured folder format:
        output_dir/
        ├── train/
        │   ├── apple_pie/
        │   ├── ...
        └── test/
            ├── apple_pie/
            ├── ...

    Each class will contain up to x0 train and x test images.

    Args:
        output_dir (str): Path to save the dataset.
        train_per_class (int): Number of training images per class.
        test_per_class (int): Number of testing images per class.
    """
    print("Loading Food101 dataset from Hugging Face...")
    dataset = load_dataset("ethz/food101")

    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # Create directory structure
    for d in [train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    def save_split(split_name, split_data, max_per_class):
        print(f"\n📦 Processing split: {split_name} ...")
        class_counts = {}

        for i, example in enumerate(split_data):
            label = example["label"]
            label_name = split_data.features["label"].int2str(label)
            img = example["image"]

            # Initialize counter per class
            if label_name not in class_counts:
                class_counts[label_name] = 0

            # Limit number of images per class
            if class_counts[label_name] >= max_per_class:
                continue

            # Build path
            class_dir = output_dir / split_name / label_name
            class_dir.mkdir(parents=True, exist_ok=True)

            img_path = class_dir / f"{label_name}_{class_counts[label_name]:04d}.jpg"
            img.save(img_path)

            class_counts[label_name] += 1

            if i % 1000 == 0:
                print(f"   ✅ {i} images processed...")

        print(f"✅ Finished {split_name}: {sum(class_counts.values())} images saved.")

    # Save train and test splits
    # save_split("train", dataset["train"], train_per_class)
    save_split("test", dataset["validation"], test_per_class)

    print("\n🎉 Dataset successfully prepared in:", output_dir)

# prepare_food101_dataset("dataset/food101", train_per_class=750, test_per_class=250)

fruitveg81_labels = os.listdir("dataset/not_merged/fruitveg81/train")
uecfood256_labels = [label.lower() for _, label in uecfood256_labels]


def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[-_']", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

import os
import re
import csv
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Tuple
from difflib import SequenceMatcher

# ---------------------------
# Helpers
# ---------------------------

def safe_dir_name(name: str) -> str:
    """Make a filesystem-safe directory name from a label."""
    name = name.strip()
    name = re.sub(r"[^\w\s\-\.\(\)&]", " ", name)  # remove exotic chars
    name = re.sub(r"\s+", " ", name)               # collapse spaces
    return name.strip()

def read_uec_label_map(labels_txt_path: str) -> Dict[str, str]:
    """
    Read the UEC label mapping file (TSV or 'id\\tname' text).
    Returns a dict: { "1": "rice", "2": "eels on rice", ... }
    """
    labels = {}
    with open(labels_txt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [row for row in reader if len(row) >= 2]
    header = rows[0]
    # try to detect header
    if header[0].lower() in {"id", "class_id"} and header[1].lower() in {"name", "class_name"}:
        data_rows = rows[1:]
    else:
        data_rows = rows

    for rid, name in data_rows:
        labels[str(rid).strip()] = name.strip()
    return labels

def list_class_names(root_dir: Path, split: str) -> List[str]:
    """
    List class directory names under root/split (e.g., Food101/FruitVeg81).
    """
    target = root_dir / split
    if not target.exists():
        return []
    return sorted([d.name for d in target.iterdir() if d.is_dir()])

def list_uec_present_ids(uec_root: Path, split: str) -> List[str]:
    """
    For UEC: list the present class IDs (folder names like '1', '2', ...).
    """
    target = uec_root / split
    if not target.exists():
        return []
    return sorted([d.name for d in target.iterdir() if d.is_dir() and d.name.isdigit()])

def copy_split(src_root: Path, split: str, dst_root: Path, skip_classes: set = None):
    """
    Copy split from src_root/split/* to dst_root/split/* (class folders by name).
    If skip_classes is provided, skip these class names (already normalized decision made outside).
    """
    src_split = src_root / split
    dst_split = dst_root / split
    if not src_split.exists():
        return
    dst_split.mkdir(parents=True, exist_ok=True)

    for class_dir in src_split.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if skip_classes and class_name in skip_classes:
            print(f"⏩ Skip (duplicate) class: {class_name}")
            continue

        out_dir = dst_split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in class_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)

def copy_split_uec(
    uec_root: Path,
    split: str,
    dst_root: Path,
    id_to_name: Dict[str, str],
    skip_normalized_names: set,
    normalize: Callable[[str], str],
):
    """
    Copy UEC split (uec_root/split/ID/*) into dst_root/split/CLASS_NAME/*,
    but skip classes whose normalized label is in skip_normalized_names.
    """
    src_split = uec_root / split
    dst_split = dst_root / split
    if not src_split.exists():
        return
    dst_split.mkdir(parents=True, exist_ok=True)

    for id_dir in src_split.iterdir():
        if not id_dir.is_dir() or not id_dir.name.isdigit():
            continue

        class_id = id_dir.name
        label = id_to_name.get(class_id)
        if not label:
            print(f"⚠️  Missing label for UEC id={class_id}, skipping.")
            continue

        normalized = normalize(label)
        if normalized in skip_normalized_names:
            print(f"⏩ Skip UEC (duplicate w/ Food101): id={class_id}, label='{label}'")
            continue

        class_name = safe_dir_name(label)
        out_dir = dst_split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in id_dir.glob("*"):
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)

# ---------------------------
# Your compare function (uses provided normalize)
# ---------------------------

def compare_food_labels(
    food101_labels: List[str],
    uecfood256_labels: List[str],
    normalize: Callable[[str], str],
    threshold: float = 0.8,
) -> Dict[str, List]:
    """
    Compare two label lists using the provided normalize() to compute:
      - exact matches (intersection)
      - fuzzy matches (SequenceMatcher > threshold)
    Returns dict with "exact_matches" and "fuzzy_matches" [(f101_norm, uec_norm, score)]
    """
    normalized_food101 = [normalize(n) for n in food101_labels]
    normalized_uec = [normalize(n) for n in uecfood256_labels]

    exact_matches = sorted(set(normalized_food101) & set(normalized_uec))

    fuzzy_matches: List[Tuple[str, str, float]] = []
    for f in normalized_food101:
        for u in normalized_uec:
            if f in exact_matches:
                continue
            ratio = SequenceMatcher(None, f, u).ratio()
            score = round(ratio, 2)
            if score > threshold:
                fuzzy_matches.append((f, u, score))

    print("🎯 Exact matches:", len(exact_matches))
    print("🤖 Fuzzy matches (>", threshold, "):", len(fuzzy_matches))
    return {"exact_matches": exact_matches, "fuzzy_matches": fuzzy_matches}

# ---------------------------
# Main merge function
# ---------------------------

def merge_food_datasets(
    food101_dir: str,
    fruitveg81_dir: str,
    uec_dir: str,
    uec_labels_txt: str,
    output_dir: str,
    normalize: Callable[[str], str],
    threshold: float = 0.8,
):
    """
    Merge Food101, FruitVeg81, and UECFood256 into a unified dataset:
      output/
        train/<class_name>/*.jpg
        test/<class_name>/*.jpg

    Rules:
      - Food101 & FruitVeg81 already have class-named subfolders; copy as-is.
      - UEC has id folders (1..255) with a TXT map id->name. We map to class_name folders.
      - Duplicates between Food101 and UEC (exact or fuzzy by 'normalize') => KEEP Food101, SKIP UEC.

    Args:
        food101_dir: root of Food101 (contains train/ and test/ with class folders)
        fruitveg81_dir: root of FruitVeg81 (same structure)
        uec_dir: root of UEC (contains train/ and test/ with numeric id folders)
        uec_labels_txt: path to 'id\tname' text file for UEC
        output_dir: destination root
        normalize: your normalization function
        threshold: fuzzy similarity threshold
    """
    food101_root = Path(food101_dir)
    fruitveg_root = Path(fruitveg81_dir)
    uec_root = Path(uec_dir)
    out_root = Path(output_dir)

    # 1) Collect class names (Food101 & FruitVeg81)
    f101_train = list_class_names(food101_root, "train")
    f101_test = list_class_names(food101_root, "test")
    f81_train = list_class_names(fruitveg_root, "train")
    f81_test  = list_class_names(fruitveg_root, "test")

    food101_classes_all = sorted(set(f101_train + f101_test))
    print(f"📦 Food101 classes: {len(food101_classes_all)}")

    # 2) UEC labels (id -> name) + present IDs per split
    id_to_name = read_uec_label_map(uec_labels_txt)
    uec_train_ids = list_uec_present_ids(uec_root, "train")
    uec_test_ids  = list_uec_present_ids(uec_root, "test")

    # Build UEC label lists that actually exist in filesystem per split
    uec_train_labels = [id_to_name[i] for i in uec_train_ids if i in id_to_name]
    uec_test_labels  = [id_to_name[i] for i in uec_test_ids if i in id_to_name]
    uec_all_labels   = sorted(set(uec_train_labels + uec_test_labels))
    print(f"🍱 UEC labels present: {len(uec_all_labels)}")

    # 3) Compare Food101 vs UEC using provided normalize()
    comp = compare_food_labels(food101_classes_all, uec_all_labels, normalize, threshold=threshold)
    exact_matches = set(comp["exact_matches"])
    fuzzy_matches_uec_norm = {u for _, u, _ in comp["fuzzy_matches"]}
    skip_norm = set(exact_matches) | set(fuzzy_matches_uec_norm)

    print(f"⚖️ Duplicates to skip from UEC: {len(skip_norm)}")

    # 4) Create output structure
    (out_root / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "test").mkdir(parents=True, exist_ok=True)

    # 5) Copy Food101 & FruitVeg81 as-is
    print("📂 Copy Food101...")
    copy_split(food101_root, "train", out_root)
    copy_split(food101_root, "test", out_root)

    print("📂 Copy FruitVeg81...")
    copy_split(fruitveg_root, "train", out_root)
    copy_split(fruitveg_root, "test", out_root)

    # 6) Copy UEC (mapping id->label and skipping duplicates)
    print("📂 Copy UEC (skip duplicates vs Food101)...")
    copy_split_uec(uec_root, "train", out_root, id_to_name, skip_norm, normalize)
    copy_split_uec(uec_root, "test",  out_root, id_to_name, skip_norm, normalize)

    print(f"✅ Done. Merged dataset at: {out_root.resolve()}")
    
# merge_food_datasets(
#     food101_dir="dataset/food101",
#     fruitveg81_dir="dataset/fruitveg81",
#     uec_dir="dataset/uecfood256",
#     uec_labels_txt="dataset/uecfood256/uecfood255_categories.txt",  # fichier id\tname
#     output_dir="dataset/merged_dataset",
#     normalize=normalize,
#     threshold=0.80,  # fuzzy
# )

import os

def rename_subfolders(base_folder: str):
    """
    Renomme tous les sous-dossiers d'un dossier donné en remplaçant
    les espaces et tirets par des underscores.

    Exemple :
        "apple pie"  →  "apple_pie"
        "chicken-rice"  →  "chicken_rice"
    """
    # Vérifie si le dossier existe
    if not os.path.exists(base_folder):
        print(f"❌ Le dossier {base_folder} n'existe pas.")
        return

    print(f"📁 Renommage des sous-dossiers dans : {base_folder}\n")

    # On parcourt les sous-dossiers du répertoire
    print(len(os.listdir(base_folder)))
    for item in os.listdir(base_folder):
        old_path = os.path.join(base_folder, item)

        if os.path.isdir(old_path):
            # Nouveau nom : remplace espaces et tirets par "_"
            new_name = item.replace(" ", "_").replace("-", "_")
            new_path = os.path.join(base_folder, new_name)

            # Renommer si différent
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"✅ {item}  →  {new_name}")

    print("\n🎉 Renommage terminé avec succès.")
    
import os
import csv
import re
from pathlib import Path

import re

def clean_label(label: str) -> str:
    """
    Clean and normalize class labels to be consistent and filesystem-safe.
    Example:
        "Chicken 'n' Egg on Rice" -> "chicken_n_egg_on_rice"
        "Fish & Chips" -> "fish_and_chips"
        "apple-pie," -> "apple_pie"
    """
    if not isinstance(label, str):
        return ""

    # Convert to lowercase
    label = label.lower().strip()

    # Replace '&' and 'n'' with 'and' and 'n'
    label = label.replace("&", "and")
    label = label.replace("'n'", "n")
    label = label.replace("'n", "n")
    label = label.replace("n'", "n")

    # Replace hyphens and spaces by underscores
    label = re.sub(r"[-\s]+", "_", label)

    # Remove punctuation or special characters except underscores
    label = re.sub(r"[^a-z0-9_]", "", label)

    # Replace multiple underscores with a single one
    label = re.sub(r"_+", "_", label)

    # Remove leading/trailing underscores
    label = label.strip("_")

    return label


def rename_uecfood256_folders(dataset_dir: str, txt_file: str):
    """
    Rename UECFood256 subfolders (1, 2, 3...) using the class names from the .txt file.

    Args:
        dataset_dir (str): Path to the dataset folder containing "train" and "test".
        txt_file (str): Path to the .txt file containing 'id' and 'name' columns.

    Example:
        rename_uecfood256_folders(
            dataset_dir="/path/to/UECFood256",
            txt_file="/path/to/UECFood256/UECFOOD256.txt"
        )
    """

    # --- Step 1: Read class mapping ---
    print("📂 Reading class mapping from:", txt_file)
    with open(txt_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = [row for row in reader if len(row) >= 2]

    id_to_name = {}
    for row in rows[1:]:
        try:
            id_ = int(row[0].strip())
            name = row[1].strip().lower()
            # normalize name (replace spaces, dashes, and quotes)
            name = clean_label(name)
            id_to_name[str(id_)] = name
        except Exception as e:
            print(f"⚠️ Error parsing row {row}: {e}")

    print(f"✅ Loaded {len(id_to_name)} class names.")

    # --- Step 2: Loop over train and test ---
    for split in ["train", "test"]:
        split_path = Path(dataset_dir) / split
        if not split_path.exists():
            print(f"⚠️ {split_path} does not exist, skipping.")
            continue

        print(f"\n🔄 Renaming folders in: {split_path}")

        for folder in split_path.iterdir():
            if folder.is_dir():
                old_name = folder.name
                # new_name = clean_label(old_name)
                # new_path = folder.parent / new_name
                # try:
                #         os.rename(folder, new_path)
                #         print(f"✅ {old_name} → {new_name}")
                # except Exception as e:
                #         print(f"❌ Failed to rename {old_name}: {e}")
                # else:
                #     print(f"⚠️ No match found for folder {old_name}")
                if new_name := id_to_name.get(old_name):
                    new_path = folder.parent / new_name
                    try:
                        os.rename(folder, new_path)
                        print(f"✅ {old_name} → {new_name}")
                    except Exception as e:
                        print(f"❌ Failed to rename {old_name}: {e}")
                else:
                    print(f"⚠️ No match found for folder {old_name}")

    print("\n🎉 Folder renaming complete!")
    

# rename_uecfood256_folders(
#     dataset_dir="dataset/not_merged/uecfood256",
#     txt_file="dataset/not_merged/uecfood256/uecfood255_categories.txt"
# )


    
# rename_subfolders("dataset/merged_dataset/test")


# results = compare_food_labels(food101_labels, uecfood256_labels, normalize)
# print(results)
