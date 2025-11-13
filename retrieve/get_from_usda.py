from dotenv import load_dotenv
import json
import requests
import time
import pandas as pd
import os, sys
import os
import sys

# Ajouter le dossier parent au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve.labels import food101_labels, uecfood256_labels, fruitveg81_labels

load_dotenv()

USDA_KEY = os.getenv("USDA_KEY")
USDA_BASE_URL = os.getenv("USDA_BASE_URL")

def preprocess_class_label(label):
    """convert a label to common name format"""
    return label.replace("_", " ").title()

def search_food(query):
    """Search for a food item in the USDA database and return its FDC ID."""
    url = f"{USDA_BASE_URL}/foods/search"
    params = {"api_key": USDA_KEY, "query": query, "pageSize": 1}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("foods"):
            return data["foods"][0]["fdcId"]
    return None

def get_food_details(fdc_id):
    """Retrieve detailed information about a food item using its FDC ID."""
    url = f"{USDA_BASE_URL}/food/{fdc_id}"
    params = {"api_key": USDA_KEY}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        food = response.json()
        nutrients = []
        nutrients.extend(
            {
                "name": n["nutrient"].get("name"),
                "value": n.get("amount"),
                "unit": n["nutrient"].get("unitName"),
            }
            for n in food.get("foodNutrients", [])
            if "nutrient" in n
        )
        ingredients = (
            [item.strip() for item in food.get("ingredients").split(",")]
            if food.get("ingredients")
            else [item.strip() for f in food.get("inputFoods", []) for item in f.get("ingredientDescription", "").split(",") if item.strip()]
        )
        return {
            "fdc_id": fdc_id,
            "description": food.get("description"),
            "ingredients": ingredients,
            "nutrients": nutrients,
            "source_url": f"https://fdc.nal.usda.gov/food-details/{fdc_id}/nutrients"
        }
    return None


# ===============================================================
# --- Utility functions
# ===============================================================

def load_existing_foods(output_json_file):
    """Load already processed foods if the output file exists."""
    if os.path.exists(output_json_file):
        with open(output_json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_progress(enriched_foods, output_json_file):
    """Save progress incrementally to JSON file."""
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(enriched_foods, f, ensure_ascii=False, indent=2)


def create_food_entry(label, details):
    """Construct a standardized food entry dictionary."""
    return {
        "class": label,
        "food_name": details["description"],
        "description": details.get("description"),
        "ingredients": details.get("ingredients"),
        "nutrients": details["nutrients"],
        "source_url": details["source_url"]
    }


def retrieve_food_details(query):
    """Retrieve food details from USDA API given a query."""
    fdc_id = search_food(query)
    if not fdc_id:
        print(f"Result not found: {query}")
        return None

    details = get_food_details(fdc_id)
    if not details:
        print(f"Impossible to retrieve: {query}")
        return None

    return details


# ===============================================================
# --- Main function
# ===============================================================

def process_retrieving(labels, output_json_file, is_uecfood256=False):
    """
    Process a list of food items to retrieve their details from USDA.

    Args:
        labels (list): List of food labels or (id, name) pairs.
        output_json_file (str): Path to save JSON results.
        is_uecfood256 (bool): Whether the dataset follows UECFood256 structure.
    """
    enriched_foods = load_existing_foods(output_json_file)
    done_classes = {food["class"] for food in enriched_foods}

    print(f"Starting retrieval for {len(labels)} items...")

    for i, item in enumerate(labels, 1):
        # --- Extract label based on dataset type ---
        if is_uecfood256:
            fid, name = item
            label = name
            if fid in done_classes:
                continue
        else:
            label = item
            if label in done_classes:
                continue

        # --- Process one label ---
        query = preprocess_class_label(label)
        print(f"[{i}/{len(labels)}] 🔍 {query}")

        details = retrieve_food_details(query)
        if not details:
            continue

        # --- Save entry ---
        enriched_foods.append(create_food_entry(label, details))
        print(f"{details['description']} added.")

        save_progress(enriched_foods, output_json_file)
        time.sleep(0.5)

    print(f"\n{len(enriched_foods)} food items saved in {output_json_file}")


# Start processing here
if __name__ == "__main__":
    # process food 101
    output_food101 = "json/old/food101_usda_enriched.json"
    # process_retrieving(output_json_file=output_food101, labels=food101_labels)
    
    
    # process uecfood256
    output_uecfood256 = "json/old/uecfood256_usda_enriched.json"
    # process_retrieving(output_json_file=output_uecfood256, labels=uecfood256_labels, is_uecfood256=True)
    
    # process fruitveg81
    output_fruitveg81 = "json/old/fruitveg81_usda_enriched.json"
    process_retrieving(output_json_file=output_fruitveg81, labels=fruitveg81_labels)
    