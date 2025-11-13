from orkg import ORKG
import json
from dotenv import load_dotenv
import os

load_dotenv()

EMAIL = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

host = "https://sandbox.orkg.org"

orkg = ORKG(host=host, creds=(EMAIL, PASSWORD))

FOOD_CLASS = "C124011"
COMPONENT_CLASS = "C34009"
OUTPUT_FILE = "exported_usda_food.json"


def extract_literal_value(value):
    """Retourne le label ou l'identifiant d'une valeur ORKG."""
    if isinstance(value, dict):
        return value.get("label") or value.get("id") or ""
    return str(value)


def parse_component_statements(component_id):
    """Récupère les informations d’un nutriment/component."""
    statements = orkg.statements.get_by_subject(component_id)
    component = {"name": "", "value": None, "unit": None}

    for s in statements.content:
        pred = s["predicate"]["label"]
        obj = s["object"]
        if pred == "food component name":
            component["name"] = extract_literal_value(obj)
        elif pred == "food component value":
            val = extract_literal_value(obj)
            try:
                parts = val.split()
                if len(parts) == 2:
                    component["value"] = float(parts[0])
                    component["unit"] = parts[1]
                else:
                    component["value"] = val
            except:
                component["value"] = val
    return component


def parse_food_resource(resource_id):
    """Reconstitue toutes les informations d’un aliment donné."""
    statements = orkg.statements.get_by_subject(resource_id)
    food_data = {
        "id": f"{host}/resource/{resource_id}",
        "food": [],
        "food_name": "",
        "description": "",
        "class": "",
        "components": [],
        "ingredients": [],
        "areas": [],
        "image": [],
        "usda_source": "",
        "dataset_name": "",
    }

    for s in statements.content:
        pred = s["predicate"]["label"]
        obj = s["object"]
        val = extract_literal_value(obj)

        if pred in ["usda food name", "food name"]:
            food_data["food"].append(val)
            food_data["food_name"] = val

        elif pred in ["food class name"]:
            food_data["class"] = val

        elif pred in ["description", "usda description"]:
            food_data["description"] = val

        elif pred in ["usda food ingredient", "ingredient"]:
            ings = [i.strip() for i in val.split(",") if i.strip()]
            food_data["ingredients"].extend(ings)

        elif pred in ["usda source link", "source", "link"]:
            food_data["usda_source"] = val

        elif pred in ["food image", "image", "photo"]:
            food_data["image"].append(val)

        elif pred in ["area", "origin", "region"]:
            food_data["areas"].append(val)

        elif pred == "usda food component":
            comp_id = obj["id"]
            comp = parse_component_statements(comp_id)
            if comp["name"]:
                food_data["components"].append(comp)

    return food_data


def fetch_and_stream_foods(limit=None):
    """Récupère les ressources et les écrit dans un vrai JSON (liste de dictionnaires)."""
    foods = orkg.resources.get_unpaginated(size=200, include=FOOD_CLASS)
    count = 0
    results = []

    # Charger si un fichier existe déjà (pour continuer l’écriture)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("[")  # Début de la liste JSON

        for r in foods.content:
            if FOOD_CLASS not in r.get("classes", []):
                continue

            rid = r["id"]
            try:
                data = parse_food_resource(rid)
                label = r.get("label", "")
                if "_" in label:
                    data["dataset_name"] = label.split("_")[-1]

                # Ajouter l’élément dans la liste
                results.append(data)

                # Écrire l’élément directement (avec virgule si pas premier)
                if count > 0:
                    f.write(",\n")
                f.write(json.dumps(data, ensure_ascii=False, indent=2))
                f.flush()

                count += 1
                print(f"✅ Saved {r['label']} ({rid}) [{count}]")

                if limit and count >= limit:
                    break

            except Exception as e:
                print(f"⚠️ Error processing {r.get('label', 'unknown')}: {e}")

        f.write("]")  # Fin du JSON

    print(f"\n✅ Export finished. {count} foods written to {OUTPUT_FILE}")


if __name__ == "__main__":
    print("🔍 Fetching USDA food data from ORKG incrementally...")
    fetch_and_stream_foods(limit=None)
