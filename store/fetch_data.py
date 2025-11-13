from .config.sparql_client import execute_sparql_query
from collections import defaultdict
import re

def is_image_url_string(url):
    # check if a url is an image path
    return re.search(r'\.(png|jpg|jpeg|gif|bmp|webp|tiff|svg)(\?.*)?$', url, re.IGNORECASE) 

def get_food_with_components():
    """Retrieve food list from ORKG including components, nutritional values, and units."""
    sparql_query = """
    PREFIX orkgr: <http://orkg.org/orkg/resource/>
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>

    SELECT DISTINCT ?id_food ?food_name ?food_component_name ?food_component_numeric_value ?food_component_unit
    WHERE {
        ?id_food rdf:type orkgc:C34019 .
        ?id_food rdfs:label ?food_name .

        ?id_food orkgp:P62073 ?food_component_id .
        ?food_component_id orkgp:P62093 ?food_component_name_id .
        ?food_component_name_id rdfs:label ?food_component_name .
        ?food_component_id orkgp:P5086 ?food_component_value_id .
        ?food_component_value_id orkgp:P45075 ?food_component_numeric_value .
        ?food_component_value_id orkgp:P45076 ?food_component_unit_id .
        ?food_component_unit_id rdfs:label ?food_component_unit .
    }
    """

    results = execute_sparql_query(sparql_query)

    food_dict = defaultdict(lambda: {"id": "", "food": "", "components": []})

    for item in results["results"]["bindings"]:
        food_id = item['id_food']['value']
        food_name = item['food_name']['value']

        component = {
            "name": item['food_component_name']['value'],
            "value": float(item['food_component_numeric_value']['value']),
            "unit": item['food_component_unit']['value']
        }
        food_dict[food_id]["food"] = food_name
        food_dict[food_id]["id"] = str(food_id)
        food_dict[food_id]["components"].append(component)

    return [{food_id: food_info} for food_id, food_info in food_dict.items()]

def get_food_with_ingredients():
    sparql_query = """
    PREFIX orkgr: <http://orkg.org/orkg/resource/>
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>

    SELECT DISTINCT  ?id_food ?food_ingredient_name
    WHERE {
        ?id_food rdf:type orkgc:C34019 .
        ?id_food rdfs:label ?food_name .

        ?id_food orkgp:P6003 ?food_ingredient_id .
        ?food_ingredient_id orkgp:P142024 ?food_ingredient_name_id .
        ?food_ingredient_name_id rdfs:label ?food_ingredient_name .
    }
    """
    results = execute_sparql_query(sparql_query)

    food_dict = defaultdict(lambda: {"ingredients": []})

    for item in results["results"]["bindings"]:
        food_id = item['id_food']['value']
        ingredients = item['food_ingredient_name']['value']

        food_dict[food_id]["ingredients"].append(ingredients)

    return [{food_id: food_info} for food_id, food_info in food_dict.items()]

def get_food_geography_area():
    sparql_query = """
    PREFIX orkgr: <http://orkg.org/orkg/resource/>
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>

    SELECT DISTINCT  ?id_food ?areas
    WHERE {
        ?id_food rdf:type orkgc:C34019 .
        ?id_food rdfs:label ?food_name .

        ?id_food orkgp:P135009 ?area_id .
        ?area_id rdfs:label ?areas .
    }
    """
    results = execute_sparql_query(sparql_query)

    food_dict = defaultdict(lambda: {"areas": []})

    for item in results["results"]["bindings"]:
        food_id = item['id_food']['value']
        ingredient = item['areas']['value']

        food_dict[food_id]["areas"].append(ingredient)

    return [{food_id: food_info} for food_id, food_info in food_dict.items()]


def get_local_name():
    sparql_query = """
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>

    SELECT DISTINCT  ?id_food ?local_name
    WHERE {
        ?id_food rdf:type orkgc:C34019 .
        ?id_food rdfs:label ?food_name .

        ?id_food orkgp:P142024 ?local_name_id .
        ?local_name_id rdfs:label ?local_name .
    }
    """
    results = execute_sparql_query(sparql_query)

    food_dict = defaultdict(lambda: {"food": []})

    for item in results["results"]["bindings"]:
        food_id = item['id_food']['value']
        local_name = item['local_name']['value']
        # local_name = local_name.split(',')

        food_dict[food_id]["food"].append(local_name)

    return [{food_id: food_info} for food_id, food_info in food_dict.items()]


def get_food_description():
    sparql_query = """
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>

    SELECT DISTINCT  ?id_food ?description
    WHERE {
        ?id_food rdf:type orkgc:C34019 .
        ?id_food rdfs:label ?food_name .

        ?id_food orkgp:P20098 ?description .
    }
    """
    results = execute_sparql_query(sparql_query)

    food_dict = defaultdict(lambda: {"description": ""})

    for item in results["results"]["bindings"]:
        food_id = item['id_food']['value']
        desc = item['description']['value']

        food_dict[food_id]["description"] = desc

    return [{food_id: food_info} for food_id, food_info in food_dict.items()]

def get_food_images():
    sparql_query = """
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>

    SELECT DISTINCT  ?id_food ?image
    WHERE {
        ?id_food rdf:type orkgc:C34019 .
        ?id_food rdfs:label ?food_name .

        ?id_food orkgp:P142026 ?image .
    }
    """
    results = execute_sparql_query(sparql_query)

    food_dict = defaultdict(lambda: {"image": []})

    for item in results["results"]["bindings"]:
        food_id = item['id_food']['value']
        image = item['image']['value']
        food_dict[food_id]["image"].append(image)

    return [{food_id: food_info} for food_id, food_info in food_dict.items()]

def merge_food_data():
    """Merge food components and ingredients into a single structured dataset."""
    
    food_components = get_food_with_components()
    food_ingredients = get_food_with_ingredients()
    areas = get_food_geography_area()
    local_names = get_local_name()
    descriptions = get_food_description()
    images = get_food_images()
    
    food_dict = defaultdict(lambda: {
        "id": "", "food": "", "description": "",
        "components": [], "ingredients": [],
        "areas": [], "image": []
    })

    for food_item in food_components:
        for food_id, data in food_item.items():
            food_dict[food_id]["id"] = data.get("id", "")
            food_dict[food_id]["components"] = data.get("components", [])

    for food_item in food_ingredients:
        for food_id, data in food_item.items():
            food_dict[food_id]["ingredients"] = data.get("ingredients", [])

    for food_item in areas:
        for food_id, data in food_item.items():
            food_dict[food_id]["areas"] = data.get("areas", [])

    for food_item in local_names:
        for food_id, data in food_item.items():
            food_dict[food_id]["food"] = data.get("food", "")

    for food_item in descriptions:
        for food_id, data in food_item.items():
            food_dict[food_id]["description"] = data.get("description", "")

    for food_item in images:
        for food_id, data in food_item.items():
            images = data.get("image", [])
            image = ''
            if isinstance(images, list):
                if images:
                    for item in images:
                        if is_image_url_string(item):
                            image = item
                            break
            elif is_image_url_string(item):
                image = item
            food_dict[food_id]["image"] = image

    # Fix missing IDs by setting them to the key
    for food_id, food_info in food_dict.items():
        if not food_info["id"]:
            food_info["id"] = food_id

    return [{food_id: food_info} for food_id, food_info in food_dict.items()]
