from orkg import ORKG
import os
from dotenv import load_dotenv
from typing import Literal
import json

DATA_TYPE = Literal["xsd:string", "xsd:int", "xsd:float", "xsd:url", "resource"]

load_dotenv()


class ImportORKGContribution:
    def __init__(self, email, password, host="https://sandbox.orkg.org"):
        self.orkg = ORKG(host=host, creds=(email, password))
        
    def get_host(self):
        return self.orkg.host
    
    def get_template(self, template_id):
        """ Get an ORKG template by its ID.
        Args:
            template_id (str): The ID of the template to retrieve.
        Returns:
            dict: The template data."""
        template = self.orkg.templates
    
    def check_property(self, pid):
        """ Check if an ORKG property (predicate) exists by its ID.
        Args:
            pid (str): The ID of the property to check.
        Returns:
            bool: True if the property exists, False otherwise."""
        return self.orkg.predicates.exists(id=pid)
    
    def get_property(self, label=None, pid=None, exact=True):
        """Retrieve an ORKG property (predicate) by its label.
        Args:
            label (str): The label of the property to search for.
            exact (bool): Whether to perform an exact match search. Default is True.
        Returns:
            str or None: The ID of the found property, or None if not found."""
        if pid:
            return pid
        if not label:
            print("You must provide either a label or a pid to get the property.")
            return None
        response = self.orkg.predicates.get_unpaginated(
            q=label, exact=exact, size=30, sort='label', desc=True
        )
        if not response.all_succeeded:
            print(f"Error: {response.content}")
            return None
        
        if isinstance(response.content, list) and len(response.content) > 0:
            prop_id = response.content[0]["id"]
            prop_label = response.content[0]["label"]
            # print(f"Property found : {prop_label} ({prop_id})")
            return prop_id
        else:
            return None
    
    def get_resource(self, label=None, exact=True, classes: list = None, res_id=None):
        """ Get an ORKG resource by its label.
        Args:
            label (str): The label of the resource to search for.
            exact (bool): Whether to perform an exact match search. Default is True.
        Returns:
            str or None: The ID of the found resource, or None if not found."""
        if res_id:
            return res_id
        if not label:
            print("You must provide a label to get the resource.")
            return None
        classes = classes or ""
        response = self.orkg.resources.get_unpaginated(
            q=label, exact=exact, size=30, sort='label', desc=True, include=classes
        )
        if not response.all_succeeded:
            print(f"Error: {response.content}")
            return None
        
        if isinstance(response.content, list) and len(response.content) > 0:
            resource_id = response.content[0]["id"]
            resource_label = response.content[0]["label"]
            # print(f"Resource found : {resource_label} ({resource_id})")
            return resource_id
        else:
            return None
        
        
    def update_resource(self, resource_id, classes: list = []):
        """Update an ORKG resource with new classes.
        Args:
            resource_id (str): The ID of the resource to update.
            classes (list): List of class IDs to assign to the resource.
        Returns:
            dict: The response from the ORKG API."""
        response = self.orkg.resources.update(id=resource_id, classes=classes)
        return response
    
    def create_litteral(self, label, data_type: DATA_TYPE = "xsd:string", classes: list = []):
        """
        Create an ORKG literal or resource based on the data type.
        if data_type is resource, it will create a resource with optional classes.
        Args:
            label (str): The label of the literal or resource.
            data_type (DATA_TYPE): The data type of the literal. Use "resource" to create a resource.
            classes (list): List of class IDs to assign to the resource if data_type is "resource".
        """
        
        if data_type == "resource":
            resource_id = self.get_resource(label=label, classes=classes[0] if classes else "")
            if resource_id:
                return resource_id
            resource_id = self.create_resource(label=label, classes=classes)
            return resource_id
        response = self.orkg.literals.add(label=label, datatype=data_type)
        if response.succeeded and isinstance(response.content, dict):
            literal_id = response.content.get("id")
            # print(f"Literal created : {label} ({literal_id})")
            return literal_id
        else:
            print(f"Error during predicate creation '{label}': {response.content}")
            return None
        
    
    def create_resource(self, label, classes: list = []):
        """Create an ORKG resource if it does not already exist.
        Args:
            label (str): The label of the resource to create.
            classes (list): List of class IDs to assign to the resource.
        Returns:
            str or None: The ID of the created resource, or None if creation failed.
        """
        if not classes:
            response = self.orkg.resources.add(label=label)
        elif classes:
            response = self.orkg.resources.add(label=label, classes=classes)
            # response = self.update_resource(resource_id=resource_id, classes=classes)
            
        
        if response.succeeded and isinstance(response.content, dict):
            resource_id = response.content.get("id")
            # print(f"Resource created : {label} ({resource_id})")
            return resource_id
        else:
            print(f"Error during predicate creation '{label}': {response.content}")
            return None
        
    def create_property(self, label):
        """Create an ORKG property (predicate) if it does not already exist.
        Args:
            label (str): The label of the property to create.
        Returns:
            str or None: The ID of the created property, or None if creation failed."""
        response = self.orkg.predicates.add(label=label)
        if response.succeeded and isinstance(response.content, dict):
            prop_id = response.content.get("id")
            # print(f"property created : {label} ({prop_id})")
            return prop_id
        else:
            print(f"Error during predicate creation '{label}': {response.content}")
            return None
        
    def delete_resource(self, resource_id):
        """Delete an ORKG resource by its ID.
        Args:
            resource_id (str): The ID of the resource to delete.
        Returns:
            None"""
        self.orkg.resources.delete(id=resource_id)
        print(f"resource {resource_id} was deleted successfully")
        
        
        
    def create_resource_statement(
        self, 
        literal_label, 
        resource_label=None, 
        prop_label=None, 
        is_simple_statement: bool = True, 
        resource_classes=[], 
        literal_classes=[],
        data_type="resource",
        pid=None,
        res_id=None
    ):
        """Create a statement linking a subject, property, and object(can be a resource or literal) in ORKG.
        In case the literal is a resource, it will be important to differentiate classes for object(literal) and subject(resource) in the statement.
        Args:
            resource_label (str): The label of the subject resource.
            prop_label (str): The label of the property (predicate).
            literal_label (str): The label of the object literal or resource.
            is_simple_statement (bool): Whether the statement is simple (literal) or complex (resource). Default is True.
            resource_classes (list): List of class IDs to assign to the subject resource.
            literal_classes (list): List of class IDs to assign to the object resource if is_simple_statement is False.
            data_type (DATA_TYPE): The data type of the literal. Use "resource" to create a resource.
        Returns:
            str or None: The ID of the created statement, or None if creation failed.
        """
        resource_id = self.get_resource(res_id=res_id, label=resource_label, classes=resource_classes[0] if resource_classes else "")
        if not resource_id:
            resource_id = self.create_resource(label=resource_label, classes=resource_classes)
        prop_id = self.get_property(label=prop_label, pid=pid)
        if not prop_id:
            prop_id = self.create_property(prop_label)
        if is_simple_statement:
            literal_id = self.create_litteral(label=literal_label, data_type=data_type)
        else:
            literal_id = self.create_litteral(label=literal_label, data_type=data_type, classes=literal_classes)
        if not resource_id or not prop_id or not literal_id:
            print("Error: Unable to create resource, property, or literal.")
            return None
        
        response = self.orkg.statements.add(subject_id=resource_id, predicate_id=prop_id, object_id=literal_id)
        
        if response.succeeded and isinstance(response.content, dict):
            statement_id = response.content.get("id")
            # print(f"statement created for subject={resource_label} prop={prop_label} object=({literal_label})")
            return statement_id
        else:
            print(f"Error during predicate creation {response.content}")
            return None
        
        
class USDA_Importer:
    def __init__(self, orkg_importer: ImportORKGContribution):
        self.orkg_importer = orkg_importer
        self.usda_data = []
    
    def add_food_component_statement(
        self, 
        resource_label, 
        prop_label, 
        literal_label, 
        is_simple_statement: bool = False, 
        resource_classes=["C124011"], 
        literal_classes=["C34009"],
        data_type="resource"
    ):
        """ Add a food component statement to ORKG using the ImportORKGContribution instance."""
        statement_id = self.orkg_importer.create_resource_statement(
            resource_label=resource_label,
            prop_label=prop_label,
            literal_label=literal_label,
            is_simple_statement=is_simple_statement,
            resource_classes=resource_classes,
            literal_classes=literal_classes,
            data_type=data_type
        )
        return statement_id
    
    def add_simple_statement_to_food_component_template(
        self, 
        resource_label, 
        prop_label, 
        literal_label, 
        is_simple_statement: bool = True, 
        resource_classes=["C34009"],
        data_type="xsd:string"
        ):
        """ Add a statement to a food component template in ORKG using the ImportORKGContribution instance."""
        statement_id = self.orkg_importer.create_resource_statement(
            resource_label=resource_label,
            prop_label=prop_label,
            literal_label=literal_label,
            is_simple_statement=is_simple_statement,
            resource_classes=resource_classes,
            data_type=data_type
        )
        return statement_id
    
    def add_statement_to_USDA_resource_dataset(
        self, 
        literal_label="apple_pie_food_101", 
        prop_label="has contribution", 
        res_id="R2129182", 
        is_simple_statement: bool = False, 
        resource_classes=["Dataset"], 
        literal_classes=["C124011"],
        data_type="resource"
    ):
        """ Add a statement to link USDA food resource to USDA dataset resource in ORKG using the ImportORKGContribution instance."""
        statement_id = self.orkg_importer.create_resource_statement(
            res_id=res_id,
            prop_label=prop_label,
            literal_label=literal_label,
            is_simple_statement=is_simple_statement,
            resource_classes=resource_classes,
            literal_classes=literal_classes,
            data_type=data_type
        )
        return statement_id
    
    def open_usda_food_component_data(self, file_path: str) -> dict:
        """ Open and read USDA food component data from a JSON file.
        Args:
            file_path (str): The path to the JSON file containing USDA food component data.
        Returns:
            dict: A dictionary containing the USDA food component data."""
        with open(file_path, 'r', encoding='utf-8') as file:
            usda_data = json.load(file)
            self.usda_data = usda_data
        return self.usda_data
    
    def process_usda_food_component_data(self, dataset_name="UECFOOD256"):
        """Process USDA food component data and add it to ORKG using the ImportORKGContribution instance."""
        usda_data = self.usda_data
        
        for item in usda_data:
            # === 1️⃣ Create the main resource for the food ===
            resource_label = f"{item['food_name'].lower().replace(' ', '_')}_{dataset_name.lower()}"
            print(f"\n=======Processing food item: {resource_label}===================")

            # Main class : "USDA Food" (C34009)
            food_id = self.orkg_importer.create_resource(
                label=resource_label, 
                classes=["C124011"]  # Classe for USDA Food
            )

            # === 2️⃣ add simple attribut of the food ===
            print("Adding simple attributes...(name, ingredients, description, source link)")

            # Food name
            self.orkg_importer.create_resource_statement(
                resource_label=resource_label,
                resource_classes=["C124011"],
                literal_label=item["food_name"],
                prop_label="usda food name",
                data_type="xsd:string"
            )

            # Description
            if item.get("description"):
                self.orkg_importer.create_resource_statement(
                    resource_label=resource_label,
                    resource_classes=["C124011"],
                    literal_label=item["description"],
                    prop_label="description",
                    data_type="xsd:string"
                )

            # Ingrédients (liste: Join text)
            if item.get("ingredients"):
                ingredients_field = item["ingredients"]

                # Vérifier le type avant de joindre
                if isinstance(ingredients_field, list):
                    ingredients_str = ", ".join([ing.strip() for ing in ingredients_field if isinstance(ing, str) and ing.strip()])
                elif isinstance(ingredients_field, str):
                    ingredients_str = ingredients_field.strip()
                else:
                    ingredients_str = ""
                self.orkg_importer.create_resource_statement(
                    resource_label=resource_label,
                    resource_classes=["C124011"],
                    literal_label=ingredients_str,
                    prop_label="usda food ingredient",
                    data_type="xsd:string"
                )

            # Source link
            if item.get("source_url"):
                self.orkg_importer.create_resource_statement(
                    resource_label=resource_label,
                    resource_classes=["C124011"],
                    prop_label="usda source link",
                    literal_label=item["source_url"],
                    data_type="xsd:uri"
                )

            # === 3️⃣ Add nutrient as sub resources ===
            print("Adding nutrients and his values as sub-resources...")
            if item.get("nutrients"):
                for nutrient in item["nutrients"]:
                    nutrient_name = nutrient["name"]
                    value = nutrient["value"]
                    unit = nutrient["unit"]
                    value_unit = f"{value} {unit}".strip()

                    # create a resource for the nutrient
                    nutrient_res_label = f"{nutrient_name.lower().replace(' ', '_')}_{resource_label}"

                    # nutrient_res_id = self.orkg_importer.create_resource(
                    #     label=nutrient_res_label,
                    #     classes=["C124011"]  # Class of the nutrient/ Food Component
                    # )

                    # Link food component to main resource
                    self.add_food_component_statement(
                        resource_label=resource_label,
                        prop_label="usda food component",
                        literal_label=nutrient_res_label,
                        is_simple_statement=False,
                        # resource_classes=["C34009"],
                        # literal_classes=["C124011"],
                        # data_type="resource"
                    )

                    # add internal properties to the food component resource
                    self.add_simple_statement_to_food_component_template(
                        resource_label=nutrient_res_label,
                        prop_label="food component name",
                        literal_label=nutrient_name,
                        data_type="xsd:string"
                    )

                    self.add_simple_statement_to_food_component_template(
                        resource_label=nutrient_res_label,
                        prop_label="food component value",
                        literal_label=str(value_unit),
                        data_type="xsd:string"
                    )
            
            # add resources as contribution to USDA dataset resource
            print("Linking food resource to USDA dataset resource...")
            self.add_statement_to_USDA_resource_dataset(
                literal_label=resource_label,
            )

            print(f"✅ Finished processing {resource_label}")

            
        
        
    
    # def add_statement_valute_to_food_component_template(
    #     self, 
    #     resource_label, 
    #     literal_label, 
    #     pid="P5086", # value property to food component
    #     is_simple_statement: bool = False, 
    #     resource_classes=["C34009"], # Food component classe
    #     literal_classes=["C23008"],  # Quantity value classe( numeric value with unit)
    #     data_type="resource",
    #     # pids=["P45075", "P45076"] # P45075 numeric value, P45076 unit
    #     ):
    #     """ Add a statement to a food component template in ORKG using the ImportORKGContribution instance.
    #     value to food component consit to add his numeric value and his unit
    #     """
    #     statement_id = self.orkg_importer.create_resource_statement(
    #         resource_label=resource_label,
    #         pid=pid,
    #         literal_label=literal_label,
    #         is_simple_statement=is_simple_statement,
    #         literal_classes=literal_classes,
    #         resource_classes=resource_classes,
    #         data_type=data_type
    #     )
    #     return statement_id
        
        
    

# classe
#  food component C34009
# USDA Food C124011
#  value property P5086

EMAIL=os.getenv("EMAIL")
PASSWORD=os.getenv("PASSWORD")

orkg_importer = ImportORKGContribution(email=EMAIL, password=PASSWORD)
usda_importer = USDA_Importer(orkg_importer=orkg_importer)
# print(orkg_importer.get_host())
# print(orkg_importer.check_property("P12345"))
# property_id = orkg_importer.get_property("has contribution")

# statement_id = usda_importer.add_simple_statement_to_food_component_template(
#     resource_label="sodium_baby_back_ribs",
#     prop_label="food component value",
#     literal_label="19 mg",
#     is_simple_statement=True,
#     data_type="xsd:string"
# )
# print(statement_id)

data = usda_importer.open_usda_food_component_data(file_path="data.json")

statement_id = usda_importer.process_usda_food_component_data(dataset_name="food101")

# for i in range(2129224, 2129230):
#     resource_id = f"R{i}"
#     orkg_importer.delete_resource(resource_id)
#     print(f"Deleting resource: {resource_id}")


# statement_id = usda_importer.add_statement_valute_to_food_component_template(
#     resource_label="sodium_baby_back_ribs",
#     literal_label="Quantity value"
# )
# print(statement_id)

# resource_id = orkg_importer.get_resource(
#     label="Sodium_apple_pie",
#     classes="C34009"
# )
# print(resource_id)

# orkg_importer.delete_resource("R2129231")

# resource_id = orkg_importer.create_resource(
#     "juglar",
#     classes=["C124011"]
# )
# print(resource_id)
# print("founded property ID:", property_id)