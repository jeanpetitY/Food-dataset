from huggingface_hub import login
from dotenv import load_dotenv
import os
from transformers import AutoModelForImageClassification, AutoImageProcessor

load_dotenv()

token = os.getenv("HUB_TOKEN")

model_path = "model_saved/vit-food-224/checkpoint-8058"
model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

login(token)


model.push_to_hub("yvelos/vit-food-384")
processor.push_to_hub("yvelos/vit-food-384")