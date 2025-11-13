import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset
from PIL import Image
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FoodClassifier:
    def __init__(self, model_dir: str):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.model_cv = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.id2label = self.model_cv.config.id2label

    def predict_food_name(self, image_source):
        """Image classification from different input types."""
        if isinstance(image_source, Image.Image):
            # Cas direct : image déjà chargée (comme dans Hugging Face)
            image = image_source.convert("RGB")

        elif isinstance(image_source, str) or (
            not hasattr(image_source, "file") and isinstance(image_source, BytesIO)
        ):
            image = Image.open(image_source).convert("RGB")

        elif hasattr(image_source, "file"):  # UploadFile (FastAPI)
            image = Image.open(BytesIO(image_source.file.read())).convert("RGB")

        else:
            raise ValueError(
                f"Unsupported input type: {type(image_source)}. "
                "Expected str, PIL.Image.Image, UploadFile, or BytesIO."
            )

        # Preprocessing + predictiion
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.model_cv(**inputs).logits
            pred_id = logits.argmax(-1).item()
        return self.id2label[pred_id]


    def evaluate_on_dataset(self, test_dir: str, limit: int = None):
        """Evaluate the modele on a image folder (imagefolder)"""
        print("🧩 Test dataset Loading...")
        ds_test = load_dataset("imagefolder", data_dir=test_dir, split="train")
        if limit:
            ds_test = ds_test.shuffle(seed=42).select(range(limit))
        labels = ds_test.features["label"].names

        y_true = []
        y_pred = []

        print(f"🚀 Launching prediction on  {len(ds_test)} images...\n")
        for example in tqdm(ds_test, total=len(ds_test)):
            image = example["image"]
            label = labels[example["label"]]

            # Predict
            pred = self.predict_food_name(image)
            y_true.append(label)
            y_pred.append(pred)

        # Compute metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        print("\n📊 Results :")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-macro : {f1:.4f}")
        return {"accuracy": acc, "f1_macro": f1}

if __name__ == "__main__":
    # model_dir = "nateraw/food"  
    model_dir = "model_saved/vit-food-224/checkpoint-8058"
    test_dir = "dataset/not_merged/uecfood256/test"

    classifier = FoodClassifier(model_dir)
    results = classifier.evaluate_on_dataset(test_dir)  # ou None pour tout le dataset

    # Sauvegarde facultative
    import json, os
    results_path = f"results/{model_dir.split("/")[1]}"
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, f"{test_dir.split("/")[2]}_test_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
