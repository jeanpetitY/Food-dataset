import os
from PIL import Image
import numpy as np

def analyze_food_dataset(dataset_path, dataset_name, continent=None, license=None):
    """
    Analyse basique d’un dataset d’images alimentaires.
    Renvoie un dictionnaire avec des statistiques clés.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp")

    # ✅ Vérifier que le dossier existe
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # ✅ Trouver les sous-dossiers (chaque dossier = classe)
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not class_dirs:
        print(f"⚠️ No class subdirectories found in {dataset_path}. Trying flat structure...")
        class_dirs = [""]
    
    num_classes = len(class_dirs)
    total_images = 0

    # ✅ Initialisation correcte ici
    widths = []
    heights = []

    for cls in class_dirs:
        cls_path = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(image_extensions)]
        total_images += len(images)
        
        # Échantillonner jusqu’à 10 images par classe
        for img_file in images[:10]:
            img_path = os.path.join(cls_path, img_file)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"⚠️ Error reading {img_file}: {e}")
                continue
    
    mean_per_class = total_images / num_classes if num_classes > 0 else 0
    mean_width = np.mean(widths) if widths else 0
    mean_height = np.mean(heights) if heights else 0
    
    return {
        "Dataset": dataset_name,
        "License": license,
        "Continent": continent,
        "Classes": num_classes,
        "Total Images": total_images,
        "Mean/Class": round(mean_per_class, 2),
        "Mean Width": round(mean_width, 1),
        "Mean Height": round(mean_height, 1)
    }
