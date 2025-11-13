import os
import shutil
from pathlib import Path


def count_files_in_directory(directory_path):
    """
    Compte le nombre de fichiers dans un répertoire donné.
    Renvoie le nombre total de fichiers.
    """
    # Fist check if folder if exist
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory path not found: {directory_path}")

    return sum(len(files) for root, dirs, files in os.walk(directory_path))


def list_subdirectories(directory_path):
    """
    Liste tous les sous-répertoires dans un répertoire donné.
    Renvoie une liste des noms de sous-répertoires.
    """
    # Fist check if folder if exist
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory path not found: {directory_path}")

    return [
        d
        for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]

def get_total_file_count(directory_path):
    
    subdirs_list = list_subdirectories(directory_path)
    if subdirs_list:
        total_count = 0
        for subdir in subdirs_list:
            subdir_path = os.path.join(directory_path, subdir)
            total_count += count_files_in_directory(subdir_path)
        return total_count
    else: 
        raise FileExistsError("No subdirectories found, counting files in the main directory.") 
    
def flatten_image_folders(root_dir, extensions=(".jpg", ".jpeg", ".png")):
    """
    Flatten all nested subfolders inside each class folder (e.g. apples, bananas)
    so that all images end up directly in their class folder.

    Args:
        root_dir (str): Path to the dataset root (e.g. 'fruitveg81/')
        extensions (tuple): Allowed image extensions
    """
    root = Path(root_dir)
    class_dirs = [d for d in root.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        print(f"Processing class: {class_dir.name}")
        
        # Parcourt récursivement tous les sous-dossiers
        for subpath in class_dir.rglob("*"):
            if subpath.is_file() and subpath.suffix.lower() in extensions:
                # Nouveau nom unique pour éviter les collisions
                new_name = f"{subpath.stem}_{hash(subpath)}{subpath.suffix}"
                dest_path = class_dir / new_name

                try:
                    shutil.move(str(subpath), str(dest_path))
                except Exception as e:
                    print(f"⚠️ Could not move {subpath}: {e}")

        # Supprime les anciens sous-dossiers vides
        for subfolder in class_dir.rglob("*"):
            if subfolder.is_dir() and not any(subfolder.iterdir()):
                subfolder.rmdir()

        print(f"✅ Flattened: {class_dir.name}\n")

    print("🎉 All folders have been flattened successfully!")


directory_path = "data/UECFOOD256"

# total_file = get_total_file_count(directory_path)
# print(f"Total files in '{directory_path}': {total_file}")


directory_path = "data/vegfru/fru92_images/almond"
flatten_image_folders("dataset/fruitveg-81/fruitveg81")


# list_file = os.listdir(directory_path)
# print(len(list_file))