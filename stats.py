import os



def count_files_in_directory(directory_path):
    """
    Compte le nombre de fichiers dans un répertoire donné.
    Renvoie le nombre total de fichiers.
    """
    # Fist check if folder if exist
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory path not found: {directory_path}")
    
    total_files = 0
    for root, dirs, files in os.walk(directory_path):
        total_files += len(files)
    
    return total_files


def list_subdirectories(directory_path):
    """
    Liste tous les sous-répertoires dans un répertoire donné.
    Renvoie une liste des noms de sous-répertoires.
    """
    # Fist check if folder if exist
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory path not found: {directory_path}")
    
    subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
    return subdirs

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


directory_path = "data/UECFOOD256"

total_file = get_total_file_count(directory_path)
print(f"Total files in '{directory_path}': {total_file}")