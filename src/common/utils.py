import os
from pathlib import Path

def get_projet_parent_dir(target_name="examen-dvc"):
    '''
    retrouve la racide du projet
    '''
    current_dir = Path.cwd()  # Démarre à partir du répertoire actuel
    print(current_dir, "/", current_dir.name)
    while current_dir.name != target_name:
        print(current_dir)
        if current_dir.parent == current_dir:  # Vérifie si on est à la racine
            raise FileNotFoundError(f"Dossier '{target_name}' non trouvé.")
        current_dir = current_dir.parent  # Remonte d'un niveau
    print(current_dir, "/", current_dir.name)
    return current_dir

if __name__ == '__main__':
    get_projet_parent_dir()
    