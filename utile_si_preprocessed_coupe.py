import os
import shutil

def get_midi_files_from_index(directory_path, start_index):
    """
    Récupère une liste de fichiers MIDI d'un répertoire à partir d'un index donné.

    Args:
        directory_path (str): Le chemin du répertoire contenant les fichiers MIDI.
        start_index (int): L'index de départ (basé sur 0) pour la liste de fichiers.

    Returns:
        list: Une liste de chemins d'accès complets de fichiers MIDI à partir de l'index donné.
              Renvoie une liste vide si l'index est hors de portée.
    """
    midi_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                full_path = os.path.join(root, file)
                midi_files.append(full_path)
    
    midi_files.sort()
    
    if 0 <= start_index < len(midi_files):
        return midi_files[start_index:]
    else:
        print(f"Avertissement : L'index de départ ({start_index}) est hors de portée. "
              f"Le nombre total de fichiers est {len(midi_files)}.")
        return []

def preprocess_and_save_files(files_to_process, output_directory):
    """
    Traite une liste de fichiers et les sauvegarde dans un nouveau répertoire.

    Args:
        files_to_process (list): Une liste de chemins de fichiers à traiter.
        output_directory (str): Le chemin du répertoire de sortie pour les fichiers traités.
    """
    if not files_to_process:
        print("Aucun fichier à traiter.")
        return

    # S'assurer que le répertoire de sortie existe
    os.makedirs(output_directory, exist_ok=True)
    
    for file_path in files_to_process:
        try:
            # Ici, vous ajouteriez votre logique de prétraitement réelle.
            # Pour cet exemple, nous allons simplement simuler un traitement et une copie.
            print(f"Traitement du fichier : {file_path}")
            
            # Récupérer le nom du fichier pour la sauvegarde
            filename = os.path.basename(file_path)
            output_file_path = os.path.join(output_directory, filename)
            
            # Simuler la sauvegarde du fichier traité
            shutil.copy(file_path, output_file_path)
            print(f"Fichier sauvegardé dans : {output_file_path}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de {file_path}: {e}")
            continue

# --- Exemple d'utilisation ---
if __name__ == '__main__':
    # Définir le chemin de votre répertoire de fichiers MIDI
    midi_directory = 'C:/Users/vivo-/Documents/Dauphine/StageM1/GitHub_repo/dataset/maestro_transposed'
    
    # Définir le chemin du répertoire où les fichiers traités seront enregistrés
    output_directory = 'C:/Users/vivo-/Documents/Dauphine/StageM1/GitHub_repo/dataset/preprocessed_maestro_transposed_new'
    
    # Définir l'indice à partir duquel vous voulez commencer
    start_index = 22562

    # Obtenir la liste des fichiers à traiter
    files_to_process = get_midi_files_from_index(midi_directory, start_index)
    
    # Prétraiter et sauvegarder les fichiers
    preprocess_and_save_files(files_to_process, output_directory)