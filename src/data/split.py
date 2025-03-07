import pandas as pd
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split 
import os 

sys.path.append(str(Path(__file__).parent.parent / "common"))
import utils


def check_existing_folder(folder_path):
    """Check if a folder already exists. If it doesn't, ask if we want to create it."""
    if os.path.exists(folder_path) == False:
        while True:
            response = input(
                f"{os.path.basename(folder_path)} doesn't exists. Do you want to create it? (y/n): "
            )
            if response.lower() == "y":
                return True
            elif response.lower() == "n":
                return False
            else:
                print("Invalid response. Please enter 'y' or 'n'.")
    else:
        return False


def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if os.path.exists(output_folderpath) == False:
        os.makedirs(output_folderpath)


def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    create_folder_if_necessary(output_folderpath)
    print("output_folderpath", output_folderpath)
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        output_filepath = f'{output_folderpath}/{filename}.csv'
        file.to_csv(output_filepath, index=False)

def split():
    projet_dir = utils.get_projet_parent_dir()
    filename = Path(projet_dir / "data/raw_data/raw.csv")
    # charge le csv brut
    df = pd.read_csv(filename, sep=",")
    # split les données 
    X_train, X_test, y_train, y_test = train_test_split(df.drop(["silica_concentrate", "date"], axis=1), df.silica_concentrate, test_size=0.2, random_state=42)
    # sauve les données 
    # save_dataframes(X_train, X_test, y_train, y_test, Path(projet_dir / "data/processed_data"))
    save_dataframes(X_train, X_test, y_train, y_test, "data/processed_data")

if __name__ == '__main__':
    split()
