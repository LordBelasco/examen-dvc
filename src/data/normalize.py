import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append(str(Path(__file__).parent.parent / "common"))
import utils

def save_dataframes(X_train, X_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test], ["X_train_scaled", "X_test_scaled"]):
        output_filepath = Path(output_folderpath / f'{filename}.csv')
        file.to_csv(output_filepath, index=False)

def normalize_X():
    projet_dir = utils.get_projet_parent_dir()
    X_train = pd.read_csv(Path(projet_dir / 'data/processed_data/X_train.csv'))
    X_test = pd.read_csv(Path(projet_dir / 'data/processed_data/X_test.csv'))

    numerical_cols = X_train.select_dtypes(include=["int", "float"]).columns
    
    scaler = MinMaxScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    projet_dir = utils.get_projet_parent_dir()
    save_dataframes(X_train, X_test, Path(projet_dir / "data/processed_data"))

if __name__ == '__main__':
    normalize_X()

    
