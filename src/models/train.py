import pandas as pd
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import joblib

sys.path.append(str(Path(__file__).parent.parent / "common"))
import utils

# GradientBoostingRegressor  - 0.902131027230735 0.7303601058202231
# elastic -1.0502970328309407 1.0203645836797532


def train():
    projet_dir = utils.get_projet_parent_dir()
    best_params = joblib.load(Path(projet_dir / "models/best_params.pkl"))
    print(best_params)
    X_train = pd.read_csv(Path(projet_dir / "data/processed_data/X_train_scaled.csv"))
    # X_test = pd.read_csv(Path(projet_dir / "data/processed_data/X_test_scaled.csv"))
    y_train = pd.read_csv(Path(projet_dir / "data/processed_data/y_train.csv"))
    # y_test = pd.read_csv(Path(projet_dir / "data/processed_data/y_test.csv"))

    # Fit
    gbr = GradientBoostingRegressor(**best_params)
    print(gbr)
    y_train = np.ravel(y_train)
    gbr.fit(X_train, np.ravel(y_train))
    joblib.dump(gbr, Path(projet_dir / "models/model.pkl"))


if __name__ == "__main__":
    train()
