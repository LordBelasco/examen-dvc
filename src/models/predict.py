import pandas as pd
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from sklearn.metrics import root_mean_squared_error
import json
sys.path.append(str(Path(__file__).parent.parent / "common"))
import utils

# GradientBoostingRegressor  - 0.902131027230735 0.7303601058202231
# elastic -1.0502970328309407 1.0203645836797532


def train():
    projet_dir = utils.get_projet_parent_dir()
    model = joblib.load(Path(projet_dir / "models/model.pkl"))
    print(model)
    X_test = pd.read_csv(Path(projet_dir / "data/processed_data/X_test_scaled.csv"))
    y_test = pd.read_csv(Path(projet_dir / "data/processed_data/y_test.csv"))

    # Fit
    y_test = np.ravel(y_test)

    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"RMSE test : {rmse}")
    with open(Path(projet_dir / "metrics/scores.json"), "w") as out_file:
        json.dump({"RMSE test": rmse}, out_file)

    print(y_pred)
    prediction = pd.DataFrame(y_pred, columns=["y_pred"])
    print(prediction.head())
    prediction.to_csv(Path(projet_dir / "metrics/y_test.csv"))

if __name__ == "__main__":
    train()
