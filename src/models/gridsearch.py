import pandas as pd
from pathlib import Path
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
import joblib
from sklearn.metrics import root_mean_squared_error

sys.path.append(str(Path(__file__).parent.parent / "common"))
import utils

# GradientBoostingRegressor  - 0.902131027230735 0.7303601058202231
# elastic -1.0502970328309407 1.0203645836797532

def gridsearch():
    projet_dir = utils.get_projet_parent_dir()
    X_train = pd.read_csv(Path(projet_dir / "data/processed_data/X_train_scaled.csv"))
    # X_test = pd.read_csv(Path(projet_dir / "data/processed_data/X_test_scaled.csv"))
    y_train = pd.read_csv(Path(projet_dir / "data/processed_data/y_train.csv"))
    # y_test = pd.read_csv(Path(projet_dir / "data/processed_data/y_test.csv"))

    grid = GridSearchCV(
        GradientBoostingRegressor(),
        {
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
        },
        # ElasticNet(),
        # {
        #     "alpha": [0.1, 1.0, 10.0],
        #     "l1_ratio": [0.1, 0.5, 0.7, 0.9],
        # },
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=5, random_state=42, shuffle=True),
        n_jobs=-1,
        verbose=1,
    )
    # Fit
    y_train = np.ravel(y_train)
    grid.fit(X_train, np.ravel(y_train))
    # Meilleur score et meilleurs hyperparamètres
    best_params = grid.best_params_
    print(grid.best_score_, best_params)
    y_train_pred = grid.best_estimator_.predict(X_train)
    rmse_train = root_mean_squared_error(y_train, y_train_pred)
    print(f"RMSE train : {rmse_train}")
    joblib.dump(best_params, Path(projet_dir / "models/best_params.pkl"))


if __name__ == "__main__":
    gridsearch()
