from typing import Dict
import optuna
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from src.config import CONFIG


def svm_space(trial: optuna.trial.Trial) -> Dict:
    return {
        "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
        "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
        "degree": trial.suggest_int("degree", 2, 5),
    }

def rf_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

def gb_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }

def knn_space(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 15, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "algorithm": trial.suggest_categorical("algorithm", ["ball_tree", "kd_tree"]),
    }

def logreg_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        "penalty": "l2",
        "solver": "liblinear",
    }

def xgb_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

SEARCH_SPACES = {
    "svm": svm_space,
    "forest": rf_space,
    "gboost": gb_space,
    "knn": knn_space,
    "logreg": logreg_space,
    "naive_bayes": lambda trial: {},  # no hyperparameters
    "xgboost": xgb_space,
}

def build_model(name: str, params: Dict):
    """Instantiate an unfitted model given family name and parameters."""
    if name == "svm":
        return svm.SVC(**params, probability=True, random_state=CONFIG["random_state"])
    if name == "forest":
        return RandomForestClassifier(**params, random_state=CONFIG["random_state"])
    if name == "gboost":
        return GradientBoostingClassifier(**params, random_state=CONFIG["random_state"])
    if name == "knn":
        return KNeighborsClassifier(**params)
    if name == "naive_bayes":
        return GaussianNB()
    if name == "logreg":
        return LogisticRegression(**params, max_iter=2000, random_state=CONFIG["random_state"])
    if name == "xgboost":
        return xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=CONFIG["random_state"],
        )
    raise ValueError(f"Unknown model family: {name}")
