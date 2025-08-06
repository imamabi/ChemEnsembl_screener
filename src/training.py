from typing import Dict, Tuple
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.models import SEARCH_SPACES, build_model
from src.config import CONFIG
import logging

logger = logging.getLogger(__name__)

def objective(trial, model_name: str, X, y, cv) -> float:
    params = SEARCH_SPACES[model_name](trial)
    model = build_model(model_name, params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def train_with_optuna(model_name: str, X_train, y_train) -> Tuple[object, float, Dict]:
    logger.info("Starting Optuna for %s", model_name)
    cv = StratifiedKFold(
        n_splits=CONFIG["cv_folds"],
        shuffle=True,
        random_state=CONFIG["random_state"],
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=CONFIG["random_state"]),
    )
    study.optimize(
        lambda tr: objective(tr, model_name, X_train, y_train, cv),
        n_trials=CONFIG["optuna_trials"],
        show_progress_bar=True,
    )
    best_params = study.best_params
    best_score = study.best_value
    logger.info("%s best CV accuracy = %.4f", model_name, best_score)
    final_model = build_model(model_name, best_params)
    final_model.fit(X_train, y_train)
    return final_model, best_score, best_params
