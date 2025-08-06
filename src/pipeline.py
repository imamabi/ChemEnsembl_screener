from typing import Dict, List, Optional
import os
import datetime
import json
import logging
import joblib
from sklearn.ensemble import VotingClassifier
from src.training import train_with_optuna
from src.evaluation import evaluate_model
from src.config import CONFIG

logger = logging.getLogger(__name__)

def run_pipeline(X_train, y_train, X_test, y_test, voting: str = "hard", selected_models: Optional[List[str]] = None) -> Dict:
    """
    Orchestrates training, tuning, evaluation, and saving of models and ensemble.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test : test data
    voting : voting ensemble type "hard" or "soft"
    selected_models : optional list of model keys to train
                      must include at least 3 for ensemble
                      default is all models

    Returns
    -------
    summary dict with models, metrics, and paths to saved artefacts
    """

    all_models = [
        "knn", "svm", "gboost", "forest",
        "logreg", "naive_bayes", "xgboost",
    ]

    if selected_models is None:
        model_families = all_models
    else:
        invalid = set(selected_models) - set(all_models)
        if invalid:
            raise ValueError(f"Invalid models requested: {invalid}")
        if len(selected_models) < 3:
            raise ValueError("At least 3 models must be selected for voting ensemble.")
        model_families = selected_models

    logger.info(f"Selected models: {model_families}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    trained_models = {}
    cv_scores = {}
    best_params = {}
    metrics = {}
    model_paths = {}

    # Train and tune each model
    for name in model_families:
        model, cv_score, params = train_with_optuna(name, X_train, y_train)
        trained_models[name] = model
        cv_scores[name] = cv_score
        best_params[name] = params

    logger.info("Individual models trained.")
  
    # Validate voting
    if voting not in ("hard", "soft"):
        raise ValueError("voting must be 'hard' or 'soft'")

    # Build ensemble
    ensemble = VotingClassifier(
        estimators=[(n, m) for n, m in trained_models.items()],
        voting=voting,
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)
    trained_models["ensemble"] = ensemble
    logger.info("Ensemble trained.")

    # Evaluate and save models
    for label, model in trained_models.items():
        model_path = os.path.join(CONFIG["output_dir"], f"{label}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        model_paths[label] = model_path
        logger.info(f"Saved model {label} to {model_path}")

        metrics[label] = evaluate_model(model, X_test, y_test, label)

    # Save summary
    summary = {
        "timestamp": timestamp,
        "config": CONFIG,
        "optuna_cv_scores": cv_scores,
        "best_params": best_params,
        "metrics": metrics,
        "model_paths": model_paths,
    }

    summary_path = os.path.join(CONFIG["output_dir"], f"run_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Run summary saved to {summary_path}")

    logger.info("Pipeline run completed successfully.")

    return summary
