"""
Configuration parameters for the classification project.
"""

CONFIG = {
    "random_state": 42,
    "cv_folds": 5,
    "optuna_trials": 50,
    "output_dir": "./models_optuna/",
    "log_file": "./application_optuna.log",
    "cm_figure_dpi": 120,
}
