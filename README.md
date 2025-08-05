# ChemEnsembl_screener
## Inhibitor Classification Pipeline — Optuna-Optimized Version
Automated training and hyperparameter optimization of seven machine learning models using Optuna, integrated into a robust ensemble VotingClassifier.
_________________________________________________________________________________________________________________________________________________________

### Overview
This repository implements a robust and extensible machine learning pipeline for classifying EGFR inhibitors. The workflow leverages Optuna for automated hyperparameter optimization across multiple classical ML models and assembles them into a hard-voting ensemble for improved prediction performance.

#### Key Features

- Trains and tunes upto 7 model families (SVM, RF, GBoost, KNN, Logistic Regression, Naive Bayes, XGBoost)

- Uses Optuna (TPE sampler) for efficient hyperparameter optimization

- Combines selected models using VotingClassifier

- Evaluates all models with key metrics and confusion matrices

- Saves all artefacts (models, plots, configs, summaries) for reproducibility

- Supports user-defined model selection with a minimum of 3 models


_______________________________________________________________________

### Dependencies

Ensure the following packages are installed:
```bash
pip install scikit-learn xgboost optuna matplotlib joblib numpy
```
