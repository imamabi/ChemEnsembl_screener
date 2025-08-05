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
_______________________________________________________________________

### Usage
1. Prepare Your Dataset

Ensure that your training and test sets are preprocessed, encoded, and ready for ingestion:
```python
X_train, X_test, y_train, y_test = ...
```

2. Run the Pipeline

You can run the pipeline in two ways:
- Use All Available Models
 ```python
from train_pipeline import run_pipeline

run_pipeline(X_train, y_train, X_test, y_test)
```
- Use a Subset of Models (Minimum 3)
```python
selected = ["forest", "xgboost", "svm"]
run_pipeline(
    X_train, y_train,
    X_test, y_test,
    selected_models=selected
)
```
**Note**: A ValueError will be raised if fewer than 3 models are selected. 

###### Each run is timestamped for easy tracking.

__________________________________________________________________________________________

### Evaluation Metrics

Each model is evaluated on:

-    Accuracy

-    Precision

-    Recall

-    F1-score

-    ROC-AUC (if applicable)

-    Classification Report

-    Confusion Matrix (raw + plot)
