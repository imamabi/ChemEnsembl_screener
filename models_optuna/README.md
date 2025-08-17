This directory contains the trained machine-learning models, confusion matrix visualizations, and summary artefacts from the Optuna-driven hyperparameter optimization pipeline applied to EGFR inhibitor classification.

___________________________________________________________________________________

Individual model files (*.joblib)
Each serialized model (SVM, Random Forest, XGBoost, etc.), trained on the scaffold-split dataset with the best hyperparameters found via Optuna.

Confusion matrices (*_confusion_matrix_*.png)
Heatmaps showing classification performance for each model on the held-out test set.

Run summaries (run_summary_*.json)
JSON logs that capture:

Best hyperparameters per model

Cross-validation scores during tuning

Final test metrics (accuracy, precision, recall, F1, ROC-AUC)

Paths to confusion matrix figures and model artefacts

________________________________________________________________________________________

Experimental Setup

Dataset: Scaffold-split EGFR inhibitors (ChEMBL source)

Features: 2048-bit Morgan fingerprints + selected RDKit descriptors

Tuning: Optuna Bayesian optimization with 5-fold stratified cross-validation

Models considered:
- Support Vector Machine (SVM, RBF kernel)
- Random Forest
- XGBoost
- Soft Voting Ensembles
