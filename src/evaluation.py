import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import logging
import joblib
from src.config import CONFIG

logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm: np.ndarray,
                          labels: Tuple[str, str],
                          title: str,
                          save_path: str,
                          normalize: bool = False,
                          dpi: int = 120) -> None:
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}"
            ax.text(j, i, value,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Confusion-matrix figure saved to %s", save_path)

def evaluate_model(model, X_test, y_test, label: str):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    cm_array = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else "N/A",
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": cm_array.tolist(),
    }

    cm_png = os.path.join(
        CONFIG["output_dir"],
        f"{label}_confusion_matrix.png"
    )
    plot_confusion_matrix(
        cm_array,
        labels=("Inactive", "Active"),
        title=f"{label.title()} Confusion Matrix",
        save_path=cm_png,
        normalize=False,
        dpi=CONFIG["cm_figure_dpi"],
    )
    metrics["confusion_matrix_png"] = cm_png
    return metrics
