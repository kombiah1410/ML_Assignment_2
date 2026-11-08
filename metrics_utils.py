"""Shared metric code, used by BOTH train_models.py and app.py so the numbers
in the README and the numbers in the Streamlit app are computed identically."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def averaging_for(n_classes: int) -> str:
    # Binary -> metrics for the positive class (label index 1).
    # Multi-class -> support-weighted average across classes.
    return "binary" if n_classes == 2 else "weighted"


def auc_score(y_true, y_proba, n_classes):
    """ROC-AUC. Returns NaN when it is mathematically undefined
    (e.g. the uploaded file contains only one class)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if n_classes == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="weighted",
            labels=np.arange(n_classes),
        )
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_pred, y_proba, n_classes):
    avg = averaging_for(n_classes)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_score(y_true, y_proba, n_classes),
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def confusion_df(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return pd.DataFrame(cm, index=[f"true: {c}" for c in class_names],
                        columns=[f"pred: {c}" for c in class_names])


def report_df(y_true, y_pred, class_names):
    rep = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=[str(c) for c in class_names], output_dict=True, zero_division=0,
    )
    return pd.DataFrame(rep).T
