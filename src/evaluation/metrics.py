from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """
    Calculate macro-averaged F1 score.

    Args:
        y_true: List or array of true labels
        y_pred: List or array of predicted labels
        n_classes: Number of classes

    Returns:
        Macro-averaged F1 score
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))


@dataclass(frozen=True)
class ClsMetrics:
    acc: float
    balanced_acc: float
    f1_macro: float
    precision_macro: float
    recall_macro: float


def classification_metrics(y_true, y_pred) -> ClsMetrics:
    """
    Common classification metrics for leaderboard logging.
    """
    acc = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true)))
    return ClsMetrics(
        acc=acc,
        balanced_acc=float(balanced_accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0.0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0.0)),
    )


def per_class_metrics(y_true, y_pred, class_labels: list[str]) -> list[dict[str, float | int | str]]:
    """Return precision/recall/f1/support for each class in fixed label order."""
    label_ids = list(range(len(class_labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        average=None,
        zero_division=0.0,
    )
    rows = []
    for idx, class_name in enumerate(class_labels):
        rows.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return rows
