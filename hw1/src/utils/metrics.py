from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPSILON = 1e-12


def confusion_matrix(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> np.ndarray:
    matrix = np.zeros(
        shape=(len(labels), len(labels)),
        dtype=int,
    )
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[labels.index(true_label), labels.index(pred_label)] += 1
    return matrix


def macro_f1(*, matrix: np.ndarray) -> float:
    hits = np.diag(matrix).astype(float)
    precision = hits / np.maximum(matrix.sum(axis=0), 1)
    recall = hits / np.maximum(matrix.sum(axis=1), 1)
    return float(np.mean(2 * precision * recall / np.maximum(precision + recall, EPSILON)))


def pearson_corr(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    truth = np.asarray(
        y_true,
        dtype=np.float64,
    )
    prediction = np.asarray(
        y_pred,
        dtype=np.float64,
    )
    if np.std(truth) == 0 or np.std(prediction) == 0:
        return 0.0
    return float(np.corrcoef(truth, prediction)[0, 1])


def rmse(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> float:
    truth = np.asarray(
        y_true,
        dtype=np.float64,
    )
    prediction = np.asarray(
        y_pred,
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean((truth - prediction) ** 2)))
