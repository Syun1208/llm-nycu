from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.metrics import f1_score

from src.utils.metrics import confusion_matrix, macro_f1, pearson_corr, rmse

LABELS = ("a", "b", "c")


def test_confusion_matrix_and_macro_f1_match_sklearn() -> None:
    generator = np.random.default_rng(seed=0)
    y_true = generator.choice(
        LABELS,
        size=200,
    ).tolist()
    y_pred = generator.choice(
        LABELS,
        size=200,
    ).tolist()

    matrix = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=LABELS,
    )

    assert (
        matrix
        == sklearn_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            labels=list(LABELS),
        )
    ).all()
    assert macro_f1(matrix=matrix) == pytest.approx(
        f1_score(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
        )
    )


def test_pearson_and_rmse() -> None:
    assert pearson_corr(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[2.0, 4.0, 6.0],
    ) == pytest.approx(1.0)
    assert (
        pearson_corr(
            y_true=[1.0, 1.0, 1.0],
            y_pred=[0.0, 1.0, 2.0],
        )
        == 0.0
    )
    assert rmse(
        y_true=[0.0, 0.0],
        y_pred=[3.0, 4.0],
    ) == pytest.approx(np.sqrt(12.5))
