from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(
    frozen=True,
    slots=True,
)
class Cost:
    seconds: float
    peak_mb: float


@dataclass(
    frozen=True,
    slots=True,
)
class Evaluation:
    accuracy: float
    macro_f1: float
    corpus_ppl: float
    confusion: np.ndarray
    table: pd.DataFrame


@dataclass(
    frozen=True,
    slots=True,
)
class TrainingHistory:
    train_loss: tuple[float, ...]
    val_ppl: tuple[float, ...]

    @property
    def best_epoch(self) -> int:
        return int(np.argmin(self.val_ppl)) + 1

    @property
    def best_val_ppl(self) -> float:
        return min(self.val_ppl)


@dataclass(
    frozen=True,
    slots=True,
)
class RegressionHistory:
    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]
    val_pearson: tuple[float, ...]
    val_rmse: tuple[float, ...]

    @property
    def best_epoch(self) -> int:
        return int(np.argmax(self.val_pearson)) + 1
