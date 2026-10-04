from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Self

import numpy as np

from src.data.classes.language_id import Vocabulary

ALPHA = 1.0


class NGramLanguageModel:
    def __init__(
        self,
        *,
        n: int,
        vocabulary: Vocabulary,
        alpha: float = ALPHA,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n
        self.vocabulary = vocabulary
        self.alpha = alpha
        self.ngram_counts: defaultdict[tuple[int, ...], int] = defaultdict(int)
        self.context_counts: defaultdict[tuple[int, ...], int] = defaultdict(int)

    @property
    def stored_counts(self) -> int:
        return len(self.ngram_counts) + len(self.context_counts)

    def _ngrams(self, *, ids: list[int]) -> list[tuple[tuple[int, ...], int]]:
        padded = [self.vocabulary.bos_id] * (self.n - 1) + ids + [self.vocabulary.eos_id]
        return [(tuple(padded[t - self.n + 1 : t]), padded[t]) for t in range(self.n - 1, len(padded))]

    def fit(self, *, sequences: Sequence[list[int]]) -> Self:
        for ids in sequences:
            for context, token in self._ngrams(ids=ids):
                self.context_counts[context] += 1
                self.ngram_counts[context + (token,)] += 1
        return self

    def log_prob(
        self,
        *,
        context: tuple[int, ...],
        token: int,
    ) -> float:
        numerator = self.ngram_counts.get(context + (token,), 0) + self.alpha
        denominator = self.context_counts.get(context, 0) + self.alpha * self.vocabulary.size
        return math.log(numerator / denominator)

    def sequence_nll(self, *, sequences: Sequence[list[int]]) -> np.ndarray:
        return np.array(
            [
                -sum(
                    self.log_prob(
                        context=context,
                        token=token,
                    )
                    for context, token in self._ngrams(ids=ids)
                )
                for ids in sequences
            ]
        )
