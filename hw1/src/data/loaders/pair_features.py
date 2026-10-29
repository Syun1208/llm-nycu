from __future__ import annotations

import difflib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.data.loaders.similarity import tokenize

CHAR_GRAM_SIZE = 3
SHORT_STEM, STEM, LONG_STEM = 3, 4, 5
BIGRAM, FOURGRAM = 2, 4
LEXICAL_FEATURE_NAMES = (
    "stem_overlap",
    "char_trigram_jaccard",
    "edit_ratio",
    "char_bigram_jaccard",
    "char_fourgram_jaccard",
    "short_stem_overlap",
    "long_stem_overlap",
    "soft_match_anchor",
    "soft_match_target",
    "soft_match_min",
)
NEIGHBOUR_FEATURE_NAMES = (
    "anchor_mean_score",
    "similarity_weighted_score",
    "nearest_target_score",
    "nearest_target_similarity",
)


def jaccard(
    *,
    left: set[str],
    right: set[str],
) -> float:
    return len(left & right) / max(1, len(left | right))


def char_grams(
    *,
    text: str,
    size: int = CHAR_GRAM_SIZE,
) -> set[str]:
    padded = f" {text} "
    return {padded[i : i + size] for i in range(len(padded) - size + 1)}


def stems(
    *,
    words: Sequence[str],
    length: int,
) -> set[str]:
    return {word[:length] for word in words}


def best_word_matches(
    *,
    source: Sequence[str],
    other: Sequence[str],
) -> list[float]:
    return [
        max(
            [
                jaccard(
                    left=char_grams(text=word),
                    right=char_grams(text=candidate),
                )
                for candidate in other
            ]
            or [0.0]
        )
        for word in source
    ]


def lexical_features(
    *,
    anchor: str,
    target: str,
) -> list[float]:
    anchor_words, target_words = tokenize(text=anchor), tokenize(text=target)
    anchor_text, target_text = " ".join(anchor_words), " ".join(target_words)
    anchor_matches = best_word_matches(
        source=anchor_words,
        other=target_words,
    )
    target_matches = best_word_matches(
        source=target_words,
        other=anchor_words,
    )

    def stem_overlap(*, length: int) -> float:
        return jaccard(
            left=stems(
                words=anchor_words,
                length=length,
            ),
            right=stems(
                words=target_words,
                length=length,
            ),
        )

    def gram_overlap(*, size: int) -> float:
        return jaccard(
            left=char_grams(
                text=anchor_text,
                size=size,
            ),
            right=char_grams(
                text=target_text,
                size=size,
            ),
        )

    return [
        stem_overlap(length=STEM),
        gram_overlap(size=CHAR_GRAM_SIZE),
        difflib.SequenceMatcher(None, anchor_text, target_text).ratio(),
        gram_overlap(size=BIGRAM),
        gram_overlap(size=FOURGRAM),
        stem_overlap(length=SHORT_STEM),
        stem_overlap(length=LONG_STEM),
        float(np.mean(anchor_matches)) if anchor_matches else 0.0,
        float(np.mean(target_matches)) if target_matches else 0.0,
        min(anchor_matches + target_matches) if anchor_matches and target_matches else 0.0,
    ]


@dataclass(
    frozen=True,
    slots=True,
)
class TrainNeighbour:
    position: int
    grams: frozenset[str]
    score: float


class AnchorNeighbourIndex:
    def __init__(
        self,
        *,
        train_frame: pd.DataFrame,
        temperature: float,
    ) -> None:
        self.temperature = temperature
        self.mean_score = float(train_frame["score"].mean())
        self._by_anchor: defaultdict[str, list[TrainNeighbour]] = defaultdict(list)
        for position, row in enumerate(train_frame.itertuples(index=False)):
            self._by_anchor[row.anchor].append(
                TrainNeighbour(
                    position=position,
                    grams=frozenset(char_grams(text=" ".join(tokenize(text=row.target)))),
                    score=float(row.score),
                )
            )

    def features(
        self,
        *,
        anchor: str,
        target: str,
        position: int | None,
    ) -> list[float]:
        neighbours = [item for item in self._by_anchor.get(anchor, []) if item.position != position]
        if not neighbours:
            return [self.mean_score, self.mean_score, 0.0, 0.0]

        target_grams = char_grams(text=" ".join(tokenize(text=target)))
        similarities = np.array(
            [
                jaccard(
                    left=target_grams,
                    right=set(item.grams),
                )
                for item in neighbours
            ]
        )
        scores = np.array([item.score for item in neighbours])
        weights = np.exp(self.temperature * similarities)
        nearest = int(similarities.argmax())
        return [
            float(scores.mean()),
            float((weights * scores).sum() / weights.sum()),
            float(scores[nearest]),
            float(similarities[nearest]),
        ]


class PairFeatureBuilder:
    def __init__(self, *, neighbours: AnchorNeighbourIndex | None) -> None:
        self.neighbours = neighbours

    @property
    def names(self) -> tuple[str, ...]:
        return LEXICAL_FEATURE_NAMES + (NEIGHBOUR_FEATURE_NAMES if self.neighbours is not None else ())

    def row(
        self,
        *,
        anchor: str,
        target: str,
        position: int | None,
    ) -> list[float]:
        features = lexical_features(
            anchor=anchor,
            target=target,
        )
        if self.neighbours is not None:
            features += self.neighbours.features(
                anchor=anchor,
                target=target,
                position=position,
            )
        return features

    def build(
        self,
        *,
        frame: pd.DataFrame,
        exclude_self: bool,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                self.row(
                    anchor=row.anchor,
                    target=row.target,
                    position=position if exclude_self else None,
                )
                for position, row in enumerate(frame.itertuples(index=False))
            ],
            dtype=torch.float,
        )
