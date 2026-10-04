from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import torch

from src.data.classes.language_id import BOS, EOS, LANGUAGES, PAD, UNK, Split, Vocabulary


def load_split(
    *,
    data_dir: Path,
    name: str,
) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=data_dir / f"{name}.csv")


def build_vocabulary(*, texts: Sequence[str]) -> Vocabulary:
    tokens = [PAD, UNK, BOS, EOS, *sorted(set("".join(texts)))]
    return Vocabulary(token_to_id={token: index for index, token in enumerate(tokens)})


def build_split(
    *,
    frame: pd.DataFrame,
    vocabulary: Vocabulary,
) -> Split:
    sequences = tuple(vocabulary.encode(text=text) for text in frame["Text"])
    languages = frame["Language"].tolist()
    by_language = {
        language: tuple(ids for ids, label in zip(sequences, languages) if label == language) for language in LANGUAGES
    }
    return Split(
        frame=frame,
        sequences=sequences,
        by_language=by_language,
    )


class BatchCollator:
    def __init__(self, *, vocabulary: Vocabulary) -> None:
        self.vocabulary = vocabulary

    def __call__(self, sequences: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("cannot collate an empty batch")
        width = max(len(ids) for ids in sequences) + 1
        inputs = torch.full(
            size=(len(sequences), width),
            fill_value=self.vocabulary.pad_id,
            dtype=torch.long,
        )
        targets = torch.full(
            size=(len(sequences), width),
            fill_value=self.vocabulary.pad_id,
            dtype=torch.long,
        )
        for row, ids in enumerate(sequences):
            inputs[row, : len(ids) + 1] = torch.tensor([self.vocabulary.bos_id, *ids])
            targets[row, : len(ids) + 1] = torch.tensor([*ids, self.vocabulary.eos_id])
        return inputs, targets
