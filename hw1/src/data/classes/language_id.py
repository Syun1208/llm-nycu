from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

LANGUAGES = ("English", "French", "Spanish", "Portuguese", "German")
LANGUAGE_CODES = {"English": "EN", "French": "FR", "Spanish": "ES", "Portuguese": "PT", "German": "DE"}
PAD, UNK, BOS, EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"


@dataclass(
    frozen=True,
    slots=True,
)
class Vocabulary:
    token_to_id: Mapping[str, int]

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def encode(self, *, text: str) -> list[int]:
        return [self.token_to_id.get(character, self.unk_id) for character in text]


@dataclass(
    frozen=True,
    slots=True,
)
class Split:
    frame: pd.DataFrame
    sequences: tuple[list[int], ...]
    by_language: Mapping[str, tuple[list[int], ...]]
