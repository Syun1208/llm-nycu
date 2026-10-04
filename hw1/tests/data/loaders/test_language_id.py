from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders.language_id import BatchCollator, build_split, build_vocabulary


def test_vocabulary_starts_with_special_tokens_and_maps_unknown() -> None:
    vocabulary = build_vocabulary(texts=["ab", "ba"])

    assert (vocabulary.pad_id, vocabulary.unk_id, vocabulary.bos_id, vocabulary.eos_id) == (0, 1, 2, 3)
    assert vocabulary.encode(text="az") == [vocabulary.token_to_id["a"], vocabulary.unk_id]


def test_collator_shifts_targets_and_pads() -> None:
    vocabulary = build_vocabulary(texts=["abc"])
    collate = BatchCollator(vocabulary=vocabulary)
    ids = vocabulary.encode(text="ab")

    inputs, targets = collate([ids, vocabulary.encode(text="abc")])

    assert inputs[0].tolist() == [vocabulary.bos_id, *ids, vocabulary.pad_id]
    assert targets[0].tolist() == [*ids, vocabulary.eos_id, vocabulary.pad_id]
    with pytest.raises(ValueError):
        collate([])


def test_split_groups_sequences_by_language() -> None:
    frame = pd.DataFrame(data={"Text": ["hi", "hola", "yo"], "Language": ["English", "Spanish", "English"]})
    vocabulary = build_vocabulary(texts=frame["Text"].tolist())

    split = build_split(
        frame=frame,
        vocabulary=vocabulary,
    )

    assert len(split.by_language["English"]) == 2
    assert len(split.by_language["French"]) == 0
