from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PAD_TOKEN, UNK_TOKEN, CONTEXT_TOKEN, SEP_TOKEN = "<pad>", "<unk>", "<context>", "<sep>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, CONTEXT_TOKEN, SEP_TOKEN)
NON_ALPHANUMERIC = re.compile(pattern=r"[^a-z0-9]+")


def load_frame(
    *,
    data_dir: Path,
    name: str,
) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=data_dir / f"{name}.csv")


def tokenize(*, text: str) -> list[str]:
    return NON_ALPHANUMERIC.sub(" ", str(text).lower()).strip().split()


def tokenize_input(
    *,
    context: str,
    anchor: str,
    target: str,
) -> list[str]:
    return [
        CONTEXT_TOKEN,
        *tokenize(text=context),
        SEP_TOKEN,
        *tokenize(text=anchor),
        SEP_TOKEN,
        *tokenize(text=target),
    ]


def build_vocab(
    *,
    frame: pd.DataFrame,
    min_freq: int = 1,
) -> dict[str, int]:
    counter = Counter()
    for row in frame.itertuples(index=False):
        counter.update(
            tokenize_input(
                context=row.context,
                anchor=row.anchor,
                target=row.target,
            )
        )
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token, frequency in counter.items():
        if frequency >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_frame(
    *,
    frame: pd.DataFrame,
    vocab: Mapping[str, int],
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.full(
        size=(len(frame), max_len),
        fill_value=vocab[PAD_TOKEN],
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        size=(len(frame), max_len),
        dtype=torch.float,
    )
    for row_index, row in enumerate(frame.itertuples(index=False)):
        tokens = tokenize_input(
            context=row.context,
            anchor=row.anchor,
            target=row.target,
        )
        ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens][:max_len]
        input_ids[row_index, : len(ids)] = torch.tensor(
            ids,
            dtype=torch.long,
        )
        attention_mask[row_index, : len(ids)] = 1.0
    return input_ids, attention_mask


def build_loader(
    *,
    frame: pd.DataFrame,
    vocab: Mapping[str, int],
    max_len: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    input_ids, attention_mask = encode_frame(
        frame=frame,
        vocab=vocab,
        max_len=max_len,
    )
    scores = torch.tensor(
        frame["score"].to_numpy() if "score" in frame else [0.0] * len(frame),
        dtype=torch.float,
    )
    return DataLoader(
        dataset=TensorDataset(input_ids, attention_mask, scores),
        batch_size=batch_size,
        shuffle=shuffle,
    )
