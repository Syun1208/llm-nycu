from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data.classes.language_id import Vocabulary
from src.data.classes.settings import CellType, CharLMSettings
from src.data.loaders.language_id import BatchCollator
from src.models.recurrent import LSTM, StackedRecurrent, RNN

RECURRENT_CELLS: dict[CellType, type[RNN] | type[LSTM]] = {
    "rnn": RNN,
    "lstm": LSTM,
}


class CharLanguageModel(nn.Module):
    def __init__(
        self,
        *,
        cell: CellType,
        settings: CharLMSettings,
        vocabulary: Vocabulary,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.vocabulary = vocabulary
        self.collate = BatchCollator(vocabulary=vocabulary)
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary.size,
            embedding_dim=settings.embedding_dim,
            padding_idx=vocabulary.pad_id,
        )
        self.recurrent = StackedRecurrent(
            cell_class=RECURRENT_CELLS[cell],
            input_size=settings.embedding_dim,
            hidden_size=settings.hidden_dim,
            num_layers=settings.num_layers,
        )
        self.output = nn.Linear(
            in_features=settings.hidden_dim,
            out_features=vocabulary.size,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.recurrent(self.embedding(input_ids)))

    def token_nll(
        self,
        *,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = self(input_ids=inputs)
        return F.cross_entropy(
            input=logits.transpose(1, 2),
            target=targets,
            ignore_index=self.vocabulary.pad_id,
            reduction="none",
        )

    @torch.no_grad()
    def sequence_nll(self, *, sequences: Sequence[list[int]]) -> np.ndarray:
        self.eval()
        order = np.argsort([len(ids) for ids in sequences])
        nll = np.zeros(len(sequences))
        for start in range(0, len(order), self.settings.eval_batch_size):
            index = order[start : start + self.settings.eval_batch_size]
            inputs, targets = self.collate([sequences[i] for i in index])
            token_nll = self.token_nll(
                inputs=inputs.to(device=self.device),
                targets=targets.to(device=self.device),
            )
            nll[index] = token_nll.sum(dim=1).cpu().numpy()
        return nll
