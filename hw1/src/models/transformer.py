from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.data.classes.settings import HeadType, PoolingType

SCORE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
POOLING_MULTIPLIER: dict[PoolingType, int] = {
    "mean": 1,
    "mean_max": 2,
    "segment": 5,
    "segment_max": 7,
    "align": 9,
    "align_max": 11,
}
ALIGN_SCALARS = 2
POSITIONAL_BASE = 10000.0
MAX_POSITIONS = 5000
MASK_FILL = -1e4
COUNT_FLOOR = 1e-9


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        max_len: int = MAX_POSITIONS,
    ) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(POSITIONAL_BASE) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer(
            name="pe",
            tensor=encoding.unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MyMultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.k_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.v_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.out_proj = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.attn_dropout = nn.Dropout(p=dropout)

    def split_heads(self, *, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.nhead, self.d_head).transpose(1, 2)

    def merge_heads(self, *, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.split_heads(x=self.q_proj(x))
        k = self.split_heads(x=self.k_proj(x))
        v = self.split_heads(x=self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        weights = self.attn_dropout(
            torch.softmax(
                scores,
                dim=-1,
            )
        )
        return self.out_proj(self.merge_heads(x=torch.matmul(weights, v)))


class MyTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = MyMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=d_model,
                out_features=dim_feedforward,
            ),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=dim_feedforward,
                out_features=d_model,
            ),
        )
        self.norm1 = nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self.self_attn(x, attention_mask)))
        return self.norm2(x + self.dropout2(self.feed_forward(x)))


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        pad_idx: int,
        sep_idx: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_len: int,
        pooling: PoolingType = "mean",
        head: HeadType = "sigmoid",
    ) -> None:
        super().__init__()
        self.d_model = embedding_dim
        self.sep_idx = sep_idx
        self.pooling = pooling
        self.head = head
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.pos_encoder = PositionalEncoding(
            d_model=embedding_dim,
            max_len=max_len,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_layers = nn.ModuleList(
            [
                MyTransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        in_features = embedding_dim * POOLING_MULTIPLIER[pooling] + (
            ALIGN_SCALARS if pooling.startswith("align") else 0
        )
        self.regression_head = nn.Linear(
            in_features=in_features,
            out_features=len(SCORE_LEVELS) if head == "softmax" else 1,
        )
        self.register_buffer(
            name="score_levels",
            tensor=torch.tensor(SCORE_LEVELS),
        )
        nn.init.normal_(
            tensor=self.embedding.weight,
            mean=0.0,
            std=embedding_dim**-0.5,
        )
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

    @staticmethod
    def masked_mean(
        *,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=COUNT_FLOOR)

    @staticmethod
    def masked_max(
        *,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return x.masked_fill(mask.unsqueeze(-1) == 0, MASK_FILL).max(dim=1).values

    def segment_masks(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_sep = input_ids == self.sep_idx
        segment_id = torch.cumsum(
            is_sep.long(),
            dim=1,
        )
        valid = attention_mask * (~is_sep).float()
        return valid * (segment_id == 1).float(), valid * (segment_id == 2).float()

    def align_features(
        self,
        *,
        x: torch.Tensor,
        anchor_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(x, x.transpose(1, 2)) / math.sqrt(x.size(-1))
        to_target = (
            torch.softmax(
                scores.masked_fill(target_mask[:, None, :] == 0, MASK_FILL),
                dim=-1,
            )
            @ x
        )
        to_anchor = (
            torch.softmax(
                scores.masked_fill(anchor_mask[:, None, :] == 0, MASK_FILL),
                dim=-1,
            )
            @ x
        )
        normalized = F.normalize(
            x,
            dim=-1,
        )
        cosine = torch.matmul(normalized, normalized.transpose(1, 2))
        anchor_best = cosine.masked_fill(target_mask[:, None, :] == 0, -1.0).max(dim=-1).values
        target_best = cosine.masked_fill(anchor_mask[:, None, :] == 0, -1.0).max(dim=-1).values
        return torch.cat(
            [
                self.masked_mean(
                    x=(x - to_target).abs(),
                    mask=anchor_mask,
                ),
                self.masked_mean(
                    x=x * to_target,
                    mask=anchor_mask,
                ),
                self.masked_mean(
                    x=(x - to_anchor).abs(),
                    mask=target_mask,
                ),
                self.masked_mean(
                    x=x * to_anchor,
                    mask=target_mask,
                ),
                (anchor_best * anchor_mask).sum(
                    1,
                    keepdim=True,
                )
                / anchor_mask.sum(
                    1,
                    keepdim=True,
                ).clamp(min=1),
                (target_best * target_mask).sum(
                    1,
                    keepdim=True,
                )
                / target_mask.sum(
                    1,
                    keepdim=True,
                ).clamp(min=1),
            ],
            dim=-1,
        )

    def segment_pooling(
        self,
        *,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        anchor_mask, target_mask = self.segment_masks(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        h_anchor = self.masked_mean(
            x=x,
            mask=anchor_mask,
        )
        h_target = self.masked_mean(
            x=x,
            mask=target_mask,
        )
        features = [
            self.masked_mean(
                x=x,
                mask=attention_mask,
            ),
            h_anchor,
            h_target,
            (h_anchor - h_target).abs(),
            h_anchor * h_target,
        ]
        if self.pooling.startswith("align"):
            features.append(
                self.align_features(
                    x=x,
                    anchor_mask=anchor_mask,
                    target_mask=target_mask,
                )
            )
        if self.pooling in ("segment_max", "align_max"):
            features += [
                self.masked_max(
                    x=x,
                    mask=anchor_mask,
                ),
                self.masked_max(
                    x=x,
                    mask=target_mask,
                ),
            ]
        return torch.cat(
            features,
            dim=-1,
        )

    def pool(
        self,
        *,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling.startswith(("segment", "align")):
            return self.segment_pooling(
                x=x,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        if self.pooling == "mean_max":
            return torch.cat(
                [
                    self.masked_mean(
                        x=x,
                        mask=attention_mask,
                    ),
                    self.masked_max(
                        x=x,
                        mask=attention_mask,
                    ),
                ],
                dim=-1,
            )
        return self.masked_mean(
            x=x,
            mask=attention_mask,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.dropout(self.pos_encoder(self.embedding(input_ids) * math.sqrt(self.d_model)))
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        pooled = self.pool(
            x=x,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if self.head == "softmax":
            return (
                torch.softmax(
                    self.regression_head(pooled),
                    dim=-1,
                )
                @ self.score_levels
            )
        return torch.sigmoid(self.regression_head(pooled).squeeze(-1))
