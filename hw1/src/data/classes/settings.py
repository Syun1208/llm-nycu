from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

CellType = Literal["rnn", "lstm"]
PoolingType = Literal["mean", "mean_max", "segment", "segment_max", "align", "align_max"]
HeadType = Literal["sigmoid", "softmax"]
OptimizerName = Literal["adam", "adamw"]
SchedulerName = Literal["cosine", "plateau", "none"]


@dataclass(
    frozen=True,
    slots=True,
)
class CharLMSettings:
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    learning_rate: float = 1e-3
    epochs: int = 40
    batch_size: int = 32
    grad_clip: float = 1.0
    eval_batch_size: int = 128


@dataclass(
    frozen=True,
    slots=True,
)
class SimilaritySettings:
    max_len: int = 64
    batch_size: int = 64
    embedding_dim: int = 128
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.05
    learning_rate: float = 2e-3
    weight_decay: float = 0.01
    epochs: int = 20
    optimizer: OptimizerName = "adamw"
    scheduler: SchedulerName = "cosine"
    min_learning_rate: float = 1e-5

    def __post_init__(self) -> None:
        if self.embedding_dim % 8 != 0:
            raise ValueError("embedding_dim must be divisible by 1, 4 and 8")


@dataclass(
    frozen=True,
    slots=True,
)
class EnsembleMember:
    nhead: int
    pooling: PoolingType
    head: HeadType = "sigmoid"
    learning_rate: float | None = None
    weight_decay: float | None = None

    @property
    def name(self) -> str:
        return f"nhead={self.nhead}, pooling={self.pooling}, head={self.head}"

    def resolve(self, *, settings: SimilaritySettings) -> SimilaritySettings:
        return replace(
            settings,
            learning_rate=self.learning_rate if self.learning_rate is not None else settings.learning_rate,
            weight_decay=self.weight_decay if self.weight_decay is not None else settings.weight_decay,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class LanguageIdRunConfig:
    data_dir: Path
    output_dir: Path
    device_id: int
    seed: int = 42
    cells: tuple[CellType, ...] = ("rnn", "lstm")
    char_lm: CharLMSettings = field(default_factory=CharLMSettings)


@dataclass(
    frozen=True,
    slots=True,
)
class SimilarityRunConfig:
    data_dir: Path
    output_dir: Path
    device_id: int
    members: tuple[EnsembleMember, ...]
    seed: int = 42
    settings: SimilaritySettings = field(default_factory=SimilaritySettings)
