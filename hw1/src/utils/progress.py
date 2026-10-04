from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

COUNT_BAR = (
    "{desc} ▕{bar}▏ {percentage:3.0f}% • {n_fmt}/{total_fmt} • {rate_fmt}" " • ⏳ {elapsed}<{remaining}{postfix}"
)
TIME_BAR = "{desc} ▕{bar}▏ {percentage:3.0f}% • ⏳ {elapsed}<{remaining}{postfix}"


@dataclass(
    frozen=True,
    slots=True,
)
class Stage:
    emoji: str
    title: str
    colour: str


NGRAM = Stage(
    emoji="📝",
    title="N-gram",
    colour="green",
)
RNN = Stage(
    emoji="🧠",
    title="RNN",
    colour="cyan",
)
LSTM = Stage(
    emoji="🧬",
    title="LSTM",
    colour="magenta",
)
SIMILARITY = Stage(
    emoji="🔗",
    title="Transformer",
    colour="blue",
)
SUBMISSION = Stage(
    emoji="📤",
    title="Submission",
    colour="yellow",
)


def progress(
    *,
    stage: Stage,
    total: float | None,
    detail: str = "",
    unit: str = "it",
    show_counts: bool = True,
) -> tqdm:
    layout = COUNT_BAR if show_counts else TIME_BAR
    return tqdm(
        total=total,
        unit=unit,
        desc=f"{stage.emoji} {stage.title}{f' {detail}' if detail else ''}",
        colour=stage.colour,
        bar_format=layout if total else None,
        dynamic_ncols=True,
    )
