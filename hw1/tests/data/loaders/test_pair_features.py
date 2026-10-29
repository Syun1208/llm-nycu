from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders.pair_features import (
    LEXICAL_FEATURE_NAMES,
    NEIGHBOUR_FEATURE_NAMES,
    AnchorNeighbourIndex,
    PairFeatureBuilder,
    lexical_features,
)


@pytest.fixture
def train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        data={
            "anchor": ["lift arm", "lift arm", "lift arm", "water pump"],
            "target": ["lifting arm", "lift lever", "fruit blend", "pump water"],
            "context": ["B66", "B66", "B66", "F04"],
            "score": [1.0, 0.5, 0.0, 0.75],
        }
    )


def test_lexical_features_reward_shared_stems() -> None:
    related = lexical_features(
        anchor="lift arm",
        target="lifting arm",
    )
    unrelated = lexical_features(
        anchor="lift arm",
        target="fruit blend",
    )

    assert len(related) == len(LEXICAL_FEATURE_NAMES)
    assert all(left > right for left, right in zip(related, unrelated))


def test_neighbour_features_leave_the_row_itself_out(train_frame: pd.DataFrame) -> None:
    index = AnchorNeighbourIndex(
        train_frame=train_frame,
        temperature=8.0,
    )

    with_self = index.features(
        anchor="lift arm",
        target="lifting arm",
        position=None,
    )
    without_self = index.features(
        anchor="lift arm",
        target="lifting arm",
        position=0,
    )

    assert with_self[2] == 1.0
    assert without_self[0] == pytest.approx(0.25)
    assert without_self[2] != 1.0


def test_unseen_anchor_falls_back_to_train_mean(train_frame: pd.DataFrame) -> None:
    index = AnchorNeighbourIndex(
        train_frame=train_frame,
        temperature=8.0,
    )

    features = index.features(
        anchor="unknown phrase",
        target="anything",
        position=None,
    )

    assert features == [train_frame["score"].mean(), train_frame["score"].mean(), 0.0, 0.0]


def test_builder_shapes(train_frame: pd.DataFrame) -> None:
    builder = PairFeatureBuilder(
        neighbours=AnchorNeighbourIndex(
            train_frame=train_frame,
            temperature=8.0,
        )
    )

    features = builder.build(
        frame=train_frame,
        exclude_self=True,
    )

    assert features.shape == (len(train_frame), len(LEXICAL_FEATURE_NAMES) + len(NEIGHBOUR_FEATURE_NAMES))
    assert builder.names == LEXICAL_FEATURE_NAMES + NEIGHBOUR_FEATURE_NAMES
