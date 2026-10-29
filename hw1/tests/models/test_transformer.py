from __future__ import annotations

import pytest
import torch

from src.models.transformer import MyMultiheadAttention, TransformerRegressor

VOCAB_SIZE, PAD, SEP = 20, 0, 3
LEXICAL_FEATURES = 14


@pytest.mark.parametrize(
    argnames="pooling",
    argvalues=["mean", "mean_max", "segment", "segment_max", "align", "align_max"],
)
@pytest.mark.parametrize(
    argnames="head",
    argvalues=["sigmoid", "softmax"],
)
def test_regressor_outputs_scores_in_unit_interval(
    pooling: str,
    head: str,
) -> None:
    torch.manual_seed(0)
    model = TransformerRegressor(
        vocab_size=VOCAB_SIZE,
        embedding_dim=16,
        pad_idx=PAD,
        sep_idx=SEP,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_len=10,
        pooling=pooling,
        head=head,
    )
    input_ids = torch.tensor([[2, 5, 3, 6, 7, 3, 8, 0, 0, 0], [2, 5, 3, 6, 3, 8, 9, 10, 0, 0]])

    scores = model(input_ids, (input_ids != PAD).float())

    assert scores.shape == (2,)
    assert bool(((scores >= 0) & (scores <= 1)).all())


@pytest.mark.parametrize(
    argnames="head",
    argvalues=["sigmoid", "softmax"],
)
def test_interaction_regressor_uses_lexical_features(head: str) -> None:
    torch.manual_seed(0)
    model = TransformerRegressor(
        vocab_size=VOCAB_SIZE,
        embedding_dim=16,
        pad_idx=PAD,
        sep_idx=SEP,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_len=10,
        pooling="align_max",
        head=head,
        lexical_features=LEXICAL_FEATURES,
        interaction=True,
    )
    input_ids = torch.tensor([[2, 5, 3, 6, 7, 3, 8, 0, 0, 0], [2, 5, 3, 6, 3, 8, 9, 10, 0, 0]])
    lexical = torch.rand(2, LEXICAL_FEATURES)

    scores = model(
        input_ids,
        (input_ids != PAD).float(),
        lexical,
    )
    scores.sum().backward()

    assert scores.shape == (2,)
    assert bool(((scores >= 0) & (scores <= 1)).all())
    assert model.pyramid[0].weight.grad is not None
    assert model.fusions[0].weight.grad is not None


def test_attention_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError):
        MyMultiheadAttention(
            d_model=10,
            nhead=4,
        )
