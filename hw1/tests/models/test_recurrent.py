from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.recurrent import LSTM, StackedRecurrent, RNN

INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS = 6, 5, 2


def copy_reference_weights(
    *,
    reference: nn.RNNBase,
    scratch: StackedRecurrent,
) -> None:
    with torch.no_grad():
        for index, layer in enumerate(scratch.layers):
            layer.input_to_hidden.weight.copy_(getattr(reference, f"weight_ih_l{index}"))
            layer.input_to_hidden.bias.copy_(getattr(reference, f"bias_ih_l{index}"))
            layer.hidden_to_hidden.weight.copy_(getattr(reference, f"weight_hh_l{index}"))
            layer.hidden_to_hidden.bias.copy_(getattr(reference, f"bias_hh_l{index}"))


@pytest.mark.parametrize(
    argnames=("reference_class", "cell_class"),
    argvalues=[(nn.RNN, RNN), (nn.LSTM, LSTM)],
)
def test_scratch_matches_pytorch_reference(
    reference_class: type[nn.RNNBase],
    cell_class: type[nn.Module],
) -> None:
    torch.manual_seed(0)
    reference = reference_class(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        batch_first=True,
    )
    scratch = StackedRecurrent(
        cell_class=cell_class,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    copy_reference_weights(
        reference=reference,
        scratch=scratch,
    )
    inputs = torch.randn(3, 7, INPUT_SIZE)

    expected = reference(inputs)[0]
    actual = scratch(inputs)

    assert torch.allclose(
        actual,
        expected,
        atol=1e-6,
    )
    assert sum(parameter.numel() for parameter in scratch.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )


def test_gradients_flow_through_time() -> None:
    torch.manual_seed(0)
    scratch = StackedRecurrent(
        cell_class=LSTM,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    inputs = torch.randn(
        2,
        5,
        INPUT_SIZE,
        requires_grad=True,
    )

    scratch(inputs)[:, -1].sum().backward()

    assert inputs.grad is not None
    assert inputs.grad[:, 0].abs().sum() > 0
