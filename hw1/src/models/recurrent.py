from __future__ import annotations

import math

import torch
from torch import nn

LSTM_GATES = 4


class RNN(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(
            in_features=input_size,
            out_features=hidden_size,
        )
        self.hidden_to_hidden = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        )

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(
                batch_size,
                self.hidden_size,
                device=device,
            ),
        )

    def step(
        self,
        *,
        projected_input: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden = torch.tanh(projected_input + self.hidden_to_hidden(state[0]))
        return hidden, (hidden,)


class LSTM(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(
            in_features=input_size,
            out_features=LSTM_GATES * hidden_size,
        )
        self.hidden_to_hidden = nn.Linear(
            in_features=hidden_size,
            out_features=LSTM_GATES * hidden_size,
        )

    def initial_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        zeros = torch.zeros(
            batch_size,
            self.hidden_size,
            device=device,
        )
        return zeros, zeros.clone()

    def step(
        self,
        *,
        projected_input: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        hidden, cell = state
        input_gate, forget_gate, candidate, output_gate = (projected_input + self.hidden_to_hidden(hidden)).chunk(
            chunks=LSTM_GATES,
            dim=-1,
        )
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, (hidden, cell)


class StackedRecurrent(nn.Module):
    def __init__(
        self,
        *,
        cell_class: type[RNN] | type[LSTM],
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                cell_class(
                    input_size=input_size if index == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for index in range(num_layers)
            ]
        )
        bound = 1.0 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(
                tensor=parameter,
                a=-bound,
                b=bound,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = inputs
        for layer in self.layers:
            projected = layer.input_to_hidden(sequence)
            state = layer.initial_state(
                batch_size=sequence.size(0),
                device=sequence.device,
            )
            outputs = []
            for position in range(sequence.size(1)):
                hidden, state = layer.step(
                    projected_input=projected[:, position],
                    state=state,
                )
                outputs.append(hidden)
            sequence = torch.stack(
                tensors=outputs,
                dim=1,
            )
        return sequence
