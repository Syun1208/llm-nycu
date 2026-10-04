from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np


class LanguageModel(Protocol):
    def sequence_nll(self, *, sequences: Sequence[list[int]]) -> np.ndarray: ...
