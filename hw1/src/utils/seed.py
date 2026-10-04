from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(*, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(*, device_id: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_id}" if device_id < torch.cuda.device_count() else "cuda")
