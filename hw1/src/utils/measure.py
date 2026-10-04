from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import TypeVar

import torch

from src.data.classes.results import Cost

T = TypeVar("T")
BYTES_PER_MB = 2**20


def measure_cpu(*, task: Callable[[], T]) -> tuple[T, Cost]:
    tracemalloc.start()
    start = time.perf_counter()
    output = task()
    seconds = time.perf_counter() - start
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return output, Cost(
        seconds=seconds,
        peak_mb=peak / BYTES_PER_MB,
    )


def measure_gpu(
    *,
    task: Callable[[], T],
    device: torch.device,
) -> tuple[T, Cost]:
    on_cuda = device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
    start = time.perf_counter()
    output = task()
    if on_cuda:
        torch.cuda.synchronize(device=device)
    peak = torch.cuda.max_memory_allocated(device=device) / BYTES_PER_MB if on_cuda else float("nan")
    return output, Cost(
        seconds=time.perf_counter() - start,
        peak_mb=peak,
    )
