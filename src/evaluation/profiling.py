"""Profiling utilities for latency and memory."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Tuple

import torch


def _supports_cuda() -> bool:
    return torch.cuda.is_available()


def _cuda_memory_reset() -> None:
    if _supports_cuda():
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_memory() -> int:
    if _supports_cuda():
        return torch.cuda.max_memory_reserved()
    return 0


def profile_generation(run_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, float]]:
    """Run generation under profiling; returns (result, metrics)."""
    metrics: Dict[str, float] = {}
    if _supports_cuda():
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        _cuda_memory_reset()
        start_evt.record()
        result = run_fn(*args, **kwargs)
        end_evt.record()
        torch.cuda.synchronize()
        latency_ms = start_evt.elapsed_time(end_evt)
        metrics["latency_ms"] = float(latency_ms)
        metrics["peak_memory_bytes"] = float(_cuda_peak_memory())
    else:
        start = time.monotonic()
        result = run_fn(*args, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000.0
        metrics["latency_ms"] = float(latency_ms)
        metrics["peak_memory_bytes"] = 0.0

    return result, metrics

