"""Profiling utilities for latency and memory.

Expected responsibilities:
- Wrap generate calls with torch.cuda.Event timing and peak memory stats.
- Provide helpers for TTFT/ITL (autoregressive) and wall-clock (diffusion).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


def profile_generation(run_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, float]]:
    """Run generation under profiling; returns (result, metrics)."""
    raise NotImplementedError("Add torch.cuda timing and memory tracking here.")

