"""Adapters to present a unified generate() API for benchmarking.

Expected responsibilities:
- Normalize inputs/outputs between autoregressive and diffusion generation.
- Provide hooks for step sweeps (LLaDA) and capture token counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class GenerationResult(Protocol):
    text: str
    token_count: int
    timings: dict


@dataclass
class ModelWrapper:
    model: Any
    tokenizer: Any

    def generate(self, prompt: str, steps: int | None = None, max_new_tokens: int | None = None) -> GenerationResult:
        raise NotImplementedError("Wrap model-specific generate logic here.")

