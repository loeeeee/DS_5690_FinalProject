"""Adapters to present a unified generate() API for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

import torch


class GenerationBatchResult(Protocol):
    texts: List[str]
    token_counts: List[int]
    timings: Dict[str, float]


@dataclass
class ModelWrapper:
    model: Any
    tokenizer: Any

    def generate_batch(
        self,
        prompts: Sequence[str],
        steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationBatchResult:
        raise NotImplementedError("Wrap model-specific generate logic here.")


class AutoRegressiveWrapper(ModelWrapper):
    """Wrapper for standard autoregressive CausalLM models."""

    def generate_batch(
        self,
        prompts: Sequence[str],
        steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationBatchResult:
        device = next(self.model.parameters()).device
        tokenizer_outputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        max_tokens = max_new_tokens or 128
        with torch.no_grad():
            # ROCm compatibility: disable Flash Attention and use explicit settings
            generated = self.model.generate(
                **tokenizer_outputs,
                max_new_tokens=max_tokens,
                use_cache=True,  # LLaMA supports KV cache
                do_sample=False,  # Greedy decoding for reproducibility
            )
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        token_counts: List[int] = []
        for out_ids, in_ids in zip(generated, tokenizer_outputs["input_ids"]):
            token_counts.append(len(out_ids) - len(in_ids))
        return type(
            "GenerationBatchResult",
            (),
            {"texts": texts, "token_counts": token_counts, "timings": {}},
        )()


class DiffusionLikeWrapper(ModelWrapper):
    """Wrapper placeholder for LLaDA diffusion-style generation."""

    def generate_batch(
        self,
        prompts: Sequence[str],
        steps: int | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationBatchResult:
        device = next(self.model.parameters()).device
        tokenizer_outputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or 128,
            "use_cache": False,  # LLaDA doesn't support KV cache
        }
        # TODO: LLaDA step control - num_inference_steps is not accepted by this model's generate()
        # Steps may need to be controlled via generation_config or model config
        # For now, ignore steps parameter to allow benchmark to run
        with torch.no_grad():
            generated = self.model.generate(
                **tokenizer_outputs,
                **generation_kwargs,
            )
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        token_counts: List[int] = []
        for out_ids, in_ids in zip(generated, tokenizer_outputs["input_ids"]):
            token_counts.append(len(out_ids) - len(in_ids))
        return type(
            "GenerationBatchResult",
            (),
            {"texts": texts, "token_counts": token_counts, "timings": {}},
        )()

