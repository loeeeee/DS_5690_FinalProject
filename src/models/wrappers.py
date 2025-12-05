"""Adapters to present a unified generate() API for benchmarking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

import torch

logger = logging.getLogger(__name__)


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
        logger.info(f"AutoRegressiveWrapper.generate_batch: Starting - prompts: {len(prompts)}, max_new_tokens: {max_new_tokens}")
        device = next(self.model.parameters()).device
        logger.info(f"AutoRegressiveWrapper.generate_batch: Model device: {device}")
        
        logger.info("AutoRegressiveWrapper.generate_batch: Tokenizing prompts")
        tokenizer_outputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        logger.info(f"AutoRegressiveWrapper.generate_batch: Tokenization complete - input_ids shape: {tokenizer_outputs['input_ids'].shape}")
        
        # Move all tensors in tokenizer_outputs to the model's device
        logger.info(f"AutoRegressiveWrapper.generate_batch: Moving tensors to device: {device}")
        tokenizer_outputs = {k: v.to(device) for k, v in tokenizer_outputs.items()}
        logger.info("AutoRegressiveWrapper.generate_batch: Tensors moved to device")
        
        max_tokens = max_new_tokens or 128
        logger.info(f"AutoRegressiveWrapper.generate_batch: Starting model.generate() with max_new_tokens={max_tokens}")
        with torch.no_grad():
            # Standard generation - works on CUDA, may have issues on ROCm
            generated = self.model.generate(
                **tokenizer_outputs,
                max_new_tokens=max_tokens,
                use_cache=True,  # LLaMA supports KV cache
                do_sample=False,  # Greedy decoding for reproducibility
                pad_token_id=self.tokenizer.pad_token_id,
            )
        logger.info(f"AutoRegressiveWrapper.generate_batch: model.generate() completed - output shape: {generated.shape}")
        
        logger.info("AutoRegressiveWrapper.generate_batch: Decoding generated tokens")
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        logger.info(f"AutoRegressiveWrapper.generate_batch: Decoding complete - {len(texts)} texts")
        
        logger.info("AutoRegressiveWrapper.generate_batch: Computing token counts")
        token_counts: List[int] = []
        for out_ids, in_ids in zip(generated, tokenizer_outputs["input_ids"]):
            token_counts.append(len(out_ids) - len(in_ids))
        logger.info(f"AutoRegressiveWrapper.generate_batch: Token counts: {token_counts}")
        
        logger.info("AutoRegressiveWrapper.generate_batch: Returning result")
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
        logger.info(f"DiffusionLikeWrapper.generate_batch: Starting - prompts: {len(prompts)}, steps: {steps}, max_new_tokens: {max_new_tokens}")
        device = next(self.model.parameters()).device
        logger.info(f"DiffusionLikeWrapper.generate_batch: Model device: {device}")
        
        logger.info("DiffusionLikeWrapper.generate_batch: Tokenizing prompts")
        tokenizer_outputs = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        logger.info(f"DiffusionLikeWrapper.generate_batch: Tokenization complete - input_ids shape: {tokenizer_outputs['input_ids'].shape}")
        
        # Move all tensors in tokenizer_outputs to the model's device
        logger.info(f"DiffusionLikeWrapper.generate_batch: Moving tensors to device: {device}")
        tokenizer_outputs = {k: v.to(device) for k, v in tokenizer_outputs.items()}
        logger.info("DiffusionLikeWrapper.generate_batch: Tensors moved to device")
        
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens or 128,
            "use_cache": False,  # LLaDA doesn't support KV cache
        }
        logger.info(f"DiffusionLikeWrapper.generate_batch: Generation kwargs: {generation_kwargs}")
        
        # TODO: LLaDA step control - num_inference_steps is not accepted by this model's generate()
        # Steps may need to be controlled via generation_config or model config
        # For now, ignore steps parameter to allow benchmark to run
        logger.info("DiffusionLikeWrapper.generate_batch: Starting model.generate()")
        with torch.no_grad():
            generated = self.model.generate(
                **tokenizer_outputs,
                **generation_kwargs,
            )
        logger.info(f"DiffusionLikeWrapper.generate_batch: model.generate() completed - output shape: {generated.shape}")
        
        logger.info("DiffusionLikeWrapper.generate_batch: Decoding generated tokens")
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        logger.info(f"DiffusionLikeWrapper.generate_batch: Decoding complete - {len(texts)} texts")
        
        logger.info("DiffusionLikeWrapper.generate_batch: Computing token counts")
        token_counts: List[int] = []
        for out_ids, in_ids in zip(generated, tokenizer_outputs["input_ids"]):
            token_counts.append(len(out_ids) - len(in_ids))
        logger.info(f"DiffusionLikeWrapper.generate_batch: Token counts: {token_counts}")
        
        logger.info("DiffusionLikeWrapper.generate_batch: Creating result object")
        result = type(
            "GenerationBatchResult",
            (),
            {"texts": texts, "token_counts": token_counts, "timings": {}},
        )()
        logger.info("DiffusionLikeWrapper.generate_batch: Returning result")
        return result

