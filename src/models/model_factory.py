"""Factory functions to load baseline and target models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.wrappers import ModelWrapper


@dataclass
class ModelBundle:
    baseline: ModelWrapper
    target: ModelWrapper


def _load_auto_model(model_id: str, device: str, precision: str) -> tuple[Any, Any]:
    torch_dtype = torch.bfloat16 if precision == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # ROCm compatibility: disable Flash Attention and other optimizations that may cause segfaults
    import os
    # Disable Flash Attention if available
    os.environ.setdefault("DISABLE_FLASH_ATTENTION", "1")
    # Use standard attention implementation
    os.environ.setdefault("TRANSFORMERS_NO_FLASH_ATTENTION_2", "1")
    
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    if device == "cuda":
        # Load model with explicit device placement
        # For ROCm compatibility, avoid device_map="auto" during generation
        # Load directly to GPU, but use low_cpu_mem_usage to reduce peak memory
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )
        # Move to GPU explicitly - this is more reliable for ROCm
        model = model.to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )
        model = model.to(device)
    
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ROCm compatibility: synchronize after model loading
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return model, tokenizer


def load_model_bundle(
    baseline_id: str,
    target_id: str,
    device: str = "cuda",
    precision: str = "bfloat16",
) -> ModelBundle:
    """Return initialized wrappers for baseline (LLaMA) and target (LLaDA)."""
    # For 20GB GPUs, we can't fit both 8B models simultaneously
    # Load baseline first, use it, then load target
    # This requires modifying the benchmark to load models on-demand
    # For now, try loading both with aggressive memory clearing
    
    baseline_model, baseline_tokenizer = _load_auto_model(baseline_id, device, precision)
    
    # Clear GPU cache aggressively between model loads
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    target_model, target_tokenizer = _load_auto_model(target_id, device, precision)

    from models.wrappers import AutoRegressiveWrapper, DiffusionLikeWrapper

    baseline_wrapper = AutoRegressiveWrapper(baseline_model, baseline_tokenizer)
    target_wrapper = DiffusionLikeWrapper(target_model, target_tokenizer)

    return ModelBundle(baseline=baseline_wrapper, target=target_wrapper)

