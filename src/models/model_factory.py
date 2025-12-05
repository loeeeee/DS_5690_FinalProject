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
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_bundle(
    baseline_id: str,
    target_id: str,
    device: str = "cuda",
    precision: str = "bfloat16",
) -> ModelBundle:
    """Return initialized wrappers for baseline (LLaMA) and target (LLaDA)."""
    baseline_model, baseline_tokenizer = _load_auto_model(baseline_id, device, precision)
    target_model, target_tokenizer = _load_auto_model(target_id, device, precision)

    from models.wrappers import AutoRegressiveWrapper, DiffusionLikeWrapper

    baseline_wrapper = AutoRegressiveWrapper(baseline_model, baseline_tokenizer)
    target_wrapper = DiffusionLikeWrapper(target_model, target_tokenizer)

    return ModelBundle(baseline=baseline_wrapper, target=target_wrapper)

