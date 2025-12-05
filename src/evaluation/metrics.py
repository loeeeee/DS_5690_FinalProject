"""Quality metrics helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch
from torch.nn import CrossEntropyLoss


def _prepare_texts(texts: Sequence[str], max_samples: int = 64) -> List[str]:
    filtered = [t for t in texts if t and t.strip()]
    return filtered[:max_samples]


def perplexity_from_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int = 256,
    device: str | None = None,
) -> float:
    """Compute average perplexity over the provided texts."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum")
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if "token_type_ids" in enc:
            enc["token_type_ids"] = enc["token_type_ids"].to(device)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        valid_tokens = (shift_labels != -100).sum().item()
        total_tokens += valid_tokens
        total_nll += loss.item()

    if total_tokens == 0:
        return math.inf
    avg_neg_log_likelihood = total_nll / total_tokens
    return float(math.exp(avg_neg_log_likelihood))


def compute_metrics(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int = 256,
    compute_bertscore: bool = False,
) -> Dict[str, float]:
    """Compute perplexity (and optionally BERTScore) over provided texts."""
    prepared = _prepare_texts(texts)
    ppl = perplexity_from_texts(model, tokenizer, prepared, max_length=max_length)
    metrics: Dict[str, float] = {"perplexity": ppl}

    if compute_bertscore:
        try:
            from evaluate import load as load_metric

            bertscore = load_metric("bertscore")
            refs = prepared
            preds = prepared
            score = bertscore.compute(predictions=preds, references=refs, lang="en")
            metrics["bertscore_f1"] = float(sum(score["f1"]) / len(score["f1"]))
        except Exception:
            metrics["bertscore_f1"] = float("nan")

    return metrics

