"""CLI entrypoint for headless benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import yaml
from datasets import load_dataset

from evaluation.metrics import compute_metrics
from evaluation.profiling import profile_generation
from models.model_factory import load_model_bundle
from models.wrappers import GenerationBatchResult, ModelWrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLaDA vs LLaMA benchmarking harness")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiments.yaml")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for raw metric outputs")
    parser.add_argument("--steps", type=int, help="Override diffusion steps for LLaDA")
    parser.add_argument("--batch_size", type=int, help="Override batch size for sweep/debug")
    parser.add_argument("--prompt_file", type=Path, help="Optional prompt file for reproducible inputs")
    parser.add_argument("--max_prompts", type=int, default=32, help="Limit prompts for quick runs")
    parser.add_argument("--compute_bertscore", action="store_true", help="Also compute BERTScore (slower)")
    return parser


def _load_prompts(prompt_file: Path | None, max_prompts: int, seq_length: int) -> List[str]:
    if prompt_file:
        lines = prompt_file.read_text().strip().splitlines()
        return lines[:max_prompts]
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    prompts: List[str] = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        prompts.append(text[:seq_length])
        if len(prompts) >= max_prompts:
            break
    return prompts


def _batched(iterable: Iterable[Any], batch_size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    _ensure_output_dir(output_path.parent)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "no rows recorded"
    avg_latency = sum(r["latency_ms"] for r in rows) / len(rows)
    avg_tps = sum(r["throughput_tps"] for r in rows) / len(rows)
    return f"avg latency {avg_latency:.2f} ms | avg throughput {avg_tps:.2f} tok/s over {len(rows)} batches"


def _run_model(
    name: str,
    wrapper: ModelWrapper,
    prompts: List[str],
    batch_size: int,
    seq_length: int,
    steps: int | None,
    max_new_tokens: int,
    warmup: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    warmup_prompts = prompts[: min(len(prompts), max(batch_size, 1))]
    for _ in range(warmup):
        for batch in _batched(warmup_prompts, batch_size):
            _ = wrapper.generate_batch(batch, steps=steps, max_new_tokens=max_new_tokens)

    for batch_idx, batch in enumerate(_batched(prompts, batch_size)):
        result: GenerationBatchResult
        result, profile = profile_generation(
            wrapper.generate_batch,
            batch,
            steps=steps,
            max_new_tokens=max_new_tokens,
        )
        total_tokens = sum(result.token_counts)
        throughput = total_tokens / (profile["latency_ms"] / 1000.0) if profile["latency_ms"] > 0 else 0.0
        rows.append(
            {
                "model": name,
                "batch_index": batch_idx,
                "batch_size": len(batch),
                "sequence_length": seq_length,
                "steps": steps or 0,
                "latency_ms": profile["latency_ms"],
                "peak_memory_bytes": profile.get("peak_memory_bytes", math.nan),
                "tokens_generated": total_tokens,
                "throughput_tps": throughput,
                "ttft_ms": profile.get("ttft_ms", math.nan),
                "itl_ms": profile.get("itl_ms", math.nan),
            }
        )
    return rows


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    with args.config.open("r") as f:
        config = yaml.safe_load(f)

    experiment = config.get("experiment", {})
    baseline_id = experiment.get("baseline_id")
    target_id = experiment.get("target_id")
    max_new_tokens = int(experiment.get("max_new_tokens", 128))
    warmup = int(experiment.get("warmup", 3))
    seq_lengths = experiment.get("sequence_lengths", [512])
    batch_sizes = experiment.get("batch_sizes", [1, 4])
    diffusion_steps = experiment.get("diffusion_steps", [32, 64, 128])

    if args.batch_size:
        batch_sizes = [args.batch_size]
    if args.steps:
        diffusion_steps = [args.steps]

    prompts = _load_prompts(args.prompt_file, args.max_prompts, max(seq_lengths))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = experiment.get("precision", "bfloat16")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []

    # Load baseline model first, run all baseline experiments, then unload
    from models.model_factory import _load_auto_model
    from models.wrappers import AutoRegressiveWrapper
    baseline_model, baseline_tokenizer = _load_auto_model(baseline_id, device, precision)
    baseline_wrapper = AutoRegressiveWrapper(baseline_model, baseline_tokenizer)

    for seq_length in seq_lengths:
        trimmed_prompts = [p[:seq_length] for p in prompts]
        for batch_size in batch_sizes:
            baseline_rows = _run_model(
                name="baseline",
                wrapper=baseline_wrapper,
                prompts=trimmed_prompts,
                batch_size=batch_size,
                seq_length=seq_length,
                steps=None,
                max_new_tokens=max_new_tokens,
                warmup=warmup,
            )
            all_rows.extend(baseline_rows)
    
    # Save baseline model/tokenizer for quality metrics, then unload to free memory
    baseline_quality_model = baseline_model
    baseline_quality_tokenizer = baseline_tokenizer
    
    if device == "cuda":
        del baseline_model, baseline_wrapper
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Now load LLaDA model
    from models.wrappers import DiffusionLikeWrapper
    target_model, target_tokenizer = _load_auto_model(target_id, device, precision)
    target_wrapper = DiffusionLikeWrapper(target_model, target_tokenizer)

    for seq_length in seq_lengths:
        trimmed_prompts = [p[:seq_length] for p in prompts]
        for batch_size in batch_sizes:
            for k in diffusion_steps:
                llada_rows = _run_model(
                    name=f"llada_steps_{k}",
                    wrapper=target_wrapper,
                    prompts=trimmed_prompts,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    steps=k,
                    max_new_tokens=max_new_tokens,
                    warmup=warmup,
                )
                all_rows.extend(llada_rows)

    csv_path = args.output_dir / "raw_metrics.csv"
    _write_csv(all_rows, csv_path)

    # Compute quality metrics (reload baseline if needed, or use cached)
    metric_rows: Dict[str, Dict[str, float]] = {}
    
    # Baseline metrics
    metrics = compute_metrics(
        model=baseline_quality_model,
        tokenizer=baseline_tokenizer,
        texts=prompts,
        max_length=max_new_tokens,
        compute_bertscore=args.compute_bertscore,
    )
    metric_rows["baseline"] = metrics
    
    # LLaDA metrics
    metrics = compute_metrics(
        model=target_model,
        tokenizer=target_tokenizer,
        texts=prompts,
        max_length=max_new_tokens,
        compute_bertscore=args.compute_bertscore,
    )
    metric_rows["llada"] = metrics

    metrics_path = args.output_dir / "quality_metrics.json"
    metrics_path.write_text(json.dumps(metric_rows, indent=2))

    print(f"Wrote latency/throughput rows -> {csv_path}")
    print(f"Quality metrics -> {metrics_path}")
    print(_summarize(all_rows))


if __name__ == "__main__":
    main()

