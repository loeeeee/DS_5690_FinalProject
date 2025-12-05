"""CLI entrypoint for headless benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, Iterable, List

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--max_prompts", type=int, default=None, help="Override prompt count (takes precedence over config dataset_size)")
    parser.add_argument("--compute_bertscore", action="store_true", help="Also compute BERTScore (slower)")
    return parser


def _get_prompt_count_from_size(dataset_size: str | None) -> int:
    """Map dataset size preset to prompt count."""
    size_map = {
        "small": 8,
        "medium": 32,
        "large": 128,
    }
    if dataset_size is None:
        return 32  # Default to medium
    dataset_size_lower = dataset_size.lower()
    if dataset_size_lower not in size_map:
        logger.warning(f"Unknown dataset_size '{dataset_size}', defaulting to medium (32 prompts)")
        return 32
    return size_map[dataset_size_lower]


def _load_prompts(
    prompt_file: Path | None,
    max_prompts: int | None,
    seq_length: int,
    dataset_size: str | None = None,
) -> List[str]:
    """Load prompts from file or dataset.
    
    Args:
        prompt_file: Optional file path to load prompts from
        max_prompts: CLI override for prompt count (takes precedence over dataset_size)
        seq_length: Maximum sequence length to truncate prompts
        dataset_size: Size preset ("small", "medium", "large") from config
    """
    # Determine prompt count: CLI override takes precedence
    if max_prompts is not None:
        target_count = max_prompts
        logger.info(f"Using CLI override: {target_count} prompts")
    else:
        target_count = _get_prompt_count_from_size(dataset_size)
        logger.info(f"Using dataset_size '{dataset_size}': {target_count} prompts")
    
    if prompt_file:
        lines = prompt_file.read_text().strip().splitlines()
        return lines[:target_count]
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    prompts: List[str] = []
    for row in dataset:
        text = row["text"].strip()
        if not text:
            continue
        prompts.append(text[:seq_length])
        if len(prompts) >= target_count:
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


def _save_results_incremental(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Save results incrementally, appending to CSV if it exists."""
    if not rows:
        return
    _ensure_output_dir(output_path.parent)
    file_exists = output_path.exists()
    with output_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
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
    logger.info(f"Running warmup for {name}: {warmup} iterations")
    if warmup > 1:
        warmup_iterable = tqdm(range(warmup), desc=f"Warmup {name}", leave=False)
    else:
        warmup_iterable = range(warmup)
    for warmup_idx in warmup_iterable:
        for batch in _batched(warmup_prompts, batch_size):
            try:
                logger.debug(f"Warmup iteration {warmup_idx+1}/{warmup}, batch size: {len(batch)}")
                _ = wrapper.generate_batch(batch, steps=steps, max_new_tokens=max_new_tokens)
                logger.debug(f"Warmup iteration {warmup_idx+1} completed")
            except Exception as e:
                logger.error(f"Warmup failed at iteration {warmup_idx+1}: {e}")
                raise

    batches = list(_batched(prompts, batch_size))
    logger.info(f"Running benchmark for {name}: {len(batches)} batches")
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"Processing batches {name}", leave=False)):
        logger.info(f"Processing batch {batch_idx+1}/{len(batches)} for {name}")
        logger.info(f"Batch {batch_idx+1} details: batch_size={len(batch)}, steps={steps}, max_new_tokens={max_new_tokens}")
        result: GenerationBatchResult
        try:
            logger.info(f"Batch {batch_idx+1}: About to call profile_generation")
            result, profile = profile_generation(
                wrapper.generate_batch,
                batch,
                steps=steps,
                max_new_tokens=max_new_tokens,
            )
            logger.info(f"Batch {batch_idx+1}: profile_generation completed successfully")
        except Exception as e:
            logger.error(f"Batch {batch_idx+1} failed for {name}: {e}", exc_info=True)
            raise
        logger.info(f"Batch {batch_idx+1}: Processing results - token_counts: {result.token_counts}")
        total_tokens = sum(result.token_counts)
        throughput = total_tokens / (profile["latency_ms"] / 1000.0) if profile["latency_ms"] > 0 else 0.0
        logger.info(f"Batch {batch_idx+1}: Computed metrics - total_tokens={total_tokens}, throughput={throughput:.2f} tok/s")
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
        logger.info(f"Batch {batch_idx+1}: Results appended to rows, moving to next batch")
    logger.info(f"Completed all {len(batches)} batches for {name}")
    return rows


def _setup_logging(output_dir: Path) -> None:
    """Setup logging to both console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = output_dir / "benchmark.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def _create_signal_handler(
    all_rows: List[Dict[str, Any]],
    csv_path: Path,
    partial_csv_path: Path,
    logger: logging.Logger,
) -> Callable[[int, FrameType | None], None]:
    """Create a signal handler that saves results on interrupt."""
    interrupted = False
    
    def signal_handler(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        if interrupted:
            logger.warning("Received second interrupt signal, forcing exit")
            sys.exit(1)
        
        interrupted = True
        logger.warning(f"Received interrupt signal ({signum}), saving current results...")
        
        try:
            # Save all accumulated results
            if all_rows:
                _write_csv(all_rows, csv_path)
                logger.info(f"Saved {len(all_rows)} rows to {csv_path}")
            else:
                # If no results yet, check if partial file exists
                if partial_csv_path.exists():
                    logger.info(f"Partial results exist at {partial_csv_path}")
                else:
                    logger.warning("No results to save")
            
            logger.info("Results saved. Exiting gracefully.")
        except Exception as e:
            logger.error(f"Error saving results on interrupt: {e}", exc_info=True)
        
        sys.exit(0)
    
    return signal_handler


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Ensure output directory exists before setting up logging
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to both console and file
    _setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)

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

    dataset_config = config.get("dataset", {})
    dataset_size = dataset_config.get("dataset_size")
    prompts = _load_prompts(args.prompt_file, args.max_prompts, max(seq_lengths), dataset_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = experiment.get("precision", "bfloat16")
    
    # GPU verification and logging
    if device == "cuda" and torch.cuda.is_available():
        logger.info("GPU detected and available")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"PyTorch version: {torch.__version__}")
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU total memory: {memory_total:.2f} GB")
    else:
        logger.warning("CUDA not available, falling back to CPU")
        logger.info(f"Device: {device}")
    
    all_rows: List[Dict[str, Any]] = []
    partial_csv_path = args.output_dir / "raw_metrics_partial.csv"
    csv_path = args.output_dir / "raw_metrics.csv"

    # Setup signal handlers for graceful interruption
    signal_handler = _create_signal_handler(all_rows, csv_path, partial_csv_path, logger)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers registered for graceful interruption")

    # Load baseline model first, run all baseline experiments, then unload
    from models.model_factory import _load_auto_model
    from models.wrappers import AutoRegressiveWrapper
    logger.info(f"Loading baseline model: {baseline_id}")
    baseline_model, baseline_tokenizer = _load_auto_model(baseline_id, device, precision)
    # Verify model is on correct device
    model_device = next(baseline_model.parameters()).device
    logger.info(f"Baseline model loaded on device: {model_device}")
    if device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"GPU memory after baseline load - allocated: {memory_allocated:.2f} GB, reserved: {memory_reserved:.2f} GB")
    baseline_wrapper = AutoRegressiveWrapper(baseline_model, baseline_tokenizer)

    total_baseline_configs = len(seq_lengths) * len(batch_sizes)
    baseline_pbar = tqdm(total=total_baseline_configs, desc="Baseline experiments", unit="config")
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
            # Save incrementally after each configuration
            _save_results_incremental(baseline_rows, partial_csv_path)
            baseline_pbar.update(1)
    baseline_pbar.close()
    
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
    logger.info(f"Loading target model: {target_id}")
    target_model, target_tokenizer = _load_auto_model(target_id, device, precision)
    # Verify model is on correct device
    model_device = next(target_model.parameters()).device
    logger.info(f"Target model loaded on device: {model_device}")
    if device == "cuda" and torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"GPU memory after target load - allocated: {memory_allocated:.2f} GB, reserved: {memory_reserved:.2f} GB")
    target_wrapper = DiffusionLikeWrapper(target_model, target_tokenizer)

    total_llada_configs = len(seq_lengths) * len(batch_sizes) * len(diffusion_steps)
    llada_pbar = tqdm(total=total_llada_configs, desc="LLaDA experiments", unit="config")
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
                # Save incrementally after each configuration
                _save_results_incremental(llada_rows, partial_csv_path)
                llada_pbar.update(1)
    llada_pbar.close()

    # Final complete save
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

