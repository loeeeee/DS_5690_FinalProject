"""CLI entrypoint for headless benchmarking.

Expected behavior (to be implemented):
- Parse args for model selection, steps, batch size, config path, output dir.
- Instantiate models via src.models.model_factory.
- Run benchmark loop with profiling hooks from src.evaluation.profiling.
- Write metrics to results/raw_data and print concise summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLaDA vs LLaMA benchmarking harness")
    parser.add_argument("--model_name", required=True, help="Target model identifier (e.g., llada-8b)")
    parser.add_argument("--baseline_name", required=True, help="Baseline model identifier (e.g., llama-3-8b)")
    parser.add_argument("--config", required=True, type=Path, help="Path to experiments.yaml")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory for raw metric outputs")
    parser.add_argument("--steps", type=int, help="Override diffusion steps for LLaDA")
    parser.add_argument("--batch_size", type=int, help="Override batch size for sweep/debug")
    parser.add_argument("--prompt_file", type=Path, help="Optional prompt file for reproducible inputs")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Placeholder: hook up logging, model factory, benchmark runner, and writers.
    parser.error(
        "Implementation pending: wire model loading, benchmarking, logging, and output writing."
    )


if __name__ == "__main__":
    main()

