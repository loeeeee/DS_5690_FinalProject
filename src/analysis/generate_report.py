"""Headless report generator.

Expected responsibilities:
- Read raw metrics CSV/JSON from results/raw_data.
- Plot latency/throughput/memory and quality-vs-steps curves.
- Save figures to results/figures using matplotlib Agg backend.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Parsed model information from CSV model column."""

    full_name: str
    architecture: str  # "baseline" or "llada"
    steps: Optional[int] = None  # Diffusion steps for LLaDA, None for baseline


@dataclass
class ExperimentData:
    """Container for experiment data from a single directory."""

    experiment_name: str
    metrics_df: pd.DataFrame
    quality_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_info: Dict[str, ModelInfo] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for visualization."""

    model_name: str
    architecture: str
    steps: Optional[int]
    batch_size: int
    sequence_length: int
    avg_latency_ms: float
    avg_throughput_tps: float
    avg_peak_memory_gb: float
    quality_perplexity: Optional[float] = None


def parse_model_name(model_name: str) -> ModelInfo:
    """Parse model name to extract architecture and steps.

    Args:
        model_name: Model name from CSV (e.g., "baseline", "llada_steps_16")

    Returns:
        ModelInfo with parsed architecture and steps
    """
    if model_name == "baseline":
        return ModelInfo(full_name=model_name, architecture="baseline", steps=None)

    match = re.match(r"llada_steps_(\d+)", model_name)
    if match:
        steps = int(match.group(1))
        return ModelInfo(full_name=model_name, architecture="llada", steps=steps)

    logger.warning(f"Unknown model name format: {model_name}, treating as baseline")
    return ModelInfo(full_name=model_name, architecture="baseline", steps=None)


def load_experiment_data(experiment_dir: Path) -> Optional[ExperimentData]:
    """Load CSV and JSON data from an experiment directory.

    Args:
        experiment_dir: Path to experiment directory (e.g., results/raw_data/cpu_test)

    Returns:
        ExperimentData if successful, None if data is missing or malformed
    """
    csv_path = experiment_dir / "raw_metrics.csv"
    json_path = experiment_dir / "quality_metrics.json"

    if not csv_path.exists():
        logger.warning(f"Missing raw_metrics.csv in {experiment_dir}")
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"Empty CSV file in {experiment_dir}")
            return None

        quality_metrics: Dict[str, Dict[str, float]] = {}
        if json_path.exists():
            try:
                quality_metrics = json.loads(json_path.read_text())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse quality_metrics.json in {experiment_dir}: {e}")
        else:
            logger.info(f"No quality_metrics.json found in {experiment_dir}, continuing without quality data")

        model_info: Dict[str, ModelInfo] = {}
        for model_name in df["model"].unique():
            model_info[model_name] = parse_model_name(model_name)

        return ExperimentData(
            experiment_name=experiment_dir.name,
            metrics_df=df,
            quality_metrics=quality_metrics,
            model_info=model_info,
        )
    except Exception as e:
        logger.error(f"Failed to load data from {experiment_dir}: {e}")
        return None


def scan_experiment_directories(raw_data_dir: Path) -> List[ExperimentData]:
    """Scan results/raw_data/ for all experiment directories and load data.

    Args:
        raw_data_dir: Path to results/raw_data/ directory

    Returns:
        List of ExperimentData objects, one per valid experiment directory
    """
    if not raw_data_dir.exists():
        logger.error(f"Raw data directory does not exist: {raw_data_dir}")
        return []

    experiment_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    experiments: List[ExperimentData] = []
    for exp_dir in tqdm(experiment_dirs, desc="Loading experiments"):
        data = load_experiment_data(exp_dir)
        if data:
            experiments.append(data)
        else:
            logger.warning(f"Skipping {exp_dir.name} due to missing or invalid data")

    logger.info(f"Successfully loaded {len(experiments)} experiments")
    return experiments


def aggregate_metrics(experiments: List[ExperimentData]) -> List[AggregatedMetrics]:
    """Aggregate metrics across experiments for visualization.

    Args:
        experiments: List of ExperimentData objects

    Returns:
        List of AggregatedMetrics with averaged values per model/config
    """
    aggregated: List[AggregatedMetrics] = []

    for exp in experiments:
        df = exp.metrics_df.copy()

        if df.empty:
            logger.warning(f"Empty metrics dataframe for {exp.experiment_name}, skipping")
            continue

        for model_name, model_info in exp.model_info.items():
            model_df = df[df["model"] == model_name]

            if model_df.empty:
                logger.debug(f"No data for model {model_name} in {exp.experiment_name}")
                continue

            for (batch_size, seq_length), group_df in model_df.groupby(["batch_size", "sequence_length"]):
                if group_df.empty:
                    continue

                quality_perplexity = None
                if model_info.architecture == "baseline" and "baseline" in exp.quality_metrics:
                    quality_perplexity = exp.quality_metrics["baseline"].get("perplexity")
                elif model_info.architecture == "llada" and "llada" in exp.quality_metrics:
                    quality_perplexity = exp.quality_metrics["llada"].get("perplexity")

                peak_memory_gb = group_df["peak_memory_bytes"].mean() / (1024**3)
                if pd.isna(peak_memory_gb) or peak_memory_gb == 0:
                    peak_memory_gb = 0.0

                avg_latency = group_df["latency_ms"].mean()
                avg_throughput = group_df["throughput_tps"].mean()

                if pd.isna(avg_latency) or pd.isna(avg_throughput):
                    logger.warning(
                        f"NaN values detected for {model_name} B={batch_size} N={seq_length} in {exp.experiment_name}, skipping"
                    )
                    continue

                aggregated.append(
                    AggregatedMetrics(
                        model_name=model_name,
                        architecture=model_info.architecture,
                        steps=model_info.steps,
                        batch_size=int(batch_size),
                        sequence_length=int(seq_length),
                        avg_latency_ms=float(avg_latency),
                        avg_throughput_tps=float(avg_throughput),
                        avg_peak_memory_gb=float(peak_memory_gb),
                        quality_perplexity=float(quality_perplexity) if quality_perplexity is not None else None,
                    )
                )

    return aggregated


def plot_pareto_frontier(
    aggregated: List[AggregatedMetrics],
    output_path: Path,
    experiment_name: Optional[str] = None,
) -> None:
    """Plot Pareto frontier: Latency vs Quality (Perplexity).

    Args:
        aggregated: List of AggregatedMetrics
        output_path: Path to save figure
        experiment_name: Optional experiment name for title
    """
    baseline_data = [m for m in aggregated if m.architecture == "baseline" and m.quality_perplexity is not None]
    llada_data = [
        m for m in aggregated if m.architecture == "llada" and m.quality_perplexity is not None and m.steps is not None
    ]

    if not baseline_data and not llada_data:
        logger.warning("No quality data available for Pareto frontier plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline_data:
        baseline_latency = np.mean([m.avg_latency_ms for m in baseline_data])
        baseline_perplexity = baseline_data[0].quality_perplexity
        if baseline_perplexity is not None:
            ax.scatter(
                baseline_latency,
                baseline_perplexity,
                s=200,
                marker="*",
                color="blue",
                label="LLaMA Baseline",
                zorder=5,
            )

    if llada_data:
        llada_df = pd.DataFrame(
            [
                {
                    "steps": m.steps,
                    "latency": m.avg_latency_ms,
                    "perplexity": m.quality_perplexity,
                }
                for m in llada_data
                if m.steps is not None and m.quality_perplexity is not None
            ]
        )
        if not llada_df.empty:
            llada_df = llada_df.sort_values("steps")
            ax.plot(
                llada_df["latency"],
                llada_df["perplexity"],
                marker="o",
                linestyle="-",
                color="red",
                label="LLaDA",
                linewidth=2,
                markersize=8,
            )
            for _, row in llada_df.iterrows():
                ax.annotate(
                    f"K={int(row['steps'])}",
                    (row["latency"], row["perplexity"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.set_xlabel("End-to-End Latency ($L_{e2e}$) [ms]", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("Pareto Frontier: Latency vs Quality" + (f" - {experiment_name}" if experiment_name else ""), fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Pareto frontier plot to {output_path}")


def plot_throughput_bars(
    aggregated: List[AggregatedMetrics],
    output_path: Path,
    experiment_name: Optional[str] = None,
) -> None:
    """Plot throughput bar charts comparing models across batch sizes.

    Args:
        aggregated: List of AggregatedMetrics
        output_path: Path to save figure
        experiment_name: Optional experiment name for title
    """
    baseline_data = [m for m in aggregated if m.architecture == "baseline"]
    llada_data = [m for m in aggregated if m.architecture == "llada"]

    if not baseline_data and not llada_data:
        logger.warning("No data available for throughput bar chart, skipping")
        return

    batch_sizes = sorted(set(m.batch_size for m in aggregated))
    if not batch_sizes:
        logger.warning("No batch sizes found for throughput bar chart, skipping")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(batch_sizes))
    width = 0.35

    if baseline_data:
        baseline_throughput = []
        for bs in batch_sizes:
            bs_data = [m for m in baseline_data if m.batch_size == bs]
            avg = np.mean([m.avg_throughput_tps for m in bs_data]) if bs_data else 0
            baseline_throughput.append(avg)
        ax.bar(x - width / 2, baseline_throughput, width, label="LLaMA Baseline", color="blue", alpha=0.7)

    if llada_data:
        llada_steps = sorted(set(m.steps for m in llada_data if m.steps is not None))
        if llada_steps:
            for idx, steps in enumerate(llada_steps):
                steps_data = [m for m in llada_data if m.steps == steps]
                steps_throughput = []
                for bs in batch_sizes:
                    bs_data = [m for m in steps_data if m.batch_size == bs]
                    avg = np.mean([m.avg_throughput_tps for m in bs_data]) if bs_data else 0
                    steps_throughput.append(avg)
                offset = width / len(llada_steps) * (idx - (len(llada_steps) - 1) / 2)
                ax.bar(
                    x + offset,
                    steps_throughput,
                    width / len(llada_steps),
                    label=f"LLaDA K={steps}",
                    alpha=0.7,
                )

    ax.set_xlabel("Batch Size ($B$)", fontsize=12)
    ax.set_ylabel("Generation Throughput ($T_{gen}$) [tokens/sec]", fontsize=12)
    ax.set_title("Throughput Comparison by Batch Size" + (f" - {experiment_name}" if experiment_name else ""), fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved throughput bar chart to {output_path}")


def plot_memory_heatmap(
    aggregated: List[AggregatedMetrics],
    output_path: Path,
    experiment_name: Optional[str] = None,
) -> None:
    """Plot memory heatmap showing VRAM usage across sequence lengths and models.

    Args:
        aggregated: List of AggregatedMetrics
        output_path: Path to save figure
        experiment_name: Optional experiment name for title
    """
    sequence_lengths = sorted(set(m.sequence_length for m in aggregated))
    model_configs: List[Tuple[str, Optional[int]]] = []

    for m in aggregated:
        if m.architecture == "baseline":
            config = ("LLaMA Baseline", None)
        else:
            config = (f"LLaDA K={m.steps}", m.steps)
        if config not in model_configs:
            model_configs.append(config)

    heatmap_data = []
    row_labels = []
    for config_name, _ in model_configs:
        row_labels.append(config_name)
        row_data = []
        for seq_len in sequence_lengths:
            matching = [
                m
                for m in aggregated
                if (m.architecture == "baseline" and config_name == "LLaMA Baseline")
                or (m.architecture == "llada" and m.steps is not None and config_name == f"LLaDA K={m.steps}")
            ]
            matching = [m for m in matching if m.sequence_length == seq_len]
            avg_memory = np.mean([m.avg_peak_memory_gb for m in matching]) if matching else 0.0
            row_data.append(avg_memory)
        heatmap_data.append(row_data)

    if not heatmap_data or all(all(v == 0.0 for v in row) for row in heatmap_data):
        logger.warning("No memory data available for heatmap, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(model_configs) * 0.8)))
    sns.heatmap(
        heatmap_data,
        xticklabels=sequence_lengths,
        yticklabels=row_labels,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar_kws={"label": "Peak VRAM ($M_{peak}$) [GB]"},
        ax=ax,
    )
    ax.set_xlabel("Sequence Length ($N$)", fontsize=12)
    ax.set_ylabel("Model Configuration", fontsize=12)
    ax.set_title("Memory Usage Heatmap" + (f" - {experiment_name}" if experiment_name else ""), fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved memory heatmap to {output_path}")


def plot_quality_vs_steps(
    aggregated: List[AggregatedMetrics],
    output_path: Path,
    experiment_name: Optional[str] = None,
) -> None:
    """Plot quality (perplexity) vs diffusion steps for LLaDA.

    Args:
        aggregated: List of AggregatedMetrics
        output_path: Path to save figure
        experiment_name: Optional experiment name for title
    """
    llada_data = [
        m for m in aggregated if m.architecture == "llada" and m.steps is not None and m.quality_perplexity is not None
    ]

    if not llada_data:
        logger.warning("No LLaDA quality data available for quality vs steps plot, skipping")
        return

    steps_list = sorted(set(m.steps for m in llada_data if m.steps is not None))
    if not steps_list:
        logger.warning("No valid steps found for quality vs steps plot, skipping")
        return

    perplexities = []
    for steps in steps_list:
        steps_data = [m for m in llada_data if m.steps == steps and m.quality_perplexity is not None]
        if not steps_data:
            logger.warning(f"No quality data for steps={steps}, skipping")
            continue
        avg_perplexity = np.mean([m.quality_perplexity for m in steps_data])
        perplexities.append(avg_perplexity)

    if not perplexities:
        logger.warning("No valid perplexity values for quality vs steps plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_list[: len(perplexities)], perplexities, marker="o", linestyle="-", linewidth=2, markersize=10, color="red")
    ax.set_xlabel("Diffusion Steps ($K$)", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("Quality vs Diffusion Steps (LLaDA)" + (f" - {experiment_name}" if experiment_name else ""), fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved quality vs steps plot to {output_path}")


def plot_latency_vs_steps(
    aggregated: List[AggregatedMetrics],
    output_path: Path,
    experiment_name: Optional[str] = None,
) -> None:
    """Plot latency vs diffusion steps for LLaDA.

    Args:
        aggregated: List of AggregatedMetrics
        output_path: Path to save figure
        experiment_name: Optional experiment name for title
    """
    llada_data = [m for m in aggregated if m.architecture == "llada" and m.steps is not None]

    if not llada_data:
        logger.warning("No LLaDA data available for latency vs steps plot, skipping")
        return

    batch_sizes = sorted(set(m.batch_size for m in llada_data))
    sequence_lengths = sorted(set(m.sequence_length for m in llada_data))

    if not batch_sizes or not sequence_lengths:
        logger.warning("Insufficient data for latency vs steps plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    has_data = False
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            config_data = [m for m in llada_data if m.batch_size == batch_size and m.sequence_length == seq_length]
            if not config_data:
                continue

            steps_list = sorted(set(m.steps for m in config_data if m.steps is not None))
            if not steps_list:
                continue

            latencies = []
            for steps in steps_list:
                steps_data = [m for m in config_data if m.steps == steps]
                if not steps_data:
                    continue
                avg_latency = np.mean([m.avg_latency_ms for m in steps_data])
                latencies.append(avg_latency)

            if len(latencies) == len(steps_list) and len(latencies) > 0:
                label = f"B={batch_size}, N={seq_length}"
                ax.plot(steps_list, latencies, marker="o", linestyle="-", label=label, linewidth=2, markersize=8)
                has_data = True

    if not has_data:
        logger.warning("No valid data points for latency vs steps plot, skipping")
        plt.close(fig)
        return

    ax.set_xlabel("Diffusion Steps ($K$)", fontsize=12)
    ax.set_ylabel("End-to-End Latency ($L_{e2e}$) [ms]", fontsize=12)
    ax.set_title("Latency vs Diffusion Steps (LLaDA)" + (f" - {experiment_name}" if experiment_name else ""), fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved latency vs steps plot to {output_path}")


def plot_throughput_vs_batch_size(
    aggregated: List[AggregatedMetrics],
    output_path: Path,
    experiment_name: Optional[str] = None,
) -> None:
    """Plot throughput vs batch size line plot comparing models.

    Args:
        aggregated: List of AggregatedMetrics
        output_path: Path to save figure
        experiment_name: Optional experiment name for title
    """
    baseline_data = [m for m in aggregated if m.architecture == "baseline"]
    llada_data = [m for m in aggregated if m.architecture == "llada"]

    if not baseline_data and not llada_data:
        logger.warning("No data available for throughput vs batch size plot, skipping")
        return

    batch_sizes = sorted(set(m.batch_size for m in aggregated))
    sequence_lengths = sorted(set(m.sequence_length for m in aggregated))

    if not batch_sizes:
        logger.warning("No batch sizes found for throughput vs batch size plot, skipping")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    has_data = False

    if baseline_data:
        for seq_length in sequence_lengths:
            seq_data = [m for m in baseline_data if m.sequence_length == seq_length]
            if not seq_data:
                continue
            throughputs = []
            for bs in batch_sizes:
                bs_data = [m for m in seq_data if m.batch_size == bs]
                avg = np.mean([m.avg_throughput_tps for m in bs_data]) if bs_data else 0
                throughputs.append(avg)
            if any(t > 0 for t in throughputs):
                ax.plot(
                    batch_sizes,
                    throughputs,
                    marker="o",
                    linestyle="-",
                    label=f"LLaMA Baseline N={seq_length}",
                    linewidth=2,
                    markersize=8,
                )
                has_data = True

    if llada_data:
        llada_steps = sorted(set(m.steps for m in llada_data if m.steps is not None))
        for steps in llada_steps:
            for seq_length in sequence_lengths:
                config_data = [
                    m for m in llada_data if m.steps == steps and m.sequence_length == seq_length
                ]
                if not config_data:
                    continue
                throughputs = []
                for bs in batch_sizes:
                    bs_data = [m for m in config_data if m.batch_size == bs]
                    avg = np.mean([m.avg_throughput_tps for m in bs_data]) if bs_data else 0
                    throughputs.append(avg)
                if any(t > 0 for t in throughputs):
                    ax.plot(
                        batch_sizes,
                        throughputs,
                        marker="s",
                        linestyle="--",
                        label=f"LLaDA K={steps} N={seq_length}",
                        linewidth=2,
                        markersize=8,
                    )
                    has_data = True

    if not has_data:
        logger.warning("No valid data points for throughput vs batch size plot, skipping")
        plt.close(fig)
        return

    ax.set_xlabel("Batch Size ($B$)", fontsize=12)
    ax.set_ylabel("Generation Throughput ($T_{gen}$) [tokens/sec]", fontsize=12)
    ax.set_title("Throughput vs Batch Size" + (f" - {experiment_name}" if experiment_name else ""), fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved throughput vs batch size plot to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Generate visualization plots from experiment data")
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("results/raw_data"),
        help="Directory containing experiment subdirectories (default: results/raw_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/figures"),
        help="Directory to save generated figures (default: results/figures)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Process only a specific experiment directory name (default: process all)",
    )
    return parser


def main() -> None:
    """Main entry point for visualization script."""
    parser = build_parser()
    args = parser.parse_args()

    raw_data_dir = args.raw_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Scanning experiments in {raw_data_dir}")
    experiments = scan_experiment_directories(raw_data_dir)

    if not experiments:
        logger.error("No valid experiments found, exiting")
        return

    if args.experiment:
        experiments = [e for e in experiments if e.experiment_name == args.experiment]
        if not experiments:
            logger.error(f"Experiment '{args.experiment}' not found")
            return

    logger.info(f"Processing {len(experiments)} experiment(s)")

    generated_figures: List[Path] = []

    for exp in tqdm(experiments, desc="Generating visualizations"):
        logger.info(f"Processing experiment: {exp.experiment_name}")
        aggregated = aggregate_metrics([exp])

        if not aggregated:
            logger.warning(f"No aggregated metrics for {exp.experiment_name}, skipping")
            continue

        exp_output_dir = output_dir / exp.experiment_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        plot_functions = [
            (plot_pareto_frontier, "pareto_frontier.png"),
            (plot_throughput_bars, "throughput_by_batch_size.png"),
            (plot_memory_heatmap, "memory_heatmap.png"),
            (plot_quality_vs_steps, "quality_vs_steps.png"),
            (plot_latency_vs_steps, "latency_vs_steps.png"),
            (plot_throughput_vs_batch_size, "throughput_vs_batch_size.png"),
        ]

        for plot_func, filename in plot_functions:
            try:
                output_path = exp_output_dir / filename
                plot_func(aggregated, output_path, experiment_name=exp.experiment_name)
                if output_path.exists():
                    generated_figures.append(output_path)
            except Exception as e:
                logger.error(f"Failed to generate {filename} for {exp.experiment_name}: {e}")

    if len(experiments) > 1:
        logger.info("Generating combined visualizations across all experiments")
        all_aggregated = aggregate_metrics(experiments)
        if all_aggregated:
            combined_output_dir = output_dir / "combined"
            combined_output_dir.mkdir(parents=True, exist_ok=True)

            for plot_func, filename in plot_functions:
                try:
                    output_path = combined_output_dir / filename
                    plot_func(all_aggregated, output_path)
                    if output_path.exists():
                        generated_figures.append(output_path)
                except Exception as e:
                    logger.error(f"Failed to generate combined {filename}: {e}")

    logger.info(f"Generated {len(generated_figures)} figure(s)")
    print(f"\nVisualization Summary:")
    print(f"  Generated {len(generated_figures)} figure(s)")
    print(f"  Output directory: {output_dir}")
    for fig_path in generated_figures:
        print(f"    - {fig_path.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
