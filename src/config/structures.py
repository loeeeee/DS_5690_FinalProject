"""Dataclasses for experiment variables and metric specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class IndependentVariables:
    """Sweepable knobs for experiments."""

    architecture_levels: Tuple[str, ...] = (
        "Meta-LLaMA-3-8B",
        "LLaDA-8B",
    )
    diffusion_steps: Tuple[int, ...] = (16, 32, 64, 128, 256)
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16, 32, 64)
    sequence_lengths: Tuple[int, ...] = (128, 512, 1024, 2048)
    notes: str = (
        "Diffusion steps apply only to LLaDA. Sequence length uses a fixed "
        "input/output ratio."
    )


@dataclass
class ControlVariables:
    """Constants that must remain fixed across runs."""

    hardware_node_type: str = "Single ACCRE Compute Node"
    gpu_model: str = "NVIDIA A100 80GB"
    cpu_exclusive: bool = True
    precision: str = "bfloat16"
    output_token_policy: str = (
        "Force identical output token count across models "
        "(LLaMA: generate exact tokens, LLaDA: mask exact tokens)."
    )
    software_stack: Tuple[str, ...] = (
        "Python 3.13.2",
        "PyTorch 2.4+",
        "CUDA 12.6",
        "FlashAttention-2 enabled for both or neither",
    )


@dataclass
class ProceduralVariables:
    """Protocol-related variables for measurement."""

    warmup_iterations: int = 3
    sampling_temperatures: Tuple[float, ...] = (0.0, 0.1)
    notes: str = (
        "Discard warmup passes to avoid JIT/allocator noise. "
        "Use deterministic or near-deterministic sampling for reproducibility."
    )


@dataclass
class ExperimentDesign:
    """Complete experiment variable bundle."""

    independent: IndependentVariables = field(default_factory=IndependentVariables)
    control: ControlVariables = field(default_factory=ControlVariables)
    procedural: ProceduralVariables = field(default_factory=ProceduralVariables)


@dataclass
class LatencyMetrics:
    end_to_end_description: str = (
        "Total wall-clock from input ingestion to final token/step completion."
    )
    inter_token_description: str = (
        "Average time between emitted tokens (autoregressive only)."
    )
    inter_token_applicability: Tuple[str, ...] = ("LLaMA",)


@dataclass
class ThroughputMetrics:
    generation_formula: str = "T_gen = (B * N_out) / L_e2e"
    notes: str = (
        "Measures tokens-per-second across batch; useful for high batch sizes."
    )


@dataclass
class MemoryMetrics:
    peak_vram_description: str = (
        "Peak GPU memory during generation (weights, activations, caches)."
    )
    llamma_bottleneck: str = "KV cache grows with sequence length (O(N))."
    llada_bottleneck: str = "Activation map dominates; depends on full sequence."


@dataclass
class EfficiencyMetrics:
    steps_to_quality_parity: str = (
        "Minimum diffusion steps where LLaDA quality meets/exceeds LLaMA."
    )
    speedup_formula: str = (
        "Speedup = L_e2e(LLaMA) / L_e2e(LLaDA at parity steps)"
    )
    interpretation: str = (
        "Speedup > 1 implies LLaDA is viable at matched quality; "
        "otherwise diffusion is inefficient."
    )


@dataclass
class MetricSuite:
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)

