Project intent (from user request and .clinerules):
- Benchmark LLaDA (8B diffusion LLM) vs LLaMA 3 8B on latency, throughput, and memory.
- Address paper rejection by providing rigorous inference efficiency analysis and quality vs computation trade-off.
- Target headless/HPC execution; no interactive notebooks; use SLURM submission.

What success looks like:
- Reproducible, scripted benchmarks runnable via sbatch on cluster GPUs.
- Clear metrics definitions (TTFT/ITL for LLaMA, wall-clock per sequence for LLaDA, tokens/sec, peak VRAM).
- Quality vs steps sweep for LLaDA to find cross-over with LLaMA quality.

Constraints & environment:
- Python 3.13; use shell.nix/venv under NixOS locally, CUDA on HPC.
- GPUs: prefer RTX 2080 Ti on cluster (`nvidia_geforce_rtx_2080_ti`), ROCm locally if needed.
- Non-interactive reporting (matplotlib Agg).

Planned artifacts (scaffolded here, to be filled in later):
- jobs/* for sbatch + env setup.
- src/* modular CLI benchmark harness.
- config/experiments.yaml for sweeps.
- results/* for raw data and figures; logs/slurm for job outputs.


