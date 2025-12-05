# LLaDA vs LLaMA Inference Benchmark (Headless/HPC Scaffold)

Purpose
- Benchmark LLaDA (8B diffusion) vs LLaMA 3 8B on latency, throughput, and memory.
- Provide quality vs computation trade-off via diffusion step sweeps.
- Run headlessly via SLURM; no notebooks required.

Repo layout (scaffold)
- `docs-vibe/`: intent/setup notes (update before/after coding).
- `jobs/`: sbatch + env bootstrap for cluster runs.
- `config/experiments.yaml`: grid definitions (steps, batch sizes, seq lengths, hardware notes).
- `src/`: CLI benchmark harness (models, evaluation, analysis).
- `logs/slurm/`: SLURM stdout/err.
- `results/raw_data/`: metrics CSV/JSON.
- `results/figures/`: generated plots (matplotlib Agg).

Quick start (to be finalized)
- Prepare environment: run `jobs/environment_setup.sh` on login node.
- Submit benchmark: `sbatch jobs/benchmark_gpu.sbatch`.
- Collect outputs: check `logs/slurm/` for job logs, `results/raw_data/` for metrics, `results/figures/` for plots.

Next steps
- Fill config values, implement CLI, metrics, profiling, and report generation.
- Update docs-vibe and this README after each task per `.clinerules`.
