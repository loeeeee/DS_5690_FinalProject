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

Quick start (headless)
- Configure experiments in `config/experiments.yaml` (default LLaMA: `meta-llama/Meta-Llama-3-8B`, LLaDA: `GSAI-ML/LLaDA-8B-Base`).
- Run locally: `python -m src.main --config config/experiments.yaml --output_dir results/raw_data`.
- Optional overrides: `--steps 64 --batch_size 4 --max_prompts 16 --compute_bertscore`.
- Outputs: latency/throughput CSV at `results/raw_data/raw_metrics.csv` and quality metrics JSON at `results/raw_data/quality_metrics.json`.

Notes
- Prompts default to Wikitext-103 validation text; supply `--prompt_file` to use custom prompts.
- Profiling uses CUDA events when available; falls back to CPU timers otherwise.
- Generation uses BF16 by default; adjust `precision` in `config/experiments.yaml` if required.
