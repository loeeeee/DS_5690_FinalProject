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

Local dev (ROCm, nix-shell)
- Enter shell (Python 3.13, ROCm 6.4, RX 7900 XT): `nix-shell nix-shell.nix`
- Verify ROCm torch:
  - `python - <<'PY'`
  - `import torch, platform`
  - `print("torch", torch.__version__)`
  - `print("hip version", torch.version.hip)`
  - `print("cuda available", torch.cuda.is_available())`
  - `print("device", torch.cuda.get_device_name(0))`
  - `print("arch", torch.cuda.get_device_capability())`
  - `print("platform", platform.platform())`
  - `PY`
- Run benchmark locally inside shell: `python -m src.main --config config/experiments.yaml --output_dir results/raw_data`
- Environment notes: `PYTORCH_HIP_ARCH=gfx1100` and `HSA_OVERRIDE_GFX_VERSION=11.0.0` are set for RX 7900 XT; torchWithRocm uses ROCm 6.4.

Pre-download models and data
- From inside `nix-shell`, fetch caches ahead of time:  
  `python scripts/download_assets.py --config config/experiments.yaml`  
  Add `--hf-token $HF_TOKEN` for gated LLaMA 3, `--model-cache-dir /fast/hf` or `--dataset-cache-dir /fast/hf_datasets` to steer cache locations. Use `--extra-model-id` to pull additional repos. `--local-files-only` will only validate existing cache.

Quick start (headless CUDA / SLURM)
- Configure experiments in `config/experiments.yaml` (default LLaMA: `meta-llama/Meta-Llama-3-8B`, LLaDA: `GSAI-ML/LLaDA-8B-Base`).
- Run on cluster: `sbatch jobs/benchmark_gpu.sbatch`
- Optional overrides: `--steps 64 --batch_size 4 --max_prompts 16 --compute_bertscore`.
- Outputs: latency/throughput CSV at `results/raw_data/raw_metrics.csv` and quality metrics JSON at `results/raw_data/quality_metrics.json`.

Mini test (for GPU validation)
- Run mini test: `python src/main.py --config config/experiments_mini.yaml --output_dir results/raw_data/smoke --max_prompts 4 --batch_size 1 --steps 16`
- Note: ROCm (AMD GPUs) may have compatibility issues with model.generate(). The code is designed for CUDA (NVIDIA GPUs) and should work correctly on CUDA systems.

Notes
- Prompts default to Wikitext-103 validation text; supply `--prompt_file` to use custom prompts.
- Profiling uses CUDA events when available; falls back to CPU timers otherwise.
- Generation uses BF16 by default; adjust `precision` in `config/experiments.yaml` if required.
