Setup notes (to be updated after actual installs):
- Use `shell.nix` locally; on cluster create venv via `jobs/environment_setup.sh`.
- Install torch/transformers/accelerate/pandas/matplotlib; prefer CUDA wheels on HPC, ROCm locally.
- Force matplotlib non-interactive backend (`Agg`) in any plotting script.
- Always activate environment inside sbatch before running `src/main.py`.
- Data/model paths: consider copying weights to `/tmp/$USER` in sbatch for faster IO.

Usage skeleton (will be finalized later):
- Submit benchmark: `sbatch jobs/benchmark_gpu.sbatch`
- Outputs: SLURM logs -> `logs/slurm/`; metrics CSV -> `results/raw_data/`; figures -> `results/figures/`.

Open items to fill in later:
- Exact package versions and hashes.
- Hardware inventory per site (ACCRE vs local).
- Model source locations (paths or HF repo IDs).


