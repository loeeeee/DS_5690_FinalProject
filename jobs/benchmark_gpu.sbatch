#!/bin/bash
#SBATCH --job-name=llada_vs_llama
#SBATCH --account=p_dsi_acc
#SBATCH --partition=batch_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --gres=gpu:nvidia_geforce_rtx_2080_ti:1

# Headless benchmarking entrypoint. Assumes environment is built via jobs/environment_setup.sh.

set -euo pipefail

module purge
setup_accre_software_stack
module load python/3.13.2 scipy-stack/2025a cuda/12.6

source ~/llada_bench_env/bin/activate

echo "Job started on $(hostname) at $(date)"
echo "GPU allocated:"
nvidia-smi --query-gpu=name,memory.total --format=csv

python src/main.py \
  --config config/experiments.yaml \
  --output_dir results/raw_data

echo "Job finished at $(date)"

