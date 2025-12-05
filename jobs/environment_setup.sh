#!/usr/bin/env bash
# One-time environment bootstrap on login node for Python 3.13.
set -euo pipefail

module purge
setup_accre_software_stack
module load python/3.13.2 scipy-stack/2025a cuda/12.6

ENV_PATH="${HOME}/llada_bench_env"

python -m venv "${ENV_PATH}"
source "${ENV_PATH}/bin/activate"

pip install --no-index --upgrade pip || true
pip install --no-index -r requirements.txt || pip install -r requirements.txt

echo "Env ready at ${ENV_PATH}. Activate with: source ${ENV_PATH}/bin/activate"

