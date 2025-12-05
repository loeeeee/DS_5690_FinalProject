This guide details how to execute your **LLaDA vs. LLaMA** benchmark on the ACCRE cluster using `sbatch`. It is tailored to your requirement for **Python 3.13** and non-interactive execution.

### 1\. The ACCRE "Mental Model"

To run successfully on ACCRE, you must understand three specific layers:

1.  **The Partition Layer:** You must submit to the `batch_gpu` partition using a specific account suffix (`_acc`).
2.  **The Software Layer:** ACCRE uses `Lmod`. You must explicitly load the software stack before your script runs.
3.  **The Hardware Layer (GRES):** You cannot request "any GPU." You must request a specific architecture (e.g., `nvidia_a100_80gb`).

-----

### 2\. Environment Setup (One-Time)

Before submitting jobs, create your Python 3.13 virtual environment on the login node.

```bash
# 1. Initialize the ACCRE software stack
setup_accre_software_stack

# 2. Load Python 3.13 (as per your requirement)
# Note: "scipy-stack" provides optimized numpy/pandas
module load python/3.13.2 scipy-stack/2025a

# 3. Create the virtual environment in your home or data directory
python -m venv ~/llada_bench_env

# 4. Activate and install dependencies
source ~/llada_bench_env/bin/activate

# 5. Install libraries (Transformers, Torch, etc.)
# Use --no-index to prefer pre-compiled wheels from the ACCRE/Alliance wheelhouse for speed
pip install --no-index --upgrade pip
pip install --no-index torch torchvision transformers accelerate
# If a package isn't in the wheelhouse, pip will fail. Run it again without --no-index:
pip install vllm
```

-----

### 3\. The Sbatch Script (`benchmark.sbatch`)

Create this file in your project root. This script requests an **NVIDIA A100 (80GB)** to ensure you have enough VRAM for the 8B models and long context windows.

```bash
#!/bin/bash
#SBATCH --job-name=llada_vs_llama    # Job name shown in squeue
#SBATCH --mail-user=your_email@vanderbilt.edu
#SBATCH --mail-type=FAIL,END         # Email you when job finishes or fails
#SBATCH --account=your_group_acc     # IMPORTANT: Must end in _acc for batch_gpu
#SBATCH --partition=batch_gpu        # The required partition for GPU jobs
#SBATCH --nodes=1                    # Run on a single node
#SBATCH --ntasks=1                   # Single task (process)
#SBATCH --cpus-per-task=8            # CPU cores for data loading/preprocessing
#SBATCH --mem=32G                    # System RAM (not GPU VRAM)
#SBATCH --time=04:00:00              # Time limit (HH:MM:SS)
#SBATCH --output=logs/%x_%j.out      # Standard Output file (%j = job ID)
#SBATCH --error=logs/%x_%j.err       # Standard Error file

# CRITICAL: Request specific GPU architecture. 
# Options: nvidia_a100_80gb, nvidia_rtx_a6000, nvidia_geforce_rtx_2080_ti
#SBATCH --gres=gpu:nvidia_a100_80gb:1

# ---------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------

# 1. Load the Software Stack (Clean environment)
module purge
setup_accre_software_stack
module load python/3.13.2 scipy-stack/2025a cuda/12.6

# 2. Activate your Virtual Environment
source ~/llada_bench_env/bin/activate

# 3. Debugging Info (Optional but recommended)
echo "Job started on $(hostname) at $(date)"
echo "GPU allocated:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 4. Run the Python Benchmark Harness
# We assume the script accepts CLI args for headless execution
python src/main.py \
    --model_name "llada-8b" \
    --steps 64 \
    --batch_size 32 \
    --output_dir "results/raw_data"

echo "Job finished at $(date)"
```

### 4\. Key ACCRE-Specific Directives

| Directive | Explanation |
| :--- | :--- |
| **`#SBATCH --account=...`** | For GPU jobs, this **must** be your group name followed by `_acc` (e.g., `accre_lab_acc`). Use the command `slurm_resources` to verify your exact account name. |
| **`#SBATCH --gres=gpu:...`** | ACCRE rejects generic requests. You must specify the model. For your project: <br>`nvidia_a100_80gb`: Best for performance.<br>`nvidia_rtx_a6000`: Good alternative (48GB VRAM). |
| **`setup_accre_software_stack`** | This command initializes the Lmod system. Without it, `module load` may fail or load outdated versions. |

### 5\. Managing Your Job

**Submit the job:**

```bash
sbatch jobs/benchmark.sbatch
# Output: Submitted batch job 12345678
```

**Check status:**

```bash
squeue -u your_username
# Look for 'PD' (Pending) or 'R' (Running)
```

**Check GPU usage (while running):**
If you want to see if your code is actually using the GPU while the job is running, you can SSH into the compute node listed in `squeue`.

```bash
ssh gpu0123  # Replace with the node name from squeue
nvidia-smi   # Check utilization
```

### Pro Tip: The `/tmp` Optimization

ACCRE compute nodes have local SSDs mounted at `/tmp`. Reading data/models from network storage (`/home` or `/data`) is slow and can bottleneck GPU inference.

**Recommended modification to your sbatch script:**

```bash
# Copy model weights to local /tmp for faster loading
echo "Copying model to local SSD..."
mkdir -p /tmp/$USER/models
cp -r /data/your_group/models/llada-8b /tmp/$USER/models/

# Point your script to the local path
python src/main.py --model_path "/tmp/$USER/models/llada-8b" ...
```