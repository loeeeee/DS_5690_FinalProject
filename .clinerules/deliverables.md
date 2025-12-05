This structure prioritizes **reproducibility** (critical for refuting a paper rejection) and **visual analysis** (critical for demonstrating the trade-off).

We need **headless execution**, robust **logging**, and **batch scheduling**. Interactive tools like Jupyter are useless here because the compute nodes usually don't have display servers (X11) attached, and you can't manually guide the execution.

Here is the revised list of deliverables, strictly tailored for a **headless SLURM cluster environment**.

### LLaDA-Inference-Benchmark/ (HPC Edition)

#### 1\. Cluster Orchestration (The "Sbatch" Layer)

This is the interface between you and the cluster scheduler.

  * **`jobs/benchmark_gpu.sbatch`**: The primary submission script. It must:
      * Request specific resources (e.g., `#SBATCH --gres=gpu:a100:1`).
      * Load environment modules (e.g., `module load Anaconda3`, `module load cuda/12.x`).
      * Execute the python runner.
  * **`jobs/environment_setup.sh`**: A shell script to create the non-interactive Python environment (venv/conda) on the login node, installing dependencies so the compute node is ready.

#### 2\. Source Code (Headless & Modular)

Refactored to run start-to-finish without human input.

  * **`src/main.py`**: The CLI entry point using `argparse`. It takes arguments like `--model llada`, `--steps 64`, `--batch-size 32` so the `.sbatch` file can sweep through configurations easily.
  * **`src/models/`**:
      * `model_factory.py`: Logic to load LLaMA vs LLaDA based on config.
      * `wrappers.py`: Standardized API for `generate()` so the benchmark loop doesn't care which model is running.
  * **`src/evaluation/`**:
      * `metrics.py`: Headless calculation of Perplexity/BERTScore.
      * `profiling.py`: `torch.cuda` event timers and memory tracking hooks.
  * **`src/analysis/generate_report.py`**: **(Replaces the Notebook)**. A pure Python script that reads the CSV logs and uses `matplotlib` (with the non-interactive 'Agg' backend) to save PNGs/PDFs to disk automatically.

#### 3\. Configuration & dependencies

  * **`config/experiments.yaml`**: Defines the "Grid Search" parameters (e.g., Sequence Lengths: [128, 1024], Steps: [32, 64]).
  * **`requirements.txt`**: Standard dependencies (torch, transformers, pandas).

#### 4\. Data Artifacts (Structured for Automation)

  * **`logs/slurm/`**: Storage for standard output/error files (`%j.out`, `%j.err`) from the scheduler.
  * **`results/raw_data/`**: Timestamped CSVs/JSONs containing the raw metrics (e.g., `2025-10-25_llada_steps64.csv`).
  * **`results/figures/`**: The final output folder where `generate_report.py` dumps the Pareto curves and bar charts.

#### 5\. Documentation

  * **`README.md`**: Updated to explain *how* to submit the jobs (e.g., `sbatch jobs/benchmark_gpu.sbatch`).
  * **`docs/reproducibility.md`**: Specific instructions on which CUDA version and container/venv was used on the cluster.

-----

### Key Technical Consideration for Clusters

Since you cannot open a window to look at a plot on a compute node, your `generate_report.py` **must** include this specifically at the top:

```python
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend (no X11 required)
import matplotlib.pyplot as plt
```

Would you like me to draft the **`jobs/benchmark_gpu.sbatch`** file first, so you have a template that correctly handles module loading and python execution?