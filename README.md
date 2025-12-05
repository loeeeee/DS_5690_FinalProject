# Inference Efficiency Analysis of LLaDA: A Comparative Benchmark Against LLaMA 3

## Abstract

This study addresses a critical gap in the evaluation of LLaDA (8B language diffusion model) by providing a rigorous, reproducible benchmark comparing its inference efficiency against LLaMA 3 8B. We empirically quantify the quality vs. computation trade-off through systematic evaluation of **latency**, **throughput**, and **memory consumption**. Our framework enables headless execution on HPC clusters via SLURM to isolate architectural differences from implementation optimizations. Preliminary results indicate a significant quality-efficiency trade-off: at low diffusion step counts, LLaDA fails to achieve usable quality, necessitating a determination of the exact step count ($S_{parity}$) required to match autoregressive baselines.

## 1\. Introduction

The LLaDA paper was initially rejected for lacking sufficient inference efficiency analysis. This project fills that gap. While diffusion models offer potential advantages in parallelization and batch processing, the computational cost of iterative denoising raises fundamental questions about their efficiency relative to established autoregressive models like LLaMA.

This study systematically evaluates these trade-offs. We aim to identify the practical viability of LLaDA for real-world applications by determining the specific diffusion step count required to achieve quality parity with LLaMA 3 8B.

## 2\. Related Work

  * **Diffusion Models for Language:** LLaDA adapts diffusion principles—successful in image generation—to text. While this allows for non-autoregressive parallel generation, the cost of iterative refinement is unknown.
  * **Inference Metrics:** Standard metrics like Time-to-First-Token (TTFT) are designed for autoregressive models. We adapt these to fairly compare LLaMA's sequential generation against LLaDA's "all-at-once" refinement.
  * **Quality-Efficiency Trade-offs:** Unlike autoregressive models where cost is fixed per token, diffusion models allow trading quality for speed by reducing steps. Understanding this curve is critical for deployment.

## 3\. Methodology

We define metrics mathematically to ensure reproducibility across fundamentally different architectures.

### 3.1 Latency Metrics

  * **End-to-End Latency ($L_{e2e}$)**: The primary common denominator.
    $$L_{e2e} = t_{end} - t_{start}$$

      * *LLaMA:* Time from input until `<EOS>`.
      * *LLaDA:* Time until final diffusion step completes.

  * **Inter-Token Latency ($L_{itl}$)** (LLaMA Only):
    $$L_{itl} = \frac{L_{e2e} - L_{prefill}}{N_{tokens} - 1}$$

      * *Relevance:* LLaDA has no $L_{itl}$. This metric highlights the user experience difference between "streaming" (LLaMA) and "waiting" (LLaDA).

### 3.2 Throughput Metrics

  * **Generation Throughput ($T_{gen}$)**:
    $$T_{gen} = \frac{B \times N_{out}}{L_{e2e}}$$
      * *Hypothesis:* LLaDA may show superior throughput at high batch sizes ($B$) due to parallelization, offsetting its latency cost.

### 3.3 Efficiency Trade-off (The Critical Metric)

  * **Steps-to-Quality Parity ($S_{parity}$)**: The minimum diffusion steps ($k$) required for LLaDA to match LLaMA 3's Perplexity ($Q_{llama}$).
    $$S_{parity} = \min \{ k \mid f_{llada}(k) \geq Q_{llama} \}$$

    We then calculate the **Speedup Factor**:
    $$\text{Speedup Factor} = \frac{L_{e2e}(\text{LLaMA})}{L_{e2e}(\text{LLaDA at } S_{parity})}$$

### 3.4 Measurement Protocol

To handle asynchronous CUDA execution, we strictly use `torch.cuda.Event` rather than Python time modules.

```python
start_event.record()
# ... model generation ...
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
```

## 4\. Experimental Setup

We utilize a controlled grid search to isolate the impact of architecture.

**Hardware & Software (Control Variables):**

  * **GPU:** NVIDIA A100 (80GB)
  * **Precision:** `bfloat16` (BF16)
  * **Stack:** Python 3.13, PyTorch 2.4+, CUDA 12.6
  * **Dataset:** Wikitext-103 (Validation split), packed to fill context.

**Experimental Grid (Independent Variables):**

| Variable | Configuration / Range | Rationale |
| :--- | :--- | :--- |
| **Model** | LLaMA 3 8B vs. LLaDA 8B | Compare Autoregressive vs. Diffusion. |
| **Steps ($K$)** | `[16, 32, 64, 128, 256]` | *LLaDA only.* Critical lever for Quality/Speed trade-off. |
| **Batch ($B$)** | `[1, 4, 8, 32]` | Test if diffusion scales better than AR at high load. |
| **Seq Len ($N$)** | `[128, 512, 1024, 2048]` | Test memory bottlenecks (KV Cache vs. Attention Map). |

## 5\. Preliminary Results

### 5.1 CPU Validation Data

Initial smoke tests on CPU reveal the magnitude of the challenge:

  * **Quality Gap:**
      * LLaMA Baseline Perplexity: **10.47**
      * LLaDA (16 Steps) Perplexity: **103,745.26**
      * *Finding:* 16 steps is insufficient for intelligible text. $S_{parity}$ is likely significantly higher.
  * **Latency Gap:** LLaDA (16 steps) was \~1.5x slower than LLaMA per batch on CPU.

### 5.2 Full GPU Benchmark

Comprehensive GPU execution on the ACCRE cluster is configured to populate the following output artifacts:

  * `results/figures/pareto_frontier.png`: Visualizing the crossover point where LLaDA quality matches LLaMA.
  * `results/raw_data/`: Raw CSV metrics for throughput at Batch Size 32.

## 6\. Discussion

### 6.1 The "Streaming" Problem

Regardless of raw speed, the architectures offer fundamentally different user experiences. LLaMA provides immediate feedback (low $L_{itl}$). LLaDA requires the user to wait for the full generation ($L_{e2e}$). This likely restricts LLaDA to batch-processing workloads unless its $L_{e2e}$ is faster than LLaMA's *time-to-first-token*, which is mathematically unlikely.

### 6.2 The Parity Cost

The preliminary perplexity of \>100k at 16 steps suggests the "useful" step count is likely in the 64–128 range. If LLaDA is already slower at 16 steps (CPU data), achieving parity at 64+ steps may make it prohibitively expensive compared to highly optimized autoregressive baselines.

## 7\. Conclusion

1.  **Low-Step Viability:** LLaDA is unusable at 16 steps (Perplexity \>100k).
2.  **Use Case:** LLaDA is likely unsuitable for interactive chat due to the lack of streaming. Its viability hinges entirely on **batch throughput** performance.
3.  **Future Work:** Completion of the A100 GPU benchmarks will definitively locate $S_{parity}$ and determine if high-batch parallelization can offset the computational overhead of iterative denoising.

## 8\. References

  - **Models:** `meta-llama/Meta-Llama-3-8B`, `GSAI-ML/LLaDA-8B-Base`
  - **Data:** Merity et al. (2016). *Wikitext-103*.
  - **Tools:** PyTorch 2.4, HuggingFace Transformers.

-----

## Appendix A: Reproducibility

### Repository Structure

```
.
├── config/             # Experiment grids (experiments.yaml)
├── jobs/               # SLURM sbatch scripts
├── src/                # Benchmark harness (Python 3.13)
├── results/raw_data/   # Metrics CSVs
└── results/figures/    # Generated Pareto plots
```

### Execution Guide (ACCRE Cluster)

1.  **Setup Environment:**

    ```bash
    setup_accre_software_stack
    module load python/3.13.2 scipy-stack/2025a
    python -m venv ~/llada_bench_env
    source ~/llada_bench_env/bin/activate
    pip install -r requirements.txt
    ```

2.  **Submit Benchmark:**

    ```bash
    # Runs full grid search as defined in config/experiments.yaml
    sbatch jobs/benchmark_gpu.sbatch
    ```

3.  **Local Test (NixOS/ROCm):**

    ```bash
    nix-shell shell.nix
    python -m src.main --config config/experiments_mini.yaml --output_dir results/raw_data
    ```