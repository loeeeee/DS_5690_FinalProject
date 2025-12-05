# Inference Efficiency Analysis of LLaDA: A Comparative Benchmark Against LLaMA 3

## Abstract

[To be completed] This study addresses a critical gap in the evaluation of LLaDA (8B language diffusion model) by providing a rigorous, reproducible benchmark comparing its inference efficiency against LLaMA 3 8B. We empirically quantify the quality vs. computation trade-off through systematic evaluation of latency, throughput, and memory consumption across varying diffusion step counts, batch sizes, and sequence lengths. Our analysis determines the practical viability of LLaDA for real-world applications by identifying the step count required to achieve quality parity with autoregressive baselines.

## 1. Introduction

The LLaDA (8B language diffusion model) paper was rejected for lacking a sufficient inference efficiency analysis. This project fills that critical gap by providing a rigorous, reproducible benchmark comparing LLaDA against a LLaMA 3 8B baseline on key performance metrics: **latency**, **throughput**, and **memory consumption**. The primary goal is to empirically quantify the quality vs. computation trade-off to assess LLaDA's practical viability for real-world application.

Diffusion models for language generation represent a paradigm shift from autoregressive architectures, potentially offering advantages in parallelization and batch processing. However, the computational cost of iterative denoising steps raises fundamental questions about their efficiency relative to established autoregressive models. This study systematically evaluates these trade-offs through controlled experiments that isolate architectural differences from implementation optimizations.

## 2. Related Work

[To be completed] This section will review relevant literature on:
- Diffusion models for language generation
- Inference efficiency benchmarks for large language models
- Quality-efficiency trade-offs in generative models
- Comparative analyses of autoregressive vs. non-autoregressive architectures

## 3. Methodology

### 3.1 Performance Metrics

To ensure rigor, we define our metrics mathematically to remove ambiguity and enable replication. Since we compare fundamentally different architectures—**Autoregressive (LLaMA)** vs. **Diffusion (LLaDA)**—standard metrics apply differently. LLaMA generates tokens sequentially; LLaDA typically refines entire sequences simultaneously.

#### 3.1.1 Latency Metrics

**End-to-End Latency ($L_{e2e}$)**

The total wall-clock time required to generate a full response. This is the **primary metric** for comparing the two models, as it is the only common denominator.

$$L_{e2e} = t_{end} - t_{start}$$

- **For LLaMA:** The time from input ingestion until the `<EOS>` token is generated.
- **For LLaDA:** The time from input ingestion until the final diffusion step is complete.
- **Why it matters:** This determines if LLaDA is responsive enough for real-time chat.

**Inter-Token Latency ($L_{itl}$)** (LLaMA Only)

The average time elapsing between the appearance of two consecutive tokens.

$$L_{itl} = \frac{L_{e2e} - L_{prefill}}{N_{tokens} - 1}$$

- **Why it matters:** LLaMA might have high total latency for long texts, but low $L_{itl}$ means the user sees text appearing smoothly. LLaDA typically has no $L_{itl}$ (the text appears all at once), which may feel "slower" to a user even if $L_{e2e}$ is identical.

#### 3.1.2 Throughput Metrics

**Generation Throughput ($T_{gen}$)**

The number of valid output tokens generated per second across the entire batch.

$$T_{gen} = \frac{B \times N_{out}}{L_{e2e}}$$

Where:
- $B$ is the batch size.
- $N_{out}$ is the average number of output tokens per sequence.
- $L_{e2e}$ is the end-to-end latency for that batch.

**Hypothesis:** LLaDA may show superior throughput at high batch sizes because diffusion models often parallelize better than autoregressive models (which suffer from KV cache memory bottlenecks).

#### 3.1.3 Memory Metrics

**Peak Allocated VRAM ($M_{peak}$)**

The maximum amount of GPU memory occupied by the model weights, activation states, and caches at any point during generation.

$$M_{peak} = \max_{t \in [0, T]} (M_{reserved}(t))$$

- **LLaMA bottleneck:** The **KV Cache**. As sequence length ($S$) increases, memory usage grows linearly ($O(S)$).
- **LLaDA bottleneck:** The **Activation Map**. Memory usage depends on the full sequence length from the start, but doesn't necessarily grow during the generation process.

#### 3.1.4 Efficiency Trade-off Metric

**Steps-to-Quality Parity ($S_{parity}$)**

The number of diffusion steps ($k$) LLaDA requires to match the perplexity ($\text{PPL}$) of LLaMA 3.

Let $Q_{llama}$ be the baseline quality (e.g., BERTScore or Perplexity).
Let $f_{llada}(k)$ be the quality of LLaDA at $k$ steps.

$$S_{parity} = \min \{ k \mid f_{llada}(k) \geq Q_{llama} \}$$

Once we find $S_{parity}$, we compare the Latency:

$$\text{Speedup Factor} = \frac{L_{e2e}(\text{LLaMA})}{L_{e2e}(\text{LLaDA at } S_{parity})}$$

- **Interpretation:** If the Speedup Factor is $>1$, LLaDA is viable. If $<1$, LLaDA is computationally inefficient for that quality level.

### 3.2 Measurement Protocol

To measure $L_{e2e}$ accurately on a GPU, we cannot use Python's `time.time()`. CUDA operations are asynchronous; Python will report the code has finished while the GPU is still working. We use `torch.cuda.Event` for precise GPU timing:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... run model generation ...
end_event.record()

torch.cuda.synchronize() # WAITS for GPU to finish
elapsed_time_ms = start_event.elapsed_time(end_event)
```

## 4. Experimental Setup

### 4.1 Independent Variables

These are the parameters we sweep through to observe how the models react.

#### 4.1.1 Model Architecture ($A$)

The primary categorical variable defining the generation mechanism.

- **Baseline:** `Meta-LLaMA-3-8B` (Autoregressive)
- **Target:** `LLaDA-8B` (Diffusion)

#### 4.1.2 Diffusion Steps ($K$) — LLaDA Only

The number of denoising iterations performed during inference. This is the critical lever for the "Quality vs. Computation" trade-off.

- **Range:** `[16, 32, 64, 128, 256]`
- **Rationale:** Low steps ($K=16$) yield high speed but likely gibberish; high steps ($K=256$) yield high quality but high latency. We need to find the curve.

#### 4.1.3 Batch Size ($B$)

The number of sequences generated in parallel.

- **Range:** `[1, 4, 8, 16, 32, 64]`
- **Rationale:**
  - $B=1$: Simulates a single-user interactive chat (Latency critical)
  - $B=32+$: Simulates a server environment (Throughput critical). Diffusion models often scale better here due to lack of serial dependency.

#### 4.1.4 Sequence Length ($N$)

Total token count (Input + Output). To ensure fair comparison, we fix the input-to-output ratio (e.g., 50% prompt, 50% generation).

- **Range:** `[128, 512, 1024, 2048]`
- **Rationale:**
  - **LLaMA:** Memory usage scales with $N$ due to the KV Cache ($O(N)$). Compute scales $O(N^2)$ (or $O(N)$ with FlashAttention).
  - **LLaDA:** We need to see if the attention map memory usage explodes or remains manageable at 2048.

### 4.2 Control Variables

These variables remain **identical** across all runs. If these change, the data is invalid.

#### 4.2.1 Hardware Environment ($H$)

- **Node Type:** Single ACCRE Compute Node
- **GPU:** NVIDIA A100 (80GB PCIe/SXM). *Do not mix A100 and A6000 results.*
- **CPU:** Exclusive access (ensure no other users are on the node)
- **P-State:** Performance mode (usually locked by cluster admins, but good to note)

#### 4.2.2 Precision ($P$)

- **Type:** `bfloat16` (BF16)
- **Rationale:** Standard for LLaMA 3. `float32` is too slow/memory-heavy; `int8` (quantization) introduces kernel implementation variables that confuse the architectural comparison. BF16 ensures we measure the *model's* cost, not the quantization kernel's efficiency.

#### 4.2.3 Output Token Count ($N_{out}$)

- **Definition:** For LLaMA, we force it to generate exactly $X$ tokens. For LLaDA, we mask exactly $X$ tokens.
- **Constraint:** The model must generate the *same amount of text*. You cannot compare LLaDA generating a sentence vs. LLaMA generating a paragraph.

#### 4.2.4 Software Stack ($S$)

- **Python:** 3.13.2
- **PyTorch:** Version 2.4+ (Standard stable)
- **CUDA:** 12.6 (Loaded via ACCRE modules)
- **Attention Kernel:** FlashAttention-2 (Must be enabled for *both* or *neither*). *Recommendation: Enable for both to reflect modern usage.*

### 4.3 Procedural Variables

How the measurement is taken.

#### 4.3.1 Warm-up Iterations ($W$)

- **Value:** 3 runs
- **Purpose:** The first few runs of any PyTorch model invoke JIT compilation (Triton kernels/CUDA graphs) and memory allocation overhead. We discard these to measure stable performance.

#### 4.3.2 Sampling Temperature ($T$)

- **Value:** 0 (Greedy Decoding) or fixed small value (e.g., 0.1)
- **Purpose:** While temperature doesn't directly affect speed, it affects the output *distribution*. For the "Quality" metric, we want deterministic or near-deterministic outputs to ensure reproducibility.

### 4.4 Experimental Design Summary

| Variable Type | Variable Name | Configuration / Constraint |
| :--- | :--- | :--- |
| **Independent** | Architecture | LLaMA 3 8B vs. LLaDA 8B |
| **Independent** | Batch Size ($B$) | 1, 8, 32 |
| **Independent** | Sequence Length ($N$) | 512, 1024, 2048 |
| **Independent** | Diffusion Steps ($K$) | 32, 64, 128 (LLaDA only) |
| **Control** | Precision | `bfloat16` |
| **Control** | Hardware | NVIDIA A100 80GB (ACCRE) |
| **Control** | Software | Python 3.13, PyTorch 2.x, CUDA 12.6 |
| **Control** | Warmup | 3 dummy passes before measurement |

## 5. Results

[To be completed] This section will present:

- Raw metrics data located in `results/raw_data/`
- Generated figures located in `results/figures/`
- Pareto frontier plots (X-axis = Latency, Y-axis = Quality)
- Throughput bar charts comparing tokens/sec at different batch sizes
- Memory heatmaps showing VRAM usage across sequence lengths
- Quality vs. steps analysis for LLaDA
- Steps-to-quality parity determination

## 6. Discussion

[To be completed] This section will analyze:

- The quality-computation trade-off curve for LLaDA
- Comparison of latency, throughput, and memory consumption between architectures
- Scalability characteristics (batch size and sequence length effects)
- Practical implications for real-world deployment
- Limitations of the study

## 7. Conclusion

[To be completed] This section will summarize:

- Key findings regarding LLaDA's inference efficiency
- Whether LLaDA achieves quality parity with LLaMA at acceptable computational cost
- Recommendations for practical applications
- Future research directions

## 8. References

[To be completed] Citations will include:

- LLaDA paper
- LLaMA 3 technical report
- Related work on diffusion models for language
- Inference efficiency benchmarking methodologies

## Appendix A: Reproducibility

This appendix provides detailed technical instructions for reproducing the experimental results.

### A.1 Repository Structure

```
DS_5690_FinalProject/
├── docs-vibe/          # Intent/setup notes (update before/after coding)
├── jobs/               # sbatch + env bootstrap for cluster runs
├── config/             # Grid definitions (steps, batch sizes, seq lengths, hardware notes)
│   └── experiments.yaml
├── src/                # CLI benchmark harness (models, evaluation, analysis)
├── logs/slurm/         # SLURM stdout/err
├── results/raw_data/   # Metrics CSV/JSON
└── results/figures/    # Generated plots (matplotlib Agg)
```

### A.2 Local Development Environment (NixOS, ROCm)

For local development and testing on NixOS systems with AMD GPUs:

**Enter development shell:**
```bash
nix-shell shell.nix
```

This provides Python 3.13, ROCm 6.4, and support for RX 7900 XT.

**Verify ROCm PyTorch installation:**
```bash
python - <<'PY'
import torch, platform
print("torch", torch.__version__)
print("hip version", torch.version.hip)
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
print("arch", torch.cuda.get_device_capability())
print("platform", platform.platform())
PY
```

**Run benchmark locally:**
```bash
python -m src.main --config config/experiments.yaml --output_dir results/raw_data
```

**Environment notes:**
- `PYTORCH_HIP_ARCH=gfx1100` and `HSA_OVERRIDE_GFX_VERSION=11.0.0` are set for RX 7900 XT
- torchWithRocm uses ROCm 6.4
- **Note:** ROCm (AMD GPUs) may have compatibility issues with `model.generate()`. The code is designed for CUDA (NVIDIA GPUs) and should work correctly on CUDA systems.

### A.3 Pre-downloading Models and Data

From inside `nix-shell`, fetch model caches ahead of time:

```bash
python scripts/download_assets.py --config config/experiments.yaml
```

**Additional options:**
- `--hf-token $HF_TOKEN`: Required for gated LLaMA 3 access
- `--model-cache-dir /fast/hf`: Specify model cache location
- `--dataset-cache-dir /fast/hf_datasets`: Specify dataset cache location
- `--extra-model-id`: Pull additional model repositories
- `--local-files-only`: Only validate existing cache without downloading

**Default model IDs:**
- LLaMA: `meta-llama/Meta-Llama-3-8B`
- LLaDA: `GSAI-ML/LLaDA-8B-Base`

### A.4 Cluster Execution (SLURM)

**Configure experiments:**
Edit `config/experiments.yaml` to define experimental parameters (default models listed above).

**Submit job:**
```bash
sbatch jobs/benchmark_gpu.sbatch
```

**Optional command-line overrides:**
```bash
python src/main.py \
    --config config/experiments.yaml \
    --steps 64 \
    --batch_size 4 \
    --max_prompts 16 \
    --compute_bertscore \
    --output_dir results/raw_data
```

**Output files:**
- Latency/throughput CSV: `results/raw_data/raw_metrics.csv`
- Quality metrics JSON: `results/raw_data/quality_metrics.json`

### A.5 Mini Test (GPU Validation)

For quick validation runs:

```bash
python src/main.py \
    --config config/experiments_mini.yaml \
    --output_dir results/raw_data/smoke \
    --max_prompts 4 \
    --batch_size 1 \
    --steps 16
```

### A.6 Additional Technical Notes

**Prompt sources:**
- Default prompts use Wikitext-103 validation text
- Supply `--prompt_file` to use custom prompts

**Profiling:**
- Uses CUDA events when available (`torch.cuda.Event`)
- Falls back to CPU timers otherwise

**Precision:**
- Generation uses BF16 by default
- Adjust `precision` in `config/experiments.yaml` if required

**For detailed SLURM usage instructions:**
See `.clinerules/sbatch_usage.md` for ACCRE-specific cluster configuration details.

**For detailed methodology:**
See `.clinerules/metrics.md` for formal metric definitions and `.clinerules/experiment_variables.md` for complete experimental design documentation.
