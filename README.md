# Inference Efficiency Analysis of LLaDA: A Comparative Benchmark Against LLaMA 3

## Abstract

This study addresses a critical gap in the evaluation of LLaDA (8B language diffusion model) by providing a rigorous, reproducible benchmark comparing its inference efficiency against LLaMA 3 8B. We empirically quantify the quality vs. computation trade-off through systematic evaluation of latency, throughput, and memory consumption across varying diffusion step counts, batch sizes, and sequence lengths. Our benchmark framework enables headless execution on HPC clusters via SLURM, ensuring reproducibility and scalability. Preliminary results indicate significant quality-efficiency trade-offs that require careful step selection for LLaDA to achieve competitive performance. Our analysis determines the practical viability of LLaDA for real-world applications by identifying the step count required to achieve quality parity with autoregressive baselines.

## 1. Introduction

The LLaDA (8B language diffusion model) paper was rejected for lacking a sufficient inference efficiency analysis. This project fills that critical gap by providing a rigorous, reproducible benchmark comparing LLaDA against a LLaMA 3 8B baseline on key performance metrics: **latency**, **throughput**, and **memory consumption**. The primary goal is to empirically quantify the quality vs. computation trade-off to assess LLaDA's practical viability for real-world application.

Diffusion models for language generation represent a paradigm shift from autoregressive architectures, potentially offering advantages in parallelization and batch processing. However, the computational cost of iterative denoising steps raises fundamental questions about their efficiency relative to established autoregressive models. This study systematically evaluates these trade-offs through controlled experiments that isolate architectural differences from implementation optimizations.

## 2. Related Work

### 2.1 Diffusion Models for Language Generation

Diffusion models have shown remarkable success in image generation, leading to their adaptation for language modeling. LLaDA (Language Latent Diffusion Architecture) represents a recent attempt to apply diffusion principles to text generation, offering potential advantages in parallelization over autoregressive models. However, the iterative denoising process introduces computational overhead that must be carefully evaluated.

### 2.2 Inference Efficiency Benchmarks

Previous work on inference efficiency has primarily focused on autoregressive models, with extensive benchmarking of latency, throughput, and memory consumption. Standard metrics include Time-to-First-Token (TTFT), Inter-Token Latency (ITL), and tokens-per-second throughput. However, these metrics require adaptation for diffusion models, which generate sequences differently.

### 2.3 Quality-Efficiency Trade-offs

The relationship between generation quality and computational cost is a fundamental consideration in deploying large language models. For diffusion models, this trade-off is particularly pronounced, as the number of denoising steps directly impacts both quality and latency. Understanding this relationship is critical for practical deployment decisions.

### 2.4 Autoregressive vs. Non-Autoregressive Architectures

Autoregressive models like LLaMA generate tokens sequentially, enabling streaming output but requiring sequential computation. Non-autoregressive approaches, including diffusion models, can potentially generate entire sequences in parallel but may require multiple iterations to achieve comparable quality. This architectural difference fundamentally affects inference characteristics and must be evaluated empirically.

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
- **Attention Kernel:** Standard PyTorch attention implementation (FlashAttention disabled for fair comparison). Both models use identical attention implementations to isolate architectural differences from optimization-level differences.

### 4.3 Procedural Variables

How the measurement is taken.

#### 4.3.1 Warm-up Iterations ($W$)

- **Value:** 3 runs (primary configuration) or 1 run (CPU test configuration)
- **Purpose:** The first few runs of any PyTorch model invoke JIT compilation (Triton kernels/CUDA graphs) and memory allocation overhead. We discard these to measure stable performance.

#### 4.3.2 Sampling Temperature ($T$)

- **Value:** 0 (Greedy Decoding) or fixed small value (e.g., 0.1)
- **Purpose:** While temperature doesn't directly affect speed, it affects the output *distribution*. For the "Quality" metric, we want deterministic or near-deterministic outputs to ensure reproducibility.

#### 4.3.3 Dataset Configuration

- **Dataset:** Wikitext-103 (`wikitext`, `wikitext-103-v1`)
- **Split:** Validation set
- **Usage:** Prompts are extracted from validation sequences; models generate continuations
- **Dataset Size Presets:**
  - Small: 8 prompts (for quick validation)
  - Medium: 32 prompts (primary configuration)
  - Large: 128 prompts (comprehensive evaluation)
- **Custom Prompts:** Can be supplied via `--prompt_file` argument for reproducible evaluation

### 4.4 Experimental Design Summary

| Variable Type | Variable Name | Configuration / Constraint |
| :--- | :--- | :--- |
| **Independent** | Architecture | LLaMA 3 8B vs. LLaDA 8B |
| **Independent** | Batch Size ($B$) | 1, 4 (primary); 1, 8, 32 (extended range) |
| **Independent** | Sequence Length ($N$) | 512, 1024 (primary); 128, 512, 1024, 2048 (full range) |
| **Independent** | Diffusion Steps ($K$) | 32, 64, 128 (primary); 16, 32, 64, 128, 256 (full range) |
| **Control** | Precision | `bfloat16` (GPU); `float32` (CPU test) |
| **Control** | Output Tokens | 128 tokens per sequence |
| **Control** | Hardware | NVIDIA A100 80GB (ACCRE) for GPU; CPU for validation |
| **Control** | Software | Python 3.13, PyTorch 2.4+, CUDA 12.6 |
| **Control** | Dataset | Wikitext-103 validation split |
| **Control** | Warmup | 3 dummy passes before measurement (1 for CPU test) |

## 5. Results

### 5.1 Experimental Configuration

Our benchmark evaluates two models:
- **Baseline:** Meta-Llama-3-8B (`meta-llama/Meta-Llama-3-8B`) - Autoregressive architecture
- **Target:** LLaDA-8B-Base (`GSAI-ML/LLaDA-8B-Base`) - Diffusion architecture

Experiments are conducted using the Wikitext-103 validation dataset, with evaluation metrics computed over generated sequences. The primary experimental configuration (defined in `config/experiments.yaml`) tests:
- Sequence lengths: 512, 1024 tokens
- Batch sizes: 1, 4
- Diffusion steps (LLaDA only): 32, 64, 128
- Precision: bfloat16 (BF16)
- Output tokens: 128 tokens per sequence

### 5.2 Data Availability

Raw experimental data is stored in `results/raw_data/` with the following structure:
- **CSV files:** Contain per-batch metrics including latency, throughput, memory usage, and token counts
- **JSON files:** Contain aggregated quality metrics (perplexity, optional BERTScore)

Generated visualizations are available in `results/figures/`:
- `latency_vs_steps.png`: Latency comparison across diffusion step counts
- `quality_vs_steps.png`: Quality (perplexity) as a function of steps
- `pareto_frontier.png`: Quality-latency trade-off visualization
- `throughput_vs_batch_size.png`: Throughput scaling with batch size
- `throughput_by_batch_size.png`: Comparative throughput analysis

### 5.3 Preliminary Findings (CPU Test)

Initial validation runs on CPU (using `config/experiments_cpu_test.yaml`) provide preliminary insights:

**Quality Metrics:**
- LLaMA baseline perplexity: 10.47
- LLaDA at 16 steps perplexity: 103,745.26

This substantial quality gap at low step counts confirms the critical importance of step selection for LLaDA. The high perplexity indicates that 16 steps are insufficient for meaningful text generation.

**Latency Observations:**
- LLaMA baseline: ~25-24 seconds per batch (batch size 1, sequence length 128)
- LLaDA (16 steps): ~36-44 seconds per batch under the same conditions

These preliminary CPU results suggest that LLaDA requires significantly more computation time even at low step counts, though full GPU benchmarks are needed to assess practical viability.

### 5.4 Full GPU Benchmark Status

Complete GPU benchmark results are pending execution on ACCRE cluster infrastructure. The experimental design is configured for:
- Hardware: NVIDIA A100 (80GB)
- Full experimental grid: All combinations of sequence lengths, batch sizes, and diffusion steps
- Comprehensive quality-efficiency analysis

Results will be updated in this section upon completion of full benchmark runs.

## 6. Discussion

### 6.1 Quality-Computation Trade-off

Preliminary results demonstrate a stark quality-computation trade-off for LLaDA. At 16 diffusion steps, LLaDA exhibits perplexity approximately 10,000× higher than the LLaMA baseline, indicating that low step counts produce essentially unusable outputs. This finding underscores the critical importance of identifying the minimum step count required for quality parity ($S_{parity}$).

The trade-off curve suggests that LLaDA may require substantially more computational resources (higher step counts) to achieve comparable quality to autoregressive models. This has direct implications for deployment costs and latency requirements.

### 6.2 Architectural Comparison

**Latency Characteristics:**
- LLaMA generates tokens sequentially, enabling streaming output with low inter-token latency
- LLaDA generates complete sequences after all diffusion steps, resulting in higher end-to-end latency before any output appears
- This fundamental difference affects user experience: LLaMA provides incremental feedback, while LLaDA requires waiting for complete generation

**Throughput Potential:**
- Diffusion models may benefit from better parallelization at high batch sizes due to lack of sequential dependencies
- Autoregressive models suffer from KV cache memory bottlenecks that limit batch scaling
- Full GPU benchmarks will quantify whether LLaDA's parallelization advantages offset its step count requirements

**Memory Considerations:**
- LLaMA's memory scales linearly with sequence length due to KV cache requirements
- LLaDA's memory footprint depends on full sequence length from the start but may not grow during generation
- The relative memory efficiency requires empirical evaluation across sequence lengths

### 6.3 Scalability Analysis

Batch size and sequence length scaling characteristics will be critical for determining LLaDA's practical viability:
- **Batch scaling:** If LLaDA shows superior throughput at high batch sizes, it may be viable for batch processing workloads despite higher latency
- **Sequence length scaling:** Memory and compute scaling with sequence length will determine LLaDA's applicability to long-context tasks

### 6.4 Practical Implications

For real-world deployment, several factors must be considered:

1. **Interactive Applications:** LLaDA's all-at-once generation pattern may be unsuitable for chat interfaces requiring streaming output, regardless of absolute latency.

2. **Batch Processing:** If LLaDA achieves superior throughput at high batch sizes, it may be viable for offline text generation tasks where latency is less critical.

3. **Quality Requirements:** The step count required for quality parity directly impacts computational cost. If $S_{parity}$ is high (e.g., >100 steps), LLaDA may be prohibitively expensive compared to autoregressive alternatives.

### 6.5 Limitations

This study has several limitations:

1. **Hardware Constraints:** Full GPU benchmarks are pending. CPU results provide only preliminary insights and may not reflect GPU-accelerated performance characteristics.

2. **Quality Metrics:** Current evaluation uses perplexity on Wikitext-103. Additional metrics (e.g., BERTScore, task-specific evaluations) would provide more comprehensive quality assessment.

3. **Step Range:** Preliminary tests cover limited step ranges. Full analysis requires evaluation across the complete range [16, 32, 64, 128, 256] to identify $S_{parity}$.

4. **Optimization Level:** Both models use standard HuggingFace implementations. Specialized optimizations (e.g., vLLM for LLaMA) may alter relative performance characteristics.

5. **Dataset Scope:** Evaluation uses Wikitext-103 validation set. Generalization to other domains and tasks requires additional evaluation.

## 7. Conclusion

### 7.1 Key Findings

Preliminary analysis reveals significant challenges for LLaDA's inference efficiency:

1. **Quality Gap:** At low step counts (16 steps), LLaDA exhibits dramatically higher perplexity (103,745 vs. 10.47) compared to LLaMA, indicating insufficient quality for practical use.

2. **Latency Overhead:** Even at low step counts, LLaDA requires substantially more computation time per sequence, suggesting that achieving quality parity will require significant computational investment.

3. **Architectural Trade-offs:** The fundamental difference in generation patterns (sequential streaming vs. complete sequence generation) affects both latency characteristics and user experience.

### 7.2 Quality Parity Assessment

Full determination of steps-to-quality parity ($S_{parity}$) requires completion of GPU benchmarks across the full step range. However, preliminary results suggest that achieving quality parity will require significantly more than 16 steps, potentially making LLaDA computationally inefficient compared to autoregressive alternatives.

The critical question—whether LLaDA can achieve quality parity at acceptable computational cost—remains to be answered through comprehensive GPU evaluation.

### 7.3 Recommendations

Based on current findings:

1. **For Interactive Applications:** LLaDA's all-at-once generation pattern makes it unsuitable for applications requiring streaming output, regardless of absolute latency performance.

2. **For Batch Processing:** If full benchmarks demonstrate superior throughput at high batch sizes, LLaDA may be viable for offline text generation tasks. However, quality requirements must be carefully evaluated.

3. **Step Selection:** If deploying LLaDA, careful step count selection is critical. Low step counts produce unusable quality, while high step counts may be prohibitively expensive.

### 7.4 Future Research Directions

1. **Optimization:** Investigation of specialized inference optimizations for diffusion language models may improve efficiency.

2. **Hybrid Approaches:** Exploring hybrid architectures that combine diffusion and autoregressive principles could leverage advantages of both paradigms.

3. **Quality Metrics:** Development of task-specific quality metrics beyond perplexity would provide more nuanced evaluation of generation quality.

4. **Long-Context Analysis:** Evaluation of LLaDA's performance on long-context tasks may reveal advantages not apparent in shorter sequences.

5. **Comparative Analysis:** Extension to other diffusion language models would provide broader understanding of the diffusion paradigm's efficiency characteristics.

## 8. References

### 8.1 Model References

- **LLaMA 3:** Meta AI. (2024). *Llama 3 Model Card*. HuggingFace: `meta-llama/Meta-Llama-3-8B`
- **LLaDA:** GSAI-ML. (2024). *LLaDA-8B-Base*. HuggingFace: `GSAI-ML/LLaDA-8B-Base`

### 8.2 Dataset References

- **Wikitext-103:** Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). *Pointer Sentinel Mixture Models*. arXiv preprint arXiv:1609.07843.

### 8.3 Software and Tools

- **PyTorch:** Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
- **Transformers:** Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP.
- **HuggingFace Datasets:** Lhoest, Q., et al. (2021). *Datasets: A Community Library for Natural Language Processing*. EMNLP.

### 8.4 Related Work

*Note: Full literature review citations will be added upon completion of comprehensive review. Key areas include:*

- Diffusion models for language generation
- Inference efficiency benchmarking methodologies
- Autoregressive vs. non-autoregressive language model architectures
- Quality-efficiency trade-offs in generative models

### 8.5 Benchmarking Methodology

This work follows rigorous benchmarking practices:
- Reproducible experimental design with controlled variables
- Mathematical formalization of performance metrics
- Headless execution framework for HPC environments
- Standardized evaluation protocols for fair comparison

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
