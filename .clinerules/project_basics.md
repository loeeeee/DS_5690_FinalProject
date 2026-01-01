# Project Basics

The LLaDA (8B language diffusion model) paper was rejected for lacking a sufficient inference efficiency analysis. This project will fill that critical gap. We will provide a rigorous, reproducible benchmark comparing LLaDA against a LLaMA 3 8B baseline on key performance metrics: **latency**, **throughput**, and **memory consumption**. The primary goal is to empirically quantify the quality vs. computation trade-off to assess LLaDA's practical viability for real-world application.

## Phases

### Phase 1: Experimental Design & Metrics Definition

Before writing code, you must define what "comparable" means for these two architectures.

* **Define Hardware constraints:** Lock down the hardware environment. Ideally, use a consumer-grade high-end card (e.g., RTX 4090) or a data center card (A100) and ensure no other processes are running.
* **Define Metrics:**
    * **Latency:**
        * *LLaMA:* Time to First Token (TTFT) vs. Inter-Token Latency (ITL).
        * *LLaDA:* Total Wall-clock Time for a complete sequence generation (since diffusion models often generate/refine blocks rather than single tokens).
    * **Throughput:** Tokens generated per second (TPS) under various batch sizes.
    * **Memory:** Peak VRAM usage (allocated vs. reserved).
* **Define the Variables:**
    * *Control:* Input prompt length, output sequence length, precision (BF16/FP16).
    * *Independent Variable (LLaDA):* The number of **sampling steps**. This is the critical lever. LLaDA might be faster at 10 steps but nonsensical, and slower but high quality at 100 steps.

### Phase 2: Environment Setup & Baselines

Ensure the environment is reproducible using a container or strict requirements file.

* **Software Stack:** Python 3.13, PyTorch (latest stable), and `transformers`.
* **Baseline (LLaMA 3 8B):**
    * Set up a standard inference pipeline.
    * **Optimization check:** Decide if you are using `vLLM` (highly optimized) or vanilla HuggingFace `generate()` (less optimized). *Recommendation:* Use vanilla PyTorch/HuggingFace for both to ensure you are comparing architectures, not engineering optimizations, unless LLaDA has a dedicated high-performance kernel available.
* **Target (LLaDA 8B):**
    * Clone the official repository or implementation.
    * Verify you can replicate the generation examples provided by the authors to ensure correctness before profiling.

### Phase 3: Benchmark Implementation (The Harness)

Develop a unified Python benchmarking script that wraps both models.

* **Data Preparation:** Select a standard dataset (e.g., a subset of ShareGPT or Wikitext) to provide real-world prompt distributions. Do not use random noise, as token embeddings impact memory and computation in attention layers.
* **Instrumentation:**
    * Use `torch.cuda.Event(enable_timing=True)` for precise GPU timing (Python `time` is not accurate enough for GPU async operations).
    * Use `torch.cuda.max_memory_allocated()` and `torch.cuda.reset_peak_memory_stats()` to track VRAM.
* **Warm-up:** Ensure you run 3-5 dummy passes before recording data to allow the CUDA compiler/allocator to settle.

### Phase 4: The "Quality vs. Computation" Analysis

This is the step that addresses the paper's rejection reason. You cannot measure speed in a vacuum.

* **Step Sweep:** Run LLaDA inference at varying diffusion steps (e.g., 16, 32, 64, 128, 256).
* **Quality Proxy:** For each step count, calculate a lightweight quality metric (e.g., Perplexity on a test set or BERTScore against a reference).
* **The Cross-over Point:** Determine at how many steps LLaDA matches LLaMA 3's quality.
    * *Example Question:* "Does LLaDA require 100 steps to match LLaMA? If so, is LLaDA's 100-step latency higher or lower than LLaMA's autoregressive generation?"

### Phase 5: Scalability Stress Testing

Test how the models react to extremes.

* **Batch Size Scaling:** Test batch sizes of 1, 4, 8, 16, 32. Diffusion models often parallelize better than autoregressive models. This is a potential "win" area for LLaDA.
* **Sequence Length Scaling:** Test output lengths of 128, 512, 1024, 2048 tokens. LLaMA's attention matrix grows quadratically (without FlashAttention) and KV cache grows linearly. Check how LLaDA's memory footprint scales with sequence length.

### Phase 6: Reporting & Visualization

Synthesize the data into the final report.

* **Pareto Frontier Plot:** X-axis = Latency (lower is better), Y-axis = Quality (higher is better). Plot LLaMA as a single point (or line if varying quantization) and LLaDA as a curve (varying steps).
* **Throughput Bar Charts:** Side-by-side comparison of tokens/sec at different batch sizes.
* **Memory Heatmap:** Show VRAM usage across different sequence lengths.

### Phase 7: Reproducibility Packaging

* Create a `requirements.txt` or `Dockerfile`.
* Release the benchmarking code on GitHub with a README explaining how to replicate the specific numbers in your report.
