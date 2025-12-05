To conduct a rigorous scientific benchmark, you must clearly distinguish between what you are changing (**Independent Variables**) and what you are keeping strictly constant (**Control Variables**) to isolate the architecture's impact.

Here is the detailed variable breakdown for your **Experimental Design** section.

### I. Independent Variables (The "Knobs")
These are the parameters you will sweep through in your `experiments.yaml` to observe how the models react.

#### 1. Model Architecture ($A$)
The primary categorical variable defining the generation mechanism.
* **Levels:**
    * **Baseline:** `Meta-LLaMA-3-8B` (Autoregressive).
    * **Target:** `LLaDA-8B` (Diffusion).

#### 2. Diffusion Steps ($K$) — *LLaDA Only*
The number of denoising iterations performed during inference. This is the critical lever for the "Quality vs. Computation" trade-off.
* **Range:** `[16, 32, 64, 128, 256]`
* **Rationale:** Low steps ($K=16$) yield high speed but likely gibberish; high steps ($K=256$) yield high quality but high latency. We need to find the curve.

#### 3. Batch Size ($B$)
The number of sequences generated in parallel.
* **Range:** `[1, 4, 8, 16, 32, 64]`
* **Rationale:**
    * $B=1$: Simulates a single-user interactive chat (Latency critical).
    * $B=32+$: Simulates a server environment (Throughput critical). Diffusion models often scale better here due to lack of serial dependency.

#### 4. Sequence Length ($N$)
Total token count (Input + Output). To ensure fair comparison, we fix the input-to-output ratio (e.g., 50% prompt, 50% generation).
* **Range:** `[128, 512, 1024, 2048]`
* **Rationale:**
    * **LLaMA:** Memory usage scales with $N$ due to the KV Cache ($O(N)$). Compute scales $O(N^2)$ (or $O(N)$ with FlashAttention).
    * **LLaDA:** We need to see if the attention map memory usage explodes or remains manageable at 2048.

---

### II. Control Variables (The "Constants")
These variables must remain **identical** across all runs. If these change, your data is invalid.

#### 1. Hardware Environment ($H$)
* **Node Type:** Single ACCRE Compute Node.
* **GPU:** NVIDIA A100 (80GB PCIe/SXM). *Do not mix A100 and A6000 results.*
* **CPU:** Exclusive access (ensure no other users are on the node).
* **P-State:** Performance mode (usually locked by cluster admins, but good to note).

#### 2. Precision ($P$)
* **Type:** `bfloat16` (BF16).
* **Rationale:** Standard for LLaMA 3. `float32` is too slow/memory-heavy; `int8` (quantization) introduces kernel implementation variables that confuse the architectural comparison. BF16 ensures we measure the *model's* cost, not the quantization kernel's efficiency.

#### 3. Output Token Count ($N_{out}$)
* **Definition:** For LLaMA, we force it to generate exactly $X$ tokens. For LLaDA, we mask exactly $X$ tokens.
* **Constraint:** The model must generate the *same amount of text*. You cannot compare LLaDA generating a sentence vs. LLaMA generating a paragraph.

#### 4. Software Stack ($S$)
* **Python:** 3.13.2
* **PyTorch:** Version 2.4+ (Standard stable).
* **CUDA:** 12.6 (Loaded via ACCRE modules).
* **Attention Kernel:** FlashAttention-2 (Must be enabled for *both* or *neither*). *Recommendation: Enable for both to reflect modern usage.*

---

### III. Procedural Variables (The "Protocol")
How the measurement is taken.

#### 1. Warm-up Iterations ($W$)
* **Value:** 3 runs.
* **Purpose:** The first few runs of any PyTorch model invoke JIT compilation (Triton kernels/CUDA graphs) and memory allocation overhead. We discard these to measure stable performance.

#### 2. Sampling Temperature ($T$)
* **Value:** 0 (Greedy Decoding) or fixed small value (e.g., 0.1).
* **Purpose:** While temperature doesn't directly affect speed, it affects the output *distribution*. For the "Quality" metric, we want deterministic or near-deterministic outputs to ensure reproducibility.

---

### Summary Table for Report

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

Would you like me to move to the next step: writing the **Benchmarking Harness (`src/main.py`)** that implements these variables using `argparse`?