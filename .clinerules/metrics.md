To ensure your report is rigorous, you must define your metrics mathematically. This removes ambiguity and allows other researchers to replicate your exact methodology.

Since you are comparing two fundamentally different architectures—**Autoregressive (LLaMA)** vs. **Diffusion (LLaDA)**—standard metrics like "Time to First Token" (TTFT) apply differently. LLaMA generates tokens one by one; LLaDA typically refines the entire sequence simultaneously.

Here are the formal definitions you should use in your report.

### 1\. Latency Metrics

Latency measures the delay experienced by the user.

#### A. End-to-End Latency ($L_{e2e}$)

This is the total wall-clock time required to generate a full response. This is the **primary metric** for comparing the two models, as it is the only common denominator.

$$L_{e2e} = t_{end} - t_{start}$$

  * **For LLaMA:** The time from input ingestion until the `<EOS>` token is generated.
  * **For LLaDA:** The time from input ingestion until the final diffusion step is complete.
  * **Why it matters:** This determines if LLaDA is responsive enough for real-time chat.

#### B. Inter-Token Latency ($L_{itl}$) (LLaMA Only)

The average time elapsing between the appearance of two consecutive tokens.

$$L_{itl} = \frac{L_{e2e} - L_{prefill}}{N_{tokens} - 1}$$

  * **Why it matters:** LLaMA might have a high total latency for long texts, but low $L_{itl}$ means the user sees text appearing smoothly. LLaDA typically has no $L_{itl}$ (the text appears all at once), which may feel "slower" to a user even if $L_{e2e}$ is identical.

-----

### 2\. Throughput Metrics

Throughput measures the capacity of the system, critical for batch processing and serving costs.

#### Generation Throughput ($T_{gen}$)

The number of valid output tokens generated per second across the entire batch.

$$T_{gen} = \frac{B \times N_{out}}{L_{e2e}}$$

Where:

  * $B$ is the batch size.
  * $N_{out}$ is the average number of output tokens per sequence.
  * $L_{e2e}$ is the end-to-End latency for that batch.

**Hypothesis:** LLaDA may show superior throughput at high batch sizes because diffusion models often parallelize better than autoregressive models (which suffer from KV cache memory bottlenecks).

-----

### 3\. Memory Metrics

Memory consumption dictates the hardware requirements and the maximum batch size.

#### Peak Allocated VRAM ($M_{peak}$)

The maximum amount of GPU memory occupied by the model weights, activation states, and caches at any point during generation.

$$M_{peak} = \max_{t \in [0, T]} (M_{reserved}(t))$$

  * **LLaMA bottleneck:** The **KV Cache**. As sequence length ($S$) increases, memory usage grows linearly ($O(S)$).
  * **LLaDA bottleneck:** The **Activation Map**. Memory usage depends on the full sequence length from the start, but doesn't necessarily grow during the generation process.

-----

### 4\. The "Efficiency" Trade-off (The Critical Metric)

This is the novel metric that addresses your paper's specific goal. You need to normalize speed against quality.

#### Steps-to-Quality Parity ($S_{parity}$)

The number of diffusion steps ($k$) LLaDA requires to match the perplexity ($\text{PPL}$) of LLaMA 3.

Let $Q_{llama}$ be the baseline quality (e.g., BERTScore or Perplexity).
Let $f_{llada}(k)$ be the quality of LLaDA at $k$ steps.

$$S_{parity} = \min \{ k \mid f_{llada}(k) \geq Q_{llama} \}$$

Once you find $S_{parity}$, you compare the Latency:
$$\text{Speedup Factor} = \frac{L_{e2e}(\text{LLaMA})}{L_{e2e}(\text{LLaDA at } S_{parity})}$$

  * **Interpretation:** If the Speedup Factor is $>1$, LLaDA is viable. If $<1$, LLaDA is computationally inefficient for that quality level.

-----

### 5\. Implementation Note

To measure $L_{e2e}$ accurately on a GPU, **you cannot use Python's `time.time()`**. CUDA operations are asynchronous; Python will report the code has finished while the GPU is still working.

You must use `torch.cuda.Event`:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... run model generation ...
end_event.record()

torch.cuda.synchronize() # WAITS for GPU to finish
elapsed_time_ms = start_event.elapsed_time(end_event)
```

Would you like to proceed to generating the **Python Benchmark Harness (`src/main.py`)** using these exact `torch.cuda.Event` definitions?