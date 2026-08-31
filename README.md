# Hybrid Streaming Attention

A Google Colab project that combines ideas from **StreamingLLM** and **DuoAttention** to make long-context LLM inference more memory-efficient without fully losing the ability to retrieve information from early in a prompt.

## Goal

Large language models store key-value (KV) states for every generated token. As context grows, this KV cache becomes the main memory bottleneck.

This project compares three cache strategies:

| Method | Cache policy | Expected result |
|---|---|---|
| Dense cache | Keeps every token for every attention head | Best retrieval, highest memory |
| StreamingLLM | Keeps sink tokens and recent tokens for every head | Low memory, weaker distant recall |
| Hybrid cache | Full cache for selected retrieval heads; StreamingLLM cache for other heads | Better distant recall than StreamingLLM with less memory than dense caching |

## Research motivation

### First approach: StreamingLLM

StreamingLLM shows that a language model can continue processing very long streams with a fixed-size KV cache. Instead of storing all previous tokens, it keeps:

- The first few tokens, called **attention sinks**
- A fixed recent-token window
- No middle-token KV states

This keeps memory bounded, but it can remove facts that appear far back in the context.

### Improvement: DuoAttention-inspired hybrid

DuoAttention observes that not all attention heads need the same history:

- **Retrieval heads** may need old context to recover specific facts.
- **Streaming heads** mainly rely on sink tokens and recent context.

This project uses a lightweight alternative to the original DuoAttention calibration process. A needle-in-a-haystack probe measures which heads focus most on a hidden fact. The strongest heads are selected as retrieval heads.

During generation:

- Retrieval heads keep the full KV cache.
- Streaming heads keep 4 sink tokens and a 512-token recent window.

This creates a hybrid cache policy.

## Architecture

```text
Long prompt with hidden fact
            |
            v
      Calibration probe
            |
            v
Select top retrieval heads per layer
            |
            v
      Hybrid KV-cache policy
      /                    \
Retrieval heads        Streaming heads
Full KV history        Sink + recent window
      \                    /
            v
       LLM generation
            |
            v
Memory, speed, and retrieval evaluation
```

## Model and environment

- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Runtime: Google Colab GPU
- Precision: 4-bit NF4 quantization
- Attention backend: PyTorch SDPA
- Cache API: Hugging Face `Cache` / `DynamicCache`
- Evaluation: synthetic needle-in-a-haystack retrieval prompts

A 3B parameter model was selected to leave enough GPU memory for long-context cache comparisons on free Colab GPUs.

## Experiments

Each experiment uses the same prompt, generation settings, random seed, and needle position.

### 1. Dense cache baseline

Uses the default full KV cache. This provides the reference quality result and shows the memory cost of preserving all attention history.

### 2. StreamingLLM baseline

All heads use the same bounded cache:

- Sink tokens: 4
- Recent-token window: 512
- Middle tokens: evicted

### 3. Probe-selected hybrid cache

A small set of heads per layer is selected from calibration probes.

- Retrieval heads: full KV history
- Streaming heads: 4 sink tokens + 512 recent tokens
- Retrieval-head ratio: 25% per layer

## Evaluation metrics

| Metric | Why it matters |
|---|---|
| Needle retrieval accuracy | Tests whether the model can recall a fact placed deep in context |
| Peak GPU memory | Measures the practical KV-cache memory cost |
| Cache length | Confirms whether eviction is working |
| Prefill time | Measures prompt-processing speed |
| Decode tokens/second | Measures generation speed |

Needles are placed at 10%, 50%, and 90% of the input context to test shallow, middle, and deep retrieval.

## Results

Replace this table with values produced by `results/metrics.csv`.

| Method | Deep retrieval accuracy | Peak VRAM | Decode speed |
|---|---:|---:|---:|
| Dense | [value] | [value] GB | [value] tok/s |
| StreamingLLM | [value] | [value] GB | [value] tok/s |
| Hybrid | [value] | [value] GB | [value] tok/s |

![Peak GPU memory comparison](results/memory_comparison.png)

![Needle retrieval comparison](results/retrieval_comparison.png)

```

## Reproduce

1. Open `hybrid_streaming_attention.ipynb` in Google Colab.
2. Enable a GPU runtime.
3. Run all cells in order.
4. Review the result table and charts saved in `results/`.
5. Commit the updated notebook and result files to GitHub.

## Limitations

This is not a full reproduction of DuoAttention. The original work uses a larger offline calibration procedure and optimized inference implementation. This project is a Colab-friendly prototype that tests the same central hypothesis: retaining full history only for retrieval-focused heads can improve long-range recall over uniform cache eviction.

## References

- Xiao et al. *Efficient Streaming Language Models with Attention Sinks* (StreamingLLM), ICLR 2024.
- Xiao et al. *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads*, ICLR 2025.
