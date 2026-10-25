# DS41 Metal Runtime

A model-specific Metal inference engine for **DeepSeek-V4.1-Flash-1017 GSQ-RCO** on Apple Silicon.
Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) via the
[unslothai](https://github.com/unslothai/llama.cpp) dsv41 branch, with every non-Metal backend removed.

The model is a 384-expert MoE whose routed experts (~130 GB) do not fit in RAM: they stay in the GGUF file
on SSD, are read on demand into a GPU slot arena, and are evicted with a decayed-frequency policy. The
dense weights (Q5_K / Q8_0) and the KV cache live on Metal.

## Hardware

Tested only on **Apple M5 Pro, 64 GB, macOS 27**. About 64 GB of RAM is needed (model, KV cache, arena);
other machines are untested.

## What this fork adds

| Feature | Flag | What it does |
|---|---|---|
| ISTA GGUF loading | (none) | Loads GSQ-RCO sets: HF-style keys, constant tensors, Engram sidecar |
| MoE slot arena | `LLAMA_DSV41_MOE_CACHE_MIB=<MiB\|auto>` | GPU-resident slots for routed experts, filled by pread misses |
| Arena auto-sizing | `LLAMA_DSV41_MOE_CACHE_MIB=auto` | Sizes the arena from RAM minus dense weights, KV and compute buffers (40 GiB cap) |
| Decayed-frequency eviction | `LLAMA_DSV41_MOE_EVICT=lfu-decay` | LFU with periodic decay instead of plain LRU |
| Profile seeding | `LLAMA_DSV41_MOE_SEED_PROFILE=<file>` | Seeds slot heat from a recorded routing profile |
| Heat file | `LLAMA_DSV41_MOE_HEAT_FILE`, `..._HEAT_SAVE` | Load/save the learned heat at start/exit |
| Next-layer expert prediction | `LLAMA_DSV41_MOE_PREDICT=1\|2` | Prefetches predicted misses ahead of the layer |
| Arena fill overlap | `LLAMA_DSV41_MOE_OVERLAP=1\|2` | Async arena fill overlapped with the shared expert and resident experts |
| Direct id readback | `LLAMA_DSV41_MOE_DIRECT=1` | Reads routing ids on device instead of a synchronous copy |
| Uncached arena fills | `LLAMA_DSV41_MOE_FILL_NOCACHE=1` | `F_NOCACHE` fills so expert pages do not evict the rest |
| Prefill expert staging | `LLAMA_DSV41_PREFILL_STAGE=1` | Reads the next layer's experts while the current layer computes |
| Prefill arena reads | `LLAMA_DSV41_PREFILL_ARENA=1` | Prompt reads expert rows from the arena when resident |
| Pread uploader | `LLAMA_DSV41_EXPS_OFFLOAD=1` | Reads expert tensors straight into a shared Metal buffer |
| Async expert prefetch | `LLAMA_DSV41_EXPS_PREFETCH=1` | Router-driven readahead of the selected experts |
| Q5_K decode kernels | `GGML_METAL_DS41_Q5K=1` | DS41 Q5_K mat-vec x1/x2 |
| IQ decode kernels | `GGML_METAL_DS41_IQ=1` | DS41 IQ1_M / IQ2_XXS / IQ2_XS `mul_mv_id` |
| IQ1_M v2 kernel | `GGML_METAL_DS41_IQ1M=1` | Integer grid-word decode, no value table |
| Decode fusions | `GGML_METAL_DS41_FUSE=1` | Fuses hc-gate, router and MoE-weight chains (-800 dispatches/token) |
| Wide index lists | `GGML_METAL_DS41_IDXW=1` | One threadgroup per mask row for decode attention at depth |
| Sparse prefill attention | `GGML_METAL_DSV4_SATTN=1\|2` | DS41 sparse attention kernels; 2 uses the tensor API |
| Op profiler | `GGML_METAL_OP_PROFILE=1` | Per-bucket GPU time, per-run command buffers |
| Dense checkpoints | `LLAMA_SERVER_CKPT_STEP=<tokens>` | Server checkpoints for mid-prompt prefix reuse |
| N-gram speculation | `LLAMA_SPEC_CTRL=1` | In-graph draft rollback plus a cost-aware draft-length controller |
| Routing trace | `LLAMA_DSV41_MOE_TRACE=<file>` | Writes the decode routing ids for offline policy work |

Every variable, its default and the value in the current best config: [docs/ENV.md](docs/ENV.md).

## Build

```bash
cmake -S . -B build -G Ninja \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_NATIVE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

Builds `llama-server`, `llama-cli`, `llama-completion`, `llama-bench`, `llama-perplexity`,
`llama-batched-bench` and `test-backend-ops`.

## Quick start

Presets and the model download live in [arnavprabhu/deepseek-v4-flash-mac](https://github.com/arnavprabhu/deepseek-v4-flash-mac).

```bash
./build/bin/llama-server -m <model.gguf> \
  -ngl 99 -ot 'ffn_(gate|up|down)_exps=CPU' -fa on -lm mmap -lzm on -t 6 \
  -ub 4096 -b 4096 --checkpoint-min-step 4096 \
  --spec-type ngram-mod --spec-ngram-mod-n-match 8 --spec-ngram-mod-n-min 2 --spec-ngram-mod-n-max 8
```

Set the environment flags from the table above; the current best set is listed in
[docs/ENV.md](docs/ENV.md).

## Numbers (v11, M5 Pro 64 GB)

| Metric | Value |
|---|---|
| pp4096 | ~262 tok/s A/B mean (278-283 tok/s on the warm first sample) |
| tg128 | 16.12 tok/s (A/B mean, +2.7% over 15.69) |
| copy/edit with n-gram speculation | 17.1-19.0 tok/s (was 9.9-11.6 without) |
| 128K context | TTFT 9.45 min (119,144 tokens), needle retrieved, no swap (measured before the v11 kernels) |

Details and history: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Credit and license

MIT. See [LICENSE](LICENSE) and [licenses/](licenses).

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) - upstream, MIT
- [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) - dsv41 base, MIT
- Strata - ported ideas (slot arena, decayed-frequency eviction), MIT
- Splash - ported ideas (overlap fill, direct id readback), Apache-2.0
