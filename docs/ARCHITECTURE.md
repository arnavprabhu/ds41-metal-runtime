# Architecture

How one token moves through the fork. All file references are relative to the repo root.

## What lives where

| Part | Size | Where | Read by |
|---|---|---|---|
| Dense weights (Q5_K, Q8_0) | ~10-15 GB | Metal buffers | Metal kernels |
| Routed experts (IQ1_M / IQ2_XXS / IQ2_XS) | ~130 GB | GGUF file on SSD, CPU buffers | pread into the arena, or CPU |
| KV cache / compressed KV | ~0.43 GB at 131K | Metal | sparse attention |
| Engram table (layers 1, 14) | ~0.2 GB | host, gathered per token | host gather, then Metal |

Expert tensors never reach device memory by default: `src/llama-model-loader.cpp` refuses to place
`ffn_{gate,up,down}_exps` on a non-host buffer unless `LLAMA_DSV41_EXPS_ON_DEVICE=1`. The model is loaded
with `-ot 'ffn_(gate|up|down)_exps=CPU'`, so the expert tensors are mmap'd (`-lm mmap`) and touched only by
the reader that needs them.

## Decode

1. **Graph build** (`src/models/deepseek41.cpp`). The dense part is ordinary DeepSeek-V4.1: attention over
   the compressed KV, the shared expert (Q5_K gate/up/down), the router.
2. **Router**. `gate_inp` (BF16 384x5120) scores the 384 experts. The routing-weight chain
   (`get_rows + sum_rows + scale + div + scale`) is fused into `kernel_ds41_moe_w_f32` when
   `GGML_METAL_DS41_FUSE=1`.
3. **Arena lookup** (`src/llama-moe-cache.cpp`). Each routed expert maps to a slot in its layer group. A
   hit runs `MUL_MAT_ID` on the arena buffer; a miss schedules a fill.
4. **Miss fill**. With the pread uploader (`LLAMA_DSV41_EXPS_OFFLOAD=1`), the expert tensor range is read
   from the GGUF file directly into the shared Metal arena, chunked at
   `LLAMA_DSV41_UPLOAD_CHUNK_KIB` (1024 KiB here) by `LLAMA_DSV41_UPLOAD_THREADS` threads.
5. **Overlap** (`LLAMA_DSV41_MOE_OVERLAP=2`). The fill of not-yet-resident experts runs asynchronously
   while the shared expert and the already resident experts compute, so SSD latency hides behind GPU work.
6. **Readback** (`LLAMA_DSV41_MOE_DIRECT=1`). Routing ids are read on device instead of a synchronous copy.
7. **Prediction** (`LLAMA_DSV41_MOE_PREDICT=2`). The next layer's experts are predicted from the current
   router scores; `LLAMA_DSV41_PREDICT_K` (8 here) limits candidates and `LLAMA_DSV41_PF_N` (4 here) the
   prefetches launched per layer. Predicted victims get a heat bonus (`LLAMA_DSV41_PF_BONUS`).
8. **Attention**. Decode uses sparse attention over the compressed KV with the index lists built by the
   SATTN=2 path; at depth, `GGML_METAL_DS41_IDXW=1` switches the list builder to one threadgroup per mask
   row (`kernel_flash_attn_ext_vec_idx_w`).

## The slot arena

`llama_moe_cache` (`src/llama-moe-cache.cpp`, policy in `src/llama-moe-cache-policy.h`) owns one Metal
buffer per device, cut into slots per layer group. Sizing rules:

- `LLAMA_DSV41_MOE_CACHE_MIB=<MiB>` sets the size directly (0 keeps the cache off).
- `LLAMA_DSV41_MOE_CACHE_MIB=auto` allocates at `LLAMA_DSV41_MOE_CACHE_MAX_MIB` (default 40960 MiB) first,
  because Metal wires the whole buffer: the graphs are reserved once with a small probe arena
  (`LLAMA_DSV41_MOE_PROBE_MIB`, 2048 MiB), then `llama_context::moe_cache_auto_size()` computes
  `RAM * LLAMA_DSV41_MOE_RAM_FRAC (0.90) - dense - KV - compute - LLAMA_DSV41_MOE_MARGIN_MIB (3072)`,
  clamped to `[MOE_CACHE_MIN_MIB (4096), MOE_CACHE_MAX_MIB]`, and the arena is reallocated at that size.
- Slots are split across devices by expert bytes, so each group caches the same fraction of its experts.

Eviction is `LLAMA_DSV41_MOE_EVICT=lfu-decay`: LFU with an exponential decay of the per-slot heat every
`LLAMA_DSV41_MOE_DECAY_TOKENS` (256) tokens by `LLAMA_DSV41_MOE_DECAY` (0.9). `LLAMA_DSV41_MOE_SEED_PROFILE`
preloads the heat from a routing profile (384-expert sets only); `LLAMA_DSV41_MOE_HEAT_FILE` /
`LLAMA_DSV41_MOE_HEAT_SAVE` persist it. `LLAMA_DSV41_MOE_MLOCK=1` mlock's the arena.
`LLAMA_DSV41_MOE_WARM=1` fills empty slots in the background with the hottest cold experts, up to
`LLAMA_DSV41_WARM_INFLIGHT` (12) in flight.

## Prefill

Prefill reads every expert anyway, so it does not use the arena by default. Two mechanisms overlap that I/O
with compute:

1. **Staging** (`src/llama-context.cpp`, pool in `src/llama-exps-prefetch.cpp`). For ubatches with at least
   `LLAMA_DSV41_STAGE_MIN_TOKENS` (1024) rows, the staging readers copy the *next* layer's expert ranges into
   a staging buffer while the current layer computes. `LLAMA_DSV41_STAGE_THREADS` (16 here) sets the pool
   size. After the prompt, `LLAMA_DSV41_STAGE_RELEASE` (default on) returns the staging memory.
2. **Arena reads** (`LLAMA_DSV41_PREFILL_ARENA=1`). When an expert is already resident, the prompt reader
   takes it from the arena instead of the file, which avoids repeat reads of hot experts.

Prefill attention is the sparse DS41 kernel (`GGML_METAL_DSV4_SATTN=2`,
`kernel_dsv4_sparse_attn_v2_f16`): 64 query heads over one KV head, `dk = dv = 512`, F16 KV, over the index
lists built for the sparse window. Batches use `-ub 4096 -b 4096` so each ubatch pays one pass over the
experts.

The mmap readahead (`LLAMA_DSV41_EXPS_PREFETCH`) is skipped for ubatches at or above
`GGML_OP_OFFLOAD_MIN_BATCH` (1 here) when `LLAMA_DSV41_PREFETCH_OFFLOADED=0`, because the pread uploader is
then the only reader and a second read would only cost bandwidth.

## Engram

DeepSeek-V4.1 has an Engram table on two layers. The default path gathers rows on the host in `set_input`
(`LLAMA_DSV41_HOST_GATHER` on): token hash, readahead, IQ4_NL dequant to F32, upload of `[24*256]` floats per
token. The graph only sees the input tensor, so there is no CPU split for Engram. `LLAMA_DSV41_ENGRAM_OFF=1`
disables the contribution (debug only, outputs are then not the model's).

## Speculation and checkpoints

- `LLAMA_SPEC_CTRL=1` enables the cost-aware draft controller in `tools/server/spec-cost.h`. Drafts come
  from the n-gram-mod drafter (`--spec-type ngram-mod`); `n_rs_seq` lets the graph roll back rejected drafts
  in place. The controller estimates acceptance (decayed beta-binomial per draft position) and the cost of an
  n-row step (concave in n, because extra rows reuse the same expert union), and skips drafting when even the
  best k does not beat plain decode by `LLAMA_SPEC_CTRL_MIN_GAIN` (0.05).
- `LLAMA_SERVER_CKPT_STEP=<tokens>` writes dense context checkpoints mid-prompt so a later request that
  shares the prefix can resume from one instead of re-reading the prompt.

## Threads and I/O pools

| Pool | Env | Default | Current |
|---|---|---|---|
| Demand upload | `LLAMA_DSV41_UPLOAD_THREADS` | 12 | 16 |
| Staging | `LLAMA_DSV41_STAGE_THREADS` | 8 | 16 |
| Prefetch (readahead) | `LLAMA_DSV41_PREFETCH_THREADS` | 4 | 4 |
| Arena prefetch (predict) | `LLAMA_DSV41_PF_THREADS` | 6 | 6 |

See [ENV.md](ENV.md) for the full list and [KERNELS.md](KERNELS.md) for the Metal side.
