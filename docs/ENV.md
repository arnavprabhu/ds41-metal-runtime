# Environment variables

Defaults come from the source; "current" is the value in `work/bench/config/current_best.json` (unset means
the default applies). Variables are read once, usually at context or cache construction.

## MoE slot arena

| Variable | Default | What it does | Current |
|---|---|---|---|
| `LLAMA_DSV41_MOE_CACHE_MIB` | `0` (off) | Size of the arena in MiB, or `auto` for the auto-sizer | `auto` |
| `LLAMA_DSV41_MOE_CACHE_MAX_MIB` | `40960` | Upper bound for `auto` (and the size used for the first allocation) | - |
| `LLAMA_DSV41_MOE_CACHE_MIN_MIB` | `4096` | Lower bound for `auto` | - |
| `LLAMA_DSV41_MOE_RAM_FRAC` | `0.90` | Fraction of physical RAM the arena may use | - |
| `LLAMA_DSV41_MOE_MARGIN_MIB` | `3072` | MiB kept free on top of dense + KV + compute | - |
| `LLAMA_DSV41_MOE_PROBE_MIB` | `2048` | Probe arena used to reserve the graphs before the final size is known | - |
| `LLAMA_DSV41_MOE_EVICT` | `lru` | `lru` or `lfu-decay` | `lfu-decay` |
| `LLAMA_DSV41_MOE_DECAY` | `0.9` | Heat decay factor per decay period (0 < x < 1) | - |
| `LLAMA_DSV41_MOE_DECAY_TOKENS` | `256` | Tokens between decay steps | - |
| `LLAMA_DSV41_MOE_SEED_PROFILE` | off | Seed slot heat from a routing profile (384-expert sets only) | `.../profile-2.00.bin` |
| `LLAMA_DSV41_MOE_HEAT_FILE` | off | Load a saved heat file | - |
| `LLAMA_DSV41_MOE_HEAT_SAVE` | off | Path to save the heat on exit | - |
| `LLAMA_DSV41_MOE_WARM` | off | Background fill of empty slots with the hottest cold experts | - |
| `LLAMA_DSV41_WARM_INFLIGHT` | `12` | Max concurrent warm fills | - |
| `LLAMA_DSV41_MOE_OVERLAP` | `0` | `1`/`2`: async arena fill overlapped with the shared expert / resident experts (needs the pread uploader) | `2` |
| `LLAMA_DSV41_MOE_DIRECT` | off | Read routing ids on device instead of a synchronous copy | `1` |
| `LLAMA_DSV41_MOE_FILL_NOCACHE` | off | `F_NOCACHE` on arena fills so expert pages do not evict the rest | `1` |
| `LLAMA_DSV41_MOE_MLOCK` | off | `mlock` the arena | - |
| `LLAMA_DSV41_MOE_TRACE` | off | Write `(layer, n_ids, ids...)` per decode call for offline policy work | - |
| `LLAMA_DSV41_MOE_PREDICT` | `0` | `1`: predict next-layer experts; `2`: look-ahead prefetch queue | `2` |
| `LLAMA_DSV41_PREDICT_K` | `6` (clamped 1..16) | Candidates for the next-layer prediction | `8` |
| `LLAMA_DSV41_PF_N` | `16` | Predicted experts prefetched per layer | `4` |
| `LLAMA_DSV41_PF_BONUS` | `1.0` | Heat bonus given to prefetched victims | - |

## Expert I/O

| Variable | Default | What it does | Current |
|---|---|---|---|
| `LLAMA_DSV41_EXPS_PREFETCH` | `0` | Router-driven readahead of the selected experts | `1` |
| `LLAMA_DSV41_EXPS_OFFLOAD` | `0` | Enable the pread uploader into a shared Metal buffer | `1` |
| `LLAMA_DSV41_UPLOAD_PREAD` | `1` | With `EXPS_OFFLOAD`: use pread (else the CPU path) | - |
| `LLAMA_DSV41_UPLOAD_CHUNK_KIB` | `2048` | Chunk size of upload jobs (min 64 KiB) | `1024` |
| `LLAMA_DSV41_UPLOAD_THREADS` | `12` | Demand upload threads (caller + N-1 workers) | `16` |
| `LLAMA_DSV41_STAGE_THREADS` | `8` | Background staging (prefill) threads | `16` |
| `LLAMA_DSV41_PREFETCH_THREADS` | `4` | Readahead worker threads | - |
| `LLAMA_DSV41_PF_THREADS` | `6` | Arena prefetch (prediction) workers | - |
| `LLAMA_DSV41_PREFETCH_STATS` | `0` | Print prefetch/upload stats at exit | - |
| `LLAMA_DSV41_PREFETCH_NEXT` | `0` | Prefetch the next layer's experts for ubatches >= min | - |
| `LLAMA_DSV41_PREFETCH_NEXT_MIN` | `32` | Minimum ubatch rows for `PREFETCH_NEXT` | - |
| `LLAMA_DSV41_PREFETCH_OFFLOADED` | on (`1`) | `0` skips the readahead when op offload is active (the uploader is the only reader) | `0` |
| `GGML_OP_OFFLOAD_MIN_BATCH` | `32` | Batch size from which ops are offloaded (related, used with the above) | `1` |

## Prefill

| Variable | Default | What it does | Current |
|---|---|---|---|
| `LLAMA_DSV41_PREFILL_STAGE` | off | Stage the next layer's experts during prefill | `1` |
| `LLAMA_DSV41_STAGE_MIN_TOKENS` | `1024` | Minimum ubatch rows for staging | - |
| `LLAMA_DSV41_STAGE_RELEASE` | on (`1`) | Release staging memory after the prompt (0 keeps it) | - |
| `LLAMA_DSV41_PREFILL_ARENA` | off | Prompt reads expert rows from the arena when resident | `1` |

## Model layout, Engram, KV

| Variable | Default | What it does | Current |
|---|---|---|---|
| `LLAMA_DSV41_EXPS_ON_DEVICE` | off | Allow expert tensors on device buffers (refused otherwise) | - |
| `LLAMA_DSV41_SPARSE` | on | Sparse DS41 attention; `0` restores the reference path | - |
| `LLAMA_DSV41_HOST_GATHER` | on (`1`) | Gather Engram/token rows on the host; `0` uses in-graph `GET_ROWS` on host tables | - |
| `LLAMA_DSV41_ENGRAM_OFF` | off | Disable Engram entirely (debug; outputs are not the model's) | - |
| `LLAMA_DSV41_ENGRAM_FILE` | from GGUF | Override the Engram sidecar path (ISTA layout only) | - |
| `LLAMA_DSV41_ENGRAM_STATS` | off | Print Engram timing at exit | - |
| `LLAMA_DSV41_DUMP_ENGRAM` | off | Append hashed Engram row ids per token/layer for parity tests | - |
| `LLAMA_DSV41_KV_MIRROR` | on (`1`) | `0` disables KV mirrors; consumers read the source cache across devices | - |

## Metal kernels and profiling

| Variable | Default | What it does | Current |
|---|---|---|---|
| `GGML_METAL_DS41_Q5K` | off | DS41 Q5_K mat-vec x1/x2 | `1` |
| `GGML_METAL_DS41_IQ` | off | DS41 IQ1_M / IQ2_XXS / IQ2_XS `mul_mv_id` | `1` |
| `GGML_METAL_DS41_IQ1M` | off | IQ1_M v2 integer grid-word decode | `1` |
| `GGML_METAL_DS41_FUSE` | off | hc-gate / router / MoE-weight fusions | `1` |
| `GGML_METAL_DS41_IDXW` | off | Wide sparse index-list builder (decode at depth) | `1` |
| `GGML_METAL_DSV4_SATTN` | `0` | Sparse prefill attention: `1` (simdgroup) or `2` (matmul2d) | `2` |
| `GGML_METAL_OP_PROFILE` | off | Per-bucket GPU timing, per-run command buffers | - |

## Speculation

| Variable | Default | What it does | Current |
|---|---|---|---|
| `LLAMA_SPEC_CTRL` | `0` | Enable the cost-aware draft controller (`> 0`) | `1` |
| `LLAMA_SPEC_CTRL_MIN_GAIN` | `0.05` | Skip drafting unless the best k beats plain decode by this fraction | - |
| `LLAMA_SPEC_CTRL_TRACE` | `0` | Log controller decisions | - |
| `LLAMA_SPEC_CTRL_PRIOR_A1` | `0.8` | Prior for the first-position acceptance ratio | - |
| `LLAMA_SPEC_RS_MAX` | `16` | Cap on the `n_rs_seq` rollback rows | - |

## Server

| Variable | Default | What it does | Current |
|---|---|---|---|
| `LLAMA_SERVER_CKPT_STEP` | `0` (off) | Dense context checkpoint every N prompt tokens, for prefix reuse | `4096` |
