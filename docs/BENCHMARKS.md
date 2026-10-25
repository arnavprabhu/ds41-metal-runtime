# Benchmark history: v8 to v11

M5 Pro 64 GB. Two profiles in the driver tree: the *dev* profile (pp8192 / tg2048) and the *fast* profile
(pp4096 / tg128). Numbers are the recorded runs, not re-measured here.

## Where the speed comes from

| Stage | pp8192 | tg2048 | What changed |
|---|---:|---:|---|
| baseline-v1 (ISTA shim, experts on CPU) | 5.72 (pp512) | 3.27 (tg64) | first working Metal runtime |
| iter 003 (Metal experts, pread uploader, ub2048) | 96.97 (pp512) | 7.98 (tg64) | 130.5 pp4096, 7.67 tg512 |
| iter 004 (8 GiB slot arena) | 131.1 | 10.59 | +38% decode |
| iter 005 (ub 4096) | 155.3 | 9.31 | config only |
| iter 005 (prefill staging) | 214.5 | ~9.7 | +34% prefill |
| iter 005 (SATTN=1) | 272.4 | - | +31% over 207.6 |
| baseline-v7 | 214.5 | ~9.7 | final v7 state |
| v8 | 213-222 | 16.28 | kernels + lfu-decay + seeding + checkpoints, 36 GiB arena |
| v11 | - | ~16 (tg128) | + IQ1M + FUSE + IDXW; pp4096 ~262 |

The v7 -> v8 jump (tg2048 ~9.7 -> 16.28) is the arena policy: 36 GiB LFU with decay and profile seeding
instead of an 8/20 GiB LRU, plus the DS41 decode kernels.

## v8

- Arena: 36 GiB, `LLAMA_DSV41_MOE_EVICT=lfu-decay`, seeded from a routing profile.
- DS41 decode kernels: Q5K + IQ (tg2048 +3.9%, interleaved B-C-B).
- Server dense checkpoints, prefill staging, SATTN.
- Fast gate PASS: 131072-ctx server starts, swap delta 0.

## v9 / v10

- Decode fusions (`GGML_METAL_DS41_FUSE`): hc gate, router scores, MoE weights. -800 of 3064
  dispatches per token.
- SATTN=2 (matmul2d sparse attention): pp8192 +22.5%, pp32768 +27.6%, 128K TTFT -25%.
- Fast bench moved tg128 15.69 -> 16.12 (+2.7%) across IQ1M + FUSE + IDXW.

## v11 (current)

- IQ1_M v2 (integer grid-word decode), FUSE, IDXW, SATTN=2, n-gram speculation with the cost controller.
- Fast bench: pp4096 ~262 tok/s A/B mean (278-283 first sample), tg128 16.12 tok/s.
- Copy/edit prompts with n-gram-mod speculation: 9.88 -> 17.13 (copy0), 11.55 -> 18.88 (copy1),
  11.33 -> 18.12 (edit) tok/s; controller roughly equal to fixed m=8 on high-acceptance copy and better on
  mixed prompts.
- 128K: TTFT 10.5 min (188.6 tok/s, 119,144 tokens), needle retrieved, no swap, 26.1 GB peak RSS.

| fast bench, same binary, interleaved | control | candidate |
|---|---:|---:|
| tg128 (3 pairs, mean) | 15.69 | **16.12 (+2.7%)** |
| pp4096 (1 pair) | 263.0 | 261.5 (-0.6%, noise) |

## Real-text decode (server, specbench)

| Setting | tok/s |
|---|---:|
| greedy, 8 GiB arena | 9.89 |
| greedy, 20 GiB arena | 10.84 |
| greedy, 20 GiB + `MOE_FILL_NOCACHE` | 11.05 |
| recommended sampling (temp 1.0 / top_p 1.0, thinking 50) | 11.13 |

## Tried and dropped

| Attempt | Result |
|---|---|
| Long-prompt layer skip (`LLAMA_DSV41_PROMPT_SKIP=1`) | Exact in real arithmetic (KLD ~1e-5), but pp4096 showed no gain: the head ubatch is bounded by staging I/O, and a gain only appears when the baseline needs a second ubatch. Not kept. |
| 64-entry expert tiles (`kernel_mul_mm_id_t64`) | Correct (885/885) but slower: best variant 7.66 ms vs 6.08 ms for the 32-entry kernel, because halving the threadgroups in flight outweighs the shared dequant. |
| MQA heads-as-queries sparse attention (`GGML_METAL_FA_MQA`) | Slower (the vec kernel is compute-bound); kept out of the config. |
| IQ2 sign masks / constant grid tables | No gain (80-97 us vs 81 us). |
| IQ1_M float4 decode, 16-weight lanes, NR0 16 | Slower or register spills. |
| KV cache Q8 | Not worth it: compressed KV is only ~0.43 GiB at 131072. |
| Next-layer prefetch on by default | Recall 65%, extra reads cost the bandwidth they save; left opt-in (`LLAMA_DSV41_PREFETCH_NEXT`). |
| Stack knob sweep (`cand-stack-*` configs) | Swept; the kept set is `current_best.json`. |

## Gates

- Lossless: byte-exact goldens + 24 task outputs at 131072 ctx, KLD <= 1e-4.
- Numeric: KLD <= 5e-3, top-p >= 97%, |dPPL| <= 0.5% vs v1, tasks >= baseline-1.
