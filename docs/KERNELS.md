# DS41 Metal kernels

All kernels are gated by an environment variable, default off, compiled into the embedded Metal library
(`-DGGML_METAL_EMBED_LIBRARY=ON`) and cached as pipeline objects on first use. Numbers below are measured on
the M5 Pro 64 GB.

| Kernel | Flag | Source | Measured |
|---|---|---|---|
| Q5_K mat-vec | `GGML_METAL_DS41_Q5K=1` | `ggml/src/ggml-metal/kernels/mul_mv.metal` | with IQ: tg2048 10.35 -> 10.76 (+3.9%) |
| IQ mul_mv_id | `GGML_METAL_DS41_IQ=1` | `ggml/src/ggml-metal/kernels/mul_mv.metal` | ~3.7 ms/token of the 5-6 ms estimate |
| IQ1_M v2 | `GGML_METAL_DS41_IQ1M=1` | `ggml/src/ggml-metal/kernels/mul_mv.metal` | gate/up 85.2 -> 69.1 us (1.23x) |
| Decode fusions | `GGML_METAL_DS41_FUSE=1` | `kernels/misc.metal`, `ggml-metal-fusion.cpp` | -800 of 3064 dispatches/token |
| Wide index lists | `GGML_METAL_DS41_IDXW=1` | `kernels/fa_aux.metal` | 388 -> 87.4 us at 128K (4.4x) |
| Sparse attention v1 | `GGML_METAL_DSV4_SATTN=1` | `kernels/fa_aux.metal` | pp8192 207.6 -> 272.4 (+31%) |
| Sparse attention v2 | `GGML_METAL_DSV4_SATTN=2` | `kernels/fa_tensor.metal` | pp8192 +22.5%, pp32768 +27.6% |
| Op profiler | `GGML_METAL_OP_PROFILE=1` | `ggml-metal-context.m` | per-bucket GPU time |

## Q5_K mat-vec (`GGML_METAL_DS41_Q5K`)

Pipeline selection is in `ggml-metal-device.cpp` (`ggml_metal_ds41_q5k()`); the entry kernels are
`kernel_mul_mv_q5_K_f32_x1` and `_x2`, which instantiate `kernel_mul_mv_q5_K_f32_ds41_impl<1>` and `<4>`,
chosen by `ne00/256`, and require 16-byte aligned `src1` rows. They cover the dense Q5_K mat-vecs (attention
`q_b` / `out_a` / `out_b`, shared expert, head) at one row.

Measured with `GGML_METAL_DS41_IQ` in iter 006k, interleaved B-C-B on the dev profile: tg2048 10.345 / 10.361
-> **10.761 tok/s (+3.9%)**, fill / sync / hit rate unchanged, so the whole gain is non-fill time
(96.7 -> 92.9 ms/token). Per-kernel estimate from the microbenchmarks was ~2.1 ms/token, mostly `q_b`
(k = 1280, 5 blocks per row, stock kernel only 221 GB/s there vs 283-294 on the big shapes). Prefill is
unaffected: ub4096 uses `mul_mm` / `mul_mm_id`.

## IQ mul_mv_id (`GGML_METAL_DS41_IQ`)

`kernel_mul_mv_id_*_f32_x1` table kernels for IQ1_M / IQ2_XXS / IQ2_XS routed experts, plus the x2 variant
for `ffn_down` rows of 72 chunks (4 row groups of 8 lanes). Estimated saving from the microbenchmarks:
IQ1_M ~2.4 ms/token, IQ2_XXS ~0.9, IQ2_XS ~0.4. IQ2_XXS / IQ2_XS stay on this path: sign-mask and
constant-table variants measured no better (80-97 us vs 81 us), so they were not shipped.

## IQ1_M v2 (`GGML_METAL_DS41_IQ1M`)

`kernel_mul_mv_iq1_m_f32_v2_impl` decodes one raw `iq1s_grid_gpu` word per 8 weights with integer ops
(`(g >> s) & 0x000F000F | 0x64006400`, minus 1024 in half precision is exact), so no threadgroup
pattern/value tables and no fill.

| shape | x1 (iter 006k) | v2 | gain |
|---|---:|---:|---:|
| gate/up 2304 x 5120, 6 experts | 85.2 us, 181 GB/s | 69.1 us, 224 GB/s | 1.23x |
| gate/up, test-backend-ops | 84.9 us | 69.0 us | 1.23x |
| down 5120 x 2304 | 81.3 us, 191 GB/s | 74.4 us, 208 GB/s | 1.09x |
| down, test-backend-ops | 80.8 us | 71.0 us, 218 GB/s | 1.14x |

The 1-5 expert rows that matter for `MOE_OVERLAP=2` gain 1.05-1.37x. End to end, IQ1M + FUSE + IDXW gave
tg128 **15.69 -> 16.12 tok/s (+2.7%)**; IQ1M alone was +0.5% (noise). Negative variants (uchar4 -> float4,
16-weight lanes, NR0 16) were slower and were dropped.

## Decode fusions (`GGML_METAL_DS41_FUSE`)

Patterns in the Metal fusion table (`ggml-metal-fusion.cpp`), kernels in `kernels/misc.metal`:

| pattern | kernel | chains/token | dispatches saved |
|---|---|---:|---:|
| `MUL(view(mixes), scale) + ADD(base) + sigmoid + SCALE` (hc pre/post gates) | `kernel_ds41_hc_gate_f32` | 160 | 480 |
| `softplus + SQRT + ADD(bias)` (router + next-layer predictor) | `kernel_ds41_router_f32` | 79 | 158 |
| `GET_ROWS + SUM_ROWS + SCALE + DIV + SCALE` (MoE routing weights) | `kernel_ds41_moe_w_f32` | 40 | 160 |

A decode token was 3064 dispatches; the fusions remove ~800 of them. Every kernel uses the same expressions
as the unfused path, so the greedy goldens stay byte-identical. The hc-gate kernel disables fp contraction.

## Wide index lists (`GGML_METAL_DS41_IDXW`)

`kernel_flash_attn_ext_vec_idx_w` in `kernels/fa_aux.metal`, dispatched from `ggml-metal-ops.cpp` only when
the mask rows are 8-byte aligned, there are at most 64 of them and at least 8192 entries - i.e. decode at
depth. One 32-simdgroup threadgroup per row instead of one simdgroup per row (which needs 1025 dependent
steps for a 131200-entry row).

| KV rows | 128 listed | 640 listed |
|---|---:|---:|
| 131200 (before) | 388 us | 427-457 us |
| 131200 (IDXW) | **87.4 us** (4.4x) | **132-134 us** (3.3x) |

That is ~0.3 ms per layer, roughly 10-12 ms per token over 40 layers at 128K depth. It does nothing at
fast-bench depth (the fast bench runs at depth 0), which is why the end-to-end fast-bench gain comes from
IQ1M + FUSE.

## Sparse prefill attention (`GGML_METAL_DSV4_SATTN`)

64 query heads over one KV head, `dk = dv = 512`, F16 KV, over the index lists of the sparse window.

- `=1`: `kernel_dsv4_sparse_attn_f16` (`kernels/fa_aux.metal`), simdgroup 8x8 matrices, 16 heads per
  threadgroup. pp8192 207.6 -> **272.4 tok/s (+31%)**.
- `=2`: `kernel_dsv4_sparse_attn_v2_f16` (`kernels/fa_tensor.metal`), matmul2d with K/V gathered into
  cooperative tensors, 32 heads per threadgroup, plus a coalesced index-list builder (one simdgroup per mask
  row, 8 rows per threadgroup, 8-byte aligned rows). Falls back to 1 when the tensor API is unavailable.
  Interleaved A/B, same binary: pp8192 225.0 / 220.2 -> **267.0 / 277.9 (+22.5%)**, pp32768
  197.4 -> **251.7 (+27.6%)**. 128K TTFT 32.5 -> **24.4 min (-25%)** in the memory-tight run.
  Numerics: 8K KLD 5.19e-4, same-top-p 99.45%, dPPL +0.013%; all 4996 FLASH_ATTN_EXT cases pass.

## Op profiler (`GGML_METAL_OP_PROFILE`)

Enabled in `ggml-metal-context.m`: switches to per-run command buffers and single-threaded encoding so each
op can be timed, and reports per-bucket GPU time. Diagnostic only - it changes scheduling, so its absolute
numbers are pessimistic; use the ratios.

## Tests

`test-backend-ops -o MUL_MAT_ID -b MTL0` exercises the expert path; all 885 cases pass on MTL0. The DS41
fusions have their own cases (`DS41_MOE_W`, `DS41_ROUTER`, `DS41_HC_GATE`, `DS41_NEXP`), and `DS41_CASES=1`
adds the router `gate_inp` (BF16 384x5120) and `hc_fn` (F32 24x20480) mat-vec perf cases.
