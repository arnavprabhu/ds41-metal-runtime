// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_cgraph;

enum ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(int debug);
void ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add src or dst ranges to track
bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// reorder the nodes in the graph to improve concurrency, while respecting the fusions of a device with props
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf);

// mat-mat vs mat-vec dispatch; used by both supports_op and ggml_metal_op_mul_mat*
bool ggml_metal_op_mul_mat_use_fwht (const struct ggml_tensor * op, size_t max_tg_mem);
bool ggml_metal_op_mul_mat_use_mm   (const struct ggml_tensor * op, bool has_simdgroup_mm);
bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm);

bool ggml_metal_op_mul_mat_use_mma    (const struct ggml_tensor * op);
bool ggml_metal_op_mul_mat_may_use_mma(const struct ggml_tensor * op); // graph structure only

// the few-row MMA kernel for a src0 type and rt src1 tiles: per 32-weight block (q4_0, q8_0 with one tile), q5_K, or the generic 64-weight chunk kernel
enum ggml_metal_mma_kind { GGML_METAL_MMA_KIND_BLK, GGML_METAL_MMA_KIND_Q5_K, GGML_METAL_MMA_KIND_GEN };
enum ggml_metal_mma_kind ggml_metal_mul_mv_mma_kind(enum ggml_type type, int rt);

// the src1 tiles of the few-row MMA kernels for mat-mul op: one 8-row tile, or two above 8 rows
int ggml_metal_mul_mv_mma_rt(const struct ggml_tensor * op);

// the weights of K per simdgroup step of the few-row MMA kernel for a src0 type and rt src1 tiles, 0 if none takes the type
int64_t ggml_metal_mul_mv_mma_k_step(enum ggml_type type, int rt);

// GGML_METAL_DS41_PFUSE=1: DS41 prefill traffic fusions (in-place RoPE slice dispatch, fused MoE reduce + hc_post,
// strided wo_a store). Off by default; the graph-side half reads the same variable.
bool ggml_metal_ds41_pfuse(void);

// GGML_METAL_DS41_MMID=1: routed-expert gate + up + SWIGLU in one mat-mat dispatch (MUL_MAT_ID x2 + GLU fusion)
bool ggml_metal_ds41_mmid(void);

// DS41 decode fusions (iter 016k), each off by default:
// GGML_METAL_DS41_MVGLU=1: routed-expert gate + up + GLU in one mat-vec dispatch (MUL_MAT_ID x2 + GLU, < 32 tokens)
bool ggml_metal_ds41_mvglu(void);
// GGML_METAL_DS41_MVMOE=1: routed-expert down + router-weight MUL + expert ADD chain in one mat-vec dispatch
bool ggml_metal_ds41_mvmoe(void);
// GGML_METAL_DS41_RTOPK=1: router sqrt-softplus + bias + argsort + MoE weight gather/normalize in one dispatch
bool ggml_metal_ds41_rtopk(void);

#ifdef __cplusplus
}
#endif
