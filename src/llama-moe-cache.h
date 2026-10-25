#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // LLAMA_DSV41_MOE_OVERLAP=1|2 (decode of one token): the hit map (first split of the layer) and the fill map (second
    // split) of layer il in place of the slot map, false if the layer uses the slot map. Level 1: the fill map is the full
    // slot map; level 2: it only holds the experts the fill brings, the hit map the experts already in the arena
    bool get_slot_maps(int32_t il, int64_t n_tokens, int64_t n_expert_used, ggml_tensor ** h, ggml_tensor ** m) const;
    int  overlap_level() const;

    // true for the host weights whose copy must not wait for the running split (the fill maps)
    bool copy_nosync(const ggml_tensor * src) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // bytes of the arena slots in use, and shrinking it (before the first graph is computed) to at most target bytes
    size_t active_bytes() const;
    void   shrink(size_t target);

    // pointer to expert e of the host tensor w inside the arena if it is resident (CPU-addressable shared memory), else nullptr
    const void * resident_expert(const ggml_tensor * w, int64_t e);

    // the graph about to be computed: picks up the next-layer prediction tensors (LLAMA_DSV41_MOE_PREDICT=1)
    void set_graph(ggml_cgraph * gf);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// LLAMA_DSV41_MOE_PREDICT=1: decode graphs compute layer il+1's router on layer il's FFN input, and the cache
// prefetches the predicted misses in the background (residency only, outputs unchanged)
bool llama_moe_cache_predict_enabled();
// 1 = old (prefetch after the fill of the layer, background thread), 2 = look-ahead into the prefetch queue at the plan of the layer
int llama_moe_cache_predict_mode();
// number of predicted experts per layer (LLAMA_DSV41_PREDICT_K, default 6, max 16)
int llama_moe_cache_predict_k();

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
