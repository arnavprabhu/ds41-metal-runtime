#include "llama-moe-cache.h"

#include <sys/mman.h>
#include <cerrno>
#include <cstring>

#include "llama-exps-prefetch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"
#include "llama-moe-cache-policy.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// LLAMA_DSV41_MOE_FILL_NOCACHE=1: arena fills read through an F_NOCACHE fd, so the experts held in the arena are not
// also kept in the page cache (more RAM left for the rest of the experts and Engram)
static bool moe_fill_lookup(const void * addr, size_t len, int * fd, uint64_t * off) {
    static const bool nc = [] {
        const char * v = getenv("LLAMA_DSV41_MOE_FILL_NOCACHE");
        return v != nullptr && atoi(v) != 0;
    }();
    return nc ? llama_exps_prefetch::lookup_nocache(addr, len, fd, off) : llama_exps_prefetch::lookup(addr, len, fd, off);
}

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        std::vector<ggml_backend_buffer_t> sub_bufs; // the buffers holding the banks (several if buf is a multi-buffer)
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
        std::vector<bool> filled; // slots ever fully uploaded, independent of LRU eviction
        size_t filled_bytes = 0;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_h = nullptr; // overlap mode (LLAMA_DSV41_MOE_OVERLAP): hit map, slots of the experts resident before the fill, else -1
        ggml_tensor * slot_map_m = nullptr; // overlap mode: level 1 the full map, level 2 the map of the experts the fill brings
        ggml_tensor * map_d = nullptr;      // GPU-resident mode: the resident slot map, a device weight in the arena (shared memory)
        ggml_tensor * map_f = nullptr;      // GPU-resident mode: the fill map (experts the last plan brought, else -1), device weight
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t   hits   = 0;
        size_t   misses = 0;
        size_t   bytes  = 0;
        size_t   calls  = 0; // copy() calls (cached layer x ubatch)
        size_t   tokens = 0; // sum of the ubatch sizes over the calls (/ n cached layers = tokens served)
        size_t   pread_calls    = 0; // calls whose misses were pread into the shared bank
        size_t   fallback_calls = 0; // calls whose misses went through ggml_backend_tensor_set_async
        uint64_t sync_us = 0;  // waiting for the ids (the previous splits) before planning
        uint64_t fill_us = 0;  // plan + miss upload
        uint64_t wait_us = 0;  // overlap mode: waiting for the asynchronous upload to land
    };

    // ubatches of at most max_batch tokens read their experts from the cache
    // must stay below 32: Metal picks mul_mm_id (f16 activations) over mul_mv_id (f32) when ne02 <= 1024 && ne21 >= 32
    // (ggml_metal_op_mul_mat_id_use_mm), and ne02 is the number of slots for the cache but n_expert for the host experts,
    // so a 32-token ubatch on a bank with > 1024 slots would change the arithmetic. Below 32 both use mul_mv_id, whose
    // kernel choice depends on the type and ne00 only (ne02 is only an address bound), so the cache is lossless.
    static constexpr int64_t max_batch = 16;

    int32_t n_cached_layers = 0;

    // ggml_backend_buffer_is_metal_shared of the device of each backend, resolved on first use
    using is_shared_fn = bool (*)(ggml_backend_buffer_t);
    std::unordered_map<ggml_backend_t, is_shared_fn> is_shared_of;
    std::vector<llama_exps_prefetch::upload_job> jobs;

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    // LLAMA_DSV41_MOE_TOKSTATS=1: one stderr line per decode step (small ubatch through all cached layers): hits, misses,
    // uploaded MiB, fill/sync/join ms. A step that starts > 2 s after the previous activity starts a new sequence (a prompt
    // was processed in between), so "seq S step K" is the K-th decode step after a prompt. Instrumentation only.
    struct tok_stat {
        int32_t  n_tok = 0;
        size_t   hits = 0, misses = 0, bytes = 0;
        uint64_t fill_us = 0, sync_us = 0, wait_us = 0;
    };
    const bool tok_on = [] { const char * v = getenv("LLAMA_DSV41_MOE_TOKSTATS"); return v != nullptr && atoi(v) != 0; }();
    int32_t  tok_first_il = -1;
    tok_stat tok_cur;
    int32_t  tok_seq = 0;
    int32_t  tok_idx = 0;
    std::chrono::steady_clock::time_point tok_last;

    void tok_flush() {
        if (tok_cur.n_tok == 0) {
            return;
        }
        const size_t n = tok_cur.hits + tok_cur.misses;
        fprintf(stderr, "llama_moe_cache: tokstat seq %d step %d: n_tok = %d, hits = %zu, misses = %zu, hit rate = %.2f%%, "
            "MiB = %.2f, fill ms = %.2f, sync ms = %.2f, join wait ms = %.2f\n",
            tok_seq, tok_idx, tok_cur.n_tok, tok_cur.hits, tok_cur.misses, n ? 100.0*tok_cur.hits/n : 0.0,
            tok_cur.bytes/1024.0/1024.0, tok_cur.fill_us/1000.0, tok_cur.sync_us/1000.0, tok_cur.wait_us/1000.0);
        tok_idx++;
        tok_cur = tok_stat();
    }

    void tok_add(int32_t il, int32_t n_tok, size_t hits, size_t misses, size_t bytes, uint64_t fill_us, uint64_t sync_us) {
        if (!tok_on) {
            return;
        }
        if (tok_first_il < 0) {
            for (int32_t i = 0; i < (int32_t) layers.size(); ++i) {
                if (layers[i].ig >= 0) {
                    tok_first_il = i;
                    break;
                }
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (il == tok_first_il) {
            tok_flush();
            if (tok_idx > 0 && now - tok_last > std::chrono::seconds(2)) {
                tok_seq++;
                tok_idx = 0;
            }
            tok_cur.n_tok = n_tok;
        }
        tok_cur.hits    += hits;
        tok_cur.misses  += misses;
        tok_cur.bytes   += bytes;
        tok_cur.fill_us += fill_us;
        tok_cur.sync_us += sync_us;
        tok_last = now;
    }

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer
    std::unordered_map<const ggml_tensor *, int32_t> layer_of_h; // overlap mode: hit maps -> layer
    std::unordered_map<const ggml_tensor *, int32_t> layer_of_m; // overlap mode: fill maps -> layer

    // overlap mode (LLAMA_DSV41_MOE_OVERLAP=1|2, decode of one token): the layer's lookup of the hit map starts a split whose
    // callback plans the layer and starts the upload asynchronously, so the GPU computes the work that does not need the
    // missing experts (level 1: the shared expert, level 2: also the experts already in the arena) while the upload runs;
    // the lookup of the fill map starts the next split, whose callback waits for the upload
    int  overlap_level = 0;
    // LLAMA_DSV41_MOE_OVERLAP_NT=1: also overlap ubatches of 2..max_batch tokens (speculative-decoding verify batches), else
    // only single-token decode. Lossless like the single-token mode: every (token, expert) row of the routed experts is still
    // computed exactly once, by the same mul_mv_id kernel (ne21 = n_tokens <= 16 < 32, see max_batch) on the same arena
    // bytes, in pass 1 if its slot is in the hit map, else in pass 2 (ggml_mul_mat_id_acc writes the remaining rows into the
    // buffer of pass 1); the plan keeps every expert selected by any token of the ubatch out of the victims, so no hit slot
    // is written while pass 1 runs. Default off: n_tokens == 1 only, as before.
    int64_t overlap_max_tokens = 1;
    bool overlap_small = true; // the ubatch being overlapped counts in stats_small (and the tokstat line)
    bool overlap_fill_pending = false;
    int32_t overlap_il = -1;
    size_t  overlap_pending_bytes = 0;
    std::vector<int32_t> overlap_before; // slot map of the layer before the plan

    // GPU-resident mode (LLAMA_DSV41_MOE_GPURES=1, on top of overlap level 2): the resident slot map of each layer is a
    // device weight in the arena (shared memory) that the LFU updates in place, so the lookups start no scheduler split
    // and the decode token is one Metal split. Per layer the graph signals a GPU->host event after the hit-map lookup
    // (GGML_OP_DSV41_EVENT), runs the shared expert and the hit pass, and waits for a host->GPU event before the fill-map
    // lookup. A worker thread plans the layers in order as their signals arrive (same plan calls as copy_h, so the same
    // victims), preads the misses, writes the fill map and signals; a hit-only layer costs the GPU only the event round
    // trip, hidden behind its hit pass. Safety: while layer il is planned the GPU has completed everything before the
    // signal of il (layers < il) and cannot start anything after the wait of il (layers > il), so the only arena reads in
    // flight are layer il's hits, which the plan never evicts (it excludes the experts of the call).
    using dsv41_init_fn   = bool     (*)(ggml_backend_t);
    using dsv41_base_fn   = void     (*)(ggml_backend_t, uint64_t);
    using dsv41_value_fn  = uint64_t (*)(ggml_backend_t);
    using dsv41_signal_fn = void     (*)(ggml_backend_t, uint64_t);
    bool gpures = false;
    ggml_backend_t  gr_backend     = nullptr;
    dsv41_base_fn   gr_set_base    = nullptr;
    dsv41_value_fn  gr_gpu_value   = nullptr;
    dsv41_signal_fn gr_host_signal = nullptr;
    struct gr_item {
        int32_t il;
        const ggml_tensor * ids; // selected experts, [n_ids] I32 (src[1] of the signal node)
        const ggml_tensor * hit; // the GPU's hit-map lookup, [n_ids] I32 (src[0] of the signal node)
    };
    std::vector<gr_item> gr_job;
    uint64_t gr_epoch = 0;
    uint64_t gr_base  = 0;
    std::thread gr_th;
    std::mutex gr_mu;
    std::condition_variable gr_cv;
    bool gr_run  = false; // a job is posted and not finished
    bool gr_stop = false;
    std::vector<int32_t> gr_ids;
    std::vector<int32_t> gr_hit;
    std::vector<moe_cache_lru::fill> gr_fills;
    std::vector<llama_exps_prefetch::upload_job> gr_jobs;
    struct gr_stats {
        size_t graphs = 0, layers = 0, hit_only = 0;
        uint64_t gpu_wait_us = 0;  // worker waiting for the GPU's signal (GPU busy before the layer's lookup)
        uint64_t plan_us     = 0;  // from the GPU's signal seen to the host signal, hit-only layers
        uint64_t fill_us     = 0;  // same, layers with misses (plan + pread)
        uint64_t plan_max_us = 0;
        uint64_t fill_max_us = 0;
    } grs;

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // next-layer prediction (LLAMA_DSV41_MOE_PREDICT=1): the graph computes layer il+1's router on layer il's FFN
    // input (tensors named dsv41_moe_pred-<il+1>); after layer il's own fill, the predicted misses of layer il+1 are
    // pread into free slots on a background thread while the GPU computes layer il. Residency only, same bytes.
    std::vector<ggml_tensor *> pred;  // [n_layer]
    std::thread pred_th;
    bool        pred_busy  = false;
    int32_t     pred_layer = -1;
    std::vector<int32_t> pred_ids;
    std::vector<int32_t> pred_fill_experts;
    std::vector<moe_cache_lru::fill> pred_fills;
    struct pred_stats {
        size_t launched = 0, predicted = 0, prefetched = 0, prefetched_used = 0, pred_in_actual = 0, actual = 0, bytes = 0;
        uint64_t wait_us = 0;
    } pst;

    // LLAMA_DSV41_MOE_PREDICT=2: look-ahead into the prefetch queue. At the plan of layer il (copy_h) the experts predicted
    // for layer il+1 (router of il+1 applied to il's FFN input, computed with il's own router) are read into victim slots
    // that layer il does not use, behind the demand reads (the queue yields to them). At the plan of il+1 the predicted
    // experts that were selected are kept (those still in flight are masked out of the hit map and awaited by the fill-map
    // split), the others are cancelled if unstarted, else awaited before the plan so no slot has two writers.
    struct pf_entry {
        int32_t expert;
        int32_t slot;
        std::shared_ptr<void> h;
        bool needed = false;
    };
    std::vector<pf_entry> pf_pend;      // launched for layer pf_layer
    std::vector<pf_entry> pf_join_list; // needed by the layer being planned, still in flight
    int32_t pf_layer = -1;
    std::vector<int32_t> pf_ids;
    struct pf_stats {
        size_t layers = 0, cand = 0, taken = 0, bytes = 0;
        size_t actual = 0, pred_in_actual = 0, used = 0, cancelled = 0, wasted_started = 0, late = 0, inflight_masked = 0;
        uint64_t resolve_wait_us = 0, join_wait_us = 0, launch_us = 0;
    } pfs;

    static float pf_bonus() {
        static const float v = [] { const char * e = getenv("LLAMA_DSV41_PF_BONUS"); return e ? (float) atof(e) : 1.0f; }();
        return v;
    }

    void pf_launch(ggml_backend_t backend, int32_t il) {
        const int32_t nl = il + 1;
        if (nl >= (int32_t) pred.size() || pred[nl] == nullptr || layers[nl].ig < 0) {
            return;
        }
        const layer & L = layers[nl];
        group & G = groups[L.ig];
        if (devices[G.id].backend != backend) {
            return;
        }
        const is_shared_fn is_shared = get_is_shared(backend);
        if (!is_shared || pred[nl]->data == nullptr || pred[nl]->buffer == nullptr || !is_shared(pred[nl]->buffer)) {
            return;
        }
        std::vector<int>      fds(L.experts.size());
        std::vector<uint64_t> offs(L.experts.size());
        for (size_t ip = 0; ip < L.experts.size(); ++ip) {
            const ggml_tensor * w    = L.experts[ip];
            const ggml_tensor * bank = G.banks[ip];
            if (bank->buffer == nullptr || bank->data == nullptr || !is_shared(bank->buffer) ||
                    !moe_fill_lookup(w->data, ggml_nbytes(w), &fds[ip], &offs[ip])) {
                return;
            }
        }
        const size_t n_pred = ggml_nelements(pred[nl]);
        pf_ids.resize(n_pred);
        memcpy(pf_ids.data(), pred[nl]->data, n_pred*sizeof(int32_t));
        for (int32_t e : pf_ids) {
            if (e < 0 || e >= G.lru.n_expert) {
                pf_ids.clear();
                return;
            }
        }
        // layer il's own slots stay untouched (its ids were planned just before)
        groups[layers[il].ig].lru.prefetch_begin(il, ids.data(), ids.size());
        if (layers[il].ig != L.ig) {
            G.lru.prefetch_begin(il, nullptr, 0); // different group: nothing of layer il lives there
        }
        pfs.layers++;
        pfs.cand += n_pred;
        pf_layer = nl;
        pf_pend.clear();
        static const size_t pf_n = [] { const char * e = getenv("LLAMA_DSV41_PF_N"); return e ? (size_t) atoi(e) : (size_t) 16; }();
        for (size_t r = 0; r < std::min(n_pred, pf_n); ++r) {
            moe_cache_lru::fill f;
            if (!G.lru.prefetch_take(nl, pf_ids[r], pf_bonus(), f)) {
                continue;
            }
            std::vector<llama_exps_prefetch::upload_job> pj;
            size_t slot_bytes = 0;
            for (size_t ip = 0; ip < L.experts.size(); ++ip) {
                const ggml_tensor * w    = L.experts[ip];
                ggml_tensor       * bank = G.banks[ip];
                const size_t expert_size = w->nb[2];
                slot_bytes += expert_size;
                llama_exps_prefetch::add_upload(pj, fds[ip], offs[ip] + (size_t) f.expert*expert_size,
                        (char *) bank->data + (size_t) f.slot*expert_size, expert_size, (const char *) w->data + (size_t) f.expert*expert_size);
            }
            pfs.taken++;
            pfs.bytes += slot_bytes;
            if (!G.filled[f.slot]) {
                G.filled[f.slot] = true;
                G.filled_bytes += slot_bytes;
            }
            pf_pend.push_back({ f.expert, f.slot, llama_exps_prefetch::pf_submit(std::move(pj)), false });
        }
        G.lru.prefetch_end();
        if (layers[il].ig != L.ig) {
            groups[layers[il].ig].lru.prefetch_end();
        }
    }

    // plan of layer il: score the prediction, cancel/await the wrong guesses. Returns with every non-needed item settled;
    // needed items that are still in flight stay in pf_pend (needed = true) for pf_masked()/pf_join()
    void pf_resolve(int32_t il) {
        pf_join_list.clear();
        if (pf_layer != il) {
            pf_layer = -1;
            pf_pend.clear();
            return;
        }
        group & G = groups[layers[il].ig];
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ids.size(); ++i) {
            bool dup = false;
            for (size_t j = 0; j < i && !dup; ++j) {
                dup = ids[j] == ids[i];
            }
            if (dup) {
                continue;
            }
            pfs.actual++;
            pfs.pred_in_actual += std::find(pf_ids.begin(), pf_ids.end(), ids[i]) != pf_ids.end();
        }
        std::vector<pf_entry> keep;
        for (auto & en : pf_pend) {
            if (std::find(ids.begin(), ids.end(), en.expert) != ids.end()) {
                en.needed = true;
                pfs.used++;
                if (!llama_exps_prefetch::pf_done(en.h)) {
                    pfs.inflight_masked++;
                    keep.push_back(en);
                }
                continue;
            }
            if (llama_exps_prefetch::pf_cancel(en.h)) {
                G.lru.unmap(il, en.expert, en.slot);
                pfs.cancelled++;
            } else {
                pfs.wasted_started++;
                llama_exps_prefetch::pf_wait(en.h);
            }
        }
        pf_join_list = std::move(keep);
        pf_pend.clear();
        pf_layer = -1;
        pfs.resolve_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }

    // needed prefetches that were still in flight at the plan: hidden from the hit map, awaited before the fill-map split
    void pf_mask(std::vector<int32_t> & before) {
        for (const auto & en : pf_join_list) {
            before[en.expert] = -1;
        }
    }
    void pf_join() {
        if (pf_join_list.empty()) {
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (auto & en : pf_join_list) {
            llama_exps_prefetch::pf_wait(en.h);
        }
        pf_join_list.clear();
        pfs.join_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    // everything settled before a path that does not know about the queue
    void pf_flush() {
        for (auto & en : pf_join_list) {
            llama_exps_prefetch::pf_wait(en.h);
        }
        pf_join_list.clear();
        for (auto & en : pf_pend) {
            if (!en.needed && llama_exps_prefetch::pf_cancel(en.h)) {
                groups[layers[pf_layer].ig].lru.unmap(pf_layer, en.expert, en.slot);
            } else {
                llama_exps_prefetch::pf_wait(en.h);
            }
        }
        pf_pend.clear();
        pf_layer = -1;
    }

    // LLAMA_DSV41_MOE_WARM=1: fill the empty arena slots with the hottest not-yet-resident experts (heat: profile seed or
    // heat file) through the prefetch queue, which yields to the demand reads, so a freshly started server does not decode
    // its first hundred tokens with a cold arena. A slot is reserved while its read is in flight and enters the slot map
    // only when the bytes are in place; never evicts anything.
    struct warm_cand { float heat; int32_t layer, expert; };
    struct warm_inflight { int32_t layer, expert, slot; std::shared_ptr<void> h; };
    struct warm_group {
        std::vector<warm_cand> cands;
        size_t pos = 0;
        int32_t cursor = 0;
        bool built = false;
        bool done = false;
        std::vector<warm_inflight> inflight;
    };
    std::vector<warm_group> warm;
    struct warm_stats { size_t started = 0, landed = 0, dup = 0, bytes = 0; } wst;

    static bool warm_enabled() {
        static const bool on = [] { const char * v = getenv("LLAMA_DSV41_MOE_WARM"); return v && atoi(v) != 0; }();
        return on;
    }

    void warm_step(ggml_backend_t backend, int32_t il) {
        group & G = groups[layers[il].ig];
        if (devices[G.id].backend != backend) {
            return;
        }
        const int ig = layers[il].ig;
        if (warm.size() != groups.size()) {
            warm.resize(groups.size());
        }
        warm_group & W = warm[ig];
        if (W.done) {
            return;
        }
        const is_shared_fn is_shared = get_is_shared(backend);
        if (!is_shared) {
            W.done = true;
            return;
        }
        // land the reads that finished
        for (size_t i = 0; i < W.inflight.size();) {
            auto & f = W.inflight[i];
            if (!llama_exps_prefetch::pf_done(f.h)) {
                i++;
                continue;
            }
            int32_t * sm = G.lru.slot_map[f.layer];
            if (sm[f.expert] >= 0) {
                G.lru.key_of[f.slot] = -1; // the expert came in through a demand fill meanwhile: give the slot back
                wst.dup++;
            } else {
                sm[f.expert] = f.slot;
                wst.landed++;
            }
            G.lru.reserve(f.slot, false);
            W.inflight[i] = std::move(W.inflight.back());
            W.inflight.pop_back();
        }
        if (!W.built) {
            W.built = true;
            for (int32_t l : G.layers) {
                for (int32_t e = 0; e < G.lru.n_expert; ++e) {
                    const float h = G.lru.heat[(size_t) l*G.lru.n_expert + e];
                    if (h > 0.0f) {
                        W.cands.push_back({ h, l, e });
                    }
                }
            }
            std::sort(W.cands.begin(), W.cands.end(), [](const warm_cand & a, const warm_cand & b) { return a.heat > b.heat; });
        }
        static const size_t max_inflight = [] { const char * v = getenv("LLAMA_DSV41_WARM_INFLIGHT"); return v ? (size_t) atoi(v) : (size_t) 12; }();
        int budget = 2;
        while (budget > 0 && W.inflight.size() < max_inflight) {
            while (W.cursor < G.n_slots && G.lru.key_of[W.cursor] >= 0) {
                W.cursor++;
            }
            if (W.cursor >= G.n_slots) {
                if (W.inflight.empty()) {
                    W.done = true;
                }
                return;
            }
            // next candidate that is not resident
            while (W.pos < W.cands.size() && G.lru.slot_map[W.cands[W.pos].layer][W.cands[W.pos].expert] >= 0) {
                W.pos++;
            }
            if (W.pos >= W.cands.size()) {
                if (W.inflight.empty()) {
                    W.done = true;
                }
                return;
            }
            const warm_cand c = W.cands[W.pos++];
            const layer & L = layers[c.layer];
            std::vector<llama_exps_prefetch::upload_job> pj;
            size_t slot_bytes = 0;
            bool ok = true;
            for (size_t ip = 0; ip < L.experts.size() && ok; ++ip) {
                const ggml_tensor * w    = L.experts[ip];
                ggml_tensor       * bank = G.banks[ip];
                int fd = -1; uint64_t off = 0;
                if (bank->buffer == nullptr || bank->data == nullptr || !is_shared(bank->buffer) ||
                        !moe_fill_lookup(w->data, ggml_nbytes(w), &fd, &off)) {
                    ok = false;
                    break;
                }
                const size_t es = w->nb[2];
                slot_bytes += es;
                llama_exps_prefetch::add_upload(pj, fd, off + (size_t) c.expert*es, (char *) bank->data + (size_t) W.cursor*es, es, (const char *) w->data + (size_t) c.expert*es);
            }
            if (!ok) {
                W.done = true;
                return;
            }
            const int32_t slot = W.cursor++;
            G.lru.key_of[slot] = c.layer*G.lru.n_expert + c.expert;
            G.lru.reserve(slot, true);
            if (!G.filled[slot]) {
                G.filled[slot] = true;
                G.filled_bytes += slot_bytes;
            }
            wst.started++;
            wst.bytes += slot_bytes;
            W.inflight.push_back({ c.layer, c.expert, slot, llama_exps_prefetch::pf_submit(std::move(pj)) });
            budget--;
        }
    }

    void warm_finish() {
        for (size_t ig = 0; ig < warm.size(); ++ig) {
            for (auto & f : warm[ig].inflight) {
                llama_exps_prefetch::pf_wait(f.h);
                groups[ig].lru.slot_map[f.layer][f.expert] = groups[ig].lru.slot_map[f.layer][f.expert] >= 0 ? groups[ig].lru.slot_map[f.layer][f.expert] : f.slot;
                groups[ig].lru.reserve(f.slot, false);
            }
            warm[ig].inflight.clear();
        }
    }

    void save_heat() {
        const char * path = getenv("LLAMA_DSV41_MOE_HEAT_SAVE");
        if (path == nullptr || path[0] == '\0') {
            return;
        }
        int32_t n_expert = 0;
        for (const group & g : groups) {
            if (g.lru.n_expert > 0) {
                n_expert = g.lru.n_expert;
            }
        }
        if (n_expert == 0) {
            return;
        }
        std::vector<float> h(layers.size()*(size_t) n_expert, 0.0f);
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].ig >= 0) {
                const auto & src = groups[layers[il].ig].lru.heat;
                std::copy(src.begin() + il*(size_t) n_expert, src.begin() + (il + 1)*(size_t) n_expert, h.begin() + il*(size_t) n_expert);
            }
        }
        const bool ok = moe_cache_lru::save_heat_file(path, (int32_t) layers.size(), n_expert, h);
        fprintf(stderr, "llama_moe_cache: heat %s to %s\n", ok ? "saved" : "NOT saved", path);
    }

    void pred_wait() {
        if (pred_busy) {
            const auto t0 = std::chrono::steady_clock::now();
            pred_th.join();
            pst.wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            pred_busy = false;
        }
    }

    void set_graph(ggml_cgraph * gf) {
        if (gpures) {
            gr_begin(gf);
        }
        if (!llama_moe_cache_predict_enabled()) {
            return;
        }
        pred.assign(layers.size(), nullptr);
        static const char prefix[] = "dsv41_moe_pred-";
        const int n = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n; ++i) {
            ggml_tensor * t = ggml_graph_node(gf, i);
            if (strncmp(t->name, prefix, sizeof(prefix) - 1) == 0) {
                const int il = atoi(t->name + sizeof(prefix) - 1);
                if (il >= 0 && il < (int) pred.size()) {
                    pred[il] = t;
                }
            }
        }
    }

    // GPU-resident mode ---------------------------------------------------------------------------------------------------

    // the previous graph's layers are all planned and signaled (its GPU work may still run, it reads no slot we change)
    void gr_wait_idle() {
        std::unique_lock<std::mutex> lk(gr_mu);
        gr_cv.wait(lk, [&] { return !gr_run; });
    }

    // the graph about to be computed: hand its signal nodes (in layer order) to the worker, with a new event base
    void gr_begin(ggml_cgraph * gf) {
        gr_wait_idle();
        gr_job.clear();
        const int n = ggml_graph_n_nodes(gf);
        for (int i = 0; i < n; ++i) {
            const ggml_tensor * t = ggml_graph_node(gf, i);
            const int32_t * params = (const int32_t *) t->op_params; // {mode, id}
            if (t->op != GGML_OP_DSV41_EVENT || params[0] != GGML_DSV41_EVENT_SIGNAL) {
                continue;
            }
            const int32_t il = params[1];
            GGML_ASSERT(il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 && layers[il].map_f != nullptr);
            GGML_ASSERT(t->src[0] != nullptr && t->src[1] != nullptr && ggml_nelements(t->src[0]) == ggml_nelements(t->src[1]));
            gr_job.push_back({ il, t->src[1], t->src[0] });
        }
        if (gr_job.empty()) {
            return;
        }
        // the GPU runs the layers in order (each layer's lookup depends on the previous layer's output)
        std::sort(gr_job.begin(), gr_job.end(), [](const gr_item & a, const gr_item & b) { return a.il < b.il; });
        // event values only grow: base + il + 1 with il < 255
        gr_epoch++;
        gr_base = gr_epoch << 8;
        gr_set_base(gr_backend, gr_base);
        grs.graphs++;
        {
            std::lock_guard<std::mutex> lk(gr_mu);
            gr_run = true;
        }
        gr_cv.notify_all();
    }

    void gr_loop() {
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(gr_mu);
                gr_cv.wait(lk, [&] { return gr_run || gr_stop; });
                if (!gr_run) {
                    return;
                }
            }
            for (const gr_item & it : gr_job) {
                gr_layer(it);
            }
            {
                std::lock_guard<std::mutex> lk(gr_mu);
                gr_run = false;
            }
            gr_cv.notify_all();
        }
    }

    void gr_shutdown() {
        if (!gr_th.joinable()) {
            return;
        }
        gr_wait_idle();
        {
            std::lock_guard<std::mutex> lk(gr_mu);
            gr_stop = true;
        }
        gr_cv.notify_all();
        gr_th.join();
    }

    // one layer on the worker: wait for the GPU's signal, plan exactly like copy_h, pread the misses, publish the fill map
    // (built from the GPU's own hit lookup, so every selected row is computed exactly once), signal the GPU
    void gr_layer(const gr_item & it) {
        const int32_t  il = it.il;
        const uint64_t v  = gr_base + (uint64_t) il + 1;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t spin = 0; gr_gpu_value(gr_backend) < v; ++spin) {
            if (spin >= 2048) {
                std::this_thread::yield();
                if ((spin & 0xffff) == 0 && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(30)) {
                    GGML_ABORT("LLAMA_DSV41_MOE_GPURES: no GPU signal for layer %d after 30 s (graph aborted?)", il);
                }
            }
        }
        const auto t1 = std::chrono::steady_clock::now();

        const size_t n_ids = ggml_nelements(it.ids);
        GGML_ASSERT(it.ids->data != nullptr && it.hit->data != nullptr && ggml_is_contiguous(it.ids) && ggml_is_contiguous(it.hit));
        gr_ids.resize(n_ids);
        gr_hit.resize(n_ids);
        memcpy(gr_ids.data(), it.ids->data, n_ids*sizeof(int32_t));
        memcpy(gr_hit.data(), it.hit->data, n_ids*sizeof(int32_t));

        size_t n_hit = 0;
        int32_t n_tok = 1;
        if (n_expert_used > 0 && n_ids % (size_t) n_expert_used == 0) {
            n_tok = (int32_t) (n_ids / (size_t) n_expert_used);
        }
        if (!g.lru.plan(il, gr_ids.data(), n_ids, gr_fills, n_hit, n_tok)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        size_t bytes = 0;
        if (!gr_fills.empty()) {
            overlap_start_fill(l, g, gr_fills, gr_jobs, bytes);
            if (!gr_jobs.empty()) {
                llama_exps_prefetch::upload_end();
            }
        }
        size_t slot_bytes = 0;
        for (const ggml_tensor * w : l.experts) {
            slot_bytes += w->nb[2];
        }
        for (const auto & fill : gr_fills) {
            GGML_ASSERT(fill.slot >= 0 && fill.slot < g.n_slots);
            if (!g.filled[fill.slot]) {
                g.filled[fill.slot] = true;
                g.filled_bytes += slot_bytes;
            }
        }

        // the fill map: the slot of every selected expert the GPU's lookup did not find, -1 elsewhere
        const int32_t n_expert = g.lru.n_expert;
        const int32_t * now = g.lru.slot_map[il];
        int32_t * out = (int32_t *) l.map_f->data;
        std::fill(out, out + n_expert, -1);
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t e = gr_ids[i];
            if (gr_hit[i] < 0) {
                out[e] = now[e];
            } else {
                GGML_ASSERT(now[e] == gr_hit[i] && "LLAMA_DSV41_MOE_GPURES: a slot read by the hit pass was evicted");
            }
        }
        // the slot map, the fill map and the expert bytes are in shared memory: visible to the work after the GPU's wait
        gr_host_signal(gr_backend, v);
        const auto t2 = std::chrono::steady_clock::now();

        const uint64_t wait_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        const uint64_t work_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        grs.layers++;
        grs.gpu_wait_us += wait_us;
        if (gr_fills.empty()) {
            grs.hit_only++;
            grs.plan_us += work_us;
            grs.plan_max_us = std::max(grs.plan_max_us, work_us);
        } else {
            grs.fill_us += work_us;
            grs.fill_max_us = std::max(grs.fill_max_us, work_us);
        }

        stats & st = n_ids <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += gr_fills.size();
        st.bytes  += bytes;
        st.calls  += 1;
        st.tokens += n_tok;
        st.pread_calls += !gr_fills.empty();
        st.sync_us += wait_us;
        st.fill_us += work_us;
        if (&st == &stats_small) {
            tok_add(il, n_tok, n_hit, gr_fills.size(), bytes, work_us, wait_us);
        }
    }

    // called by copy(il) after layer il's own fill: prefetch layer il+1's predicted misses in the background
    void pred_launch(ggml_backend_t backend, int32_t il) {
        const int32_t nl = il + 1;
        if (nl >= (int32_t) pred.size() || pred[nl] == nullptr || layers[nl].ig < 0) {
            return;
        }
        const layer & L = layers[nl];
        group & G = groups[L.ig];
        if (devices[G.id].backend != backend) {
            return;
        }
        const is_shared_fn is_shared = llama_exps_prefetch::upload_enabled() ? get_is_shared(backend) : nullptr;
        if (!is_shared) {
            return;
        }
        std::vector<int>      fds(L.experts.size());
        std::vector<uint64_t> offs(L.experts.size());
        for (size_t ip = 0; ip < L.experts.size(); ++ip) {
            const ggml_tensor * w    = L.experts[ip];
            const ggml_tensor * bank = G.banks[ip];
            if (bank->buffer == nullptr || bank->data == nullptr || !is_shared(bank->buffer) ||
                    !moe_fill_lookup(w->data, ggml_nbytes(w), &fds[ip], &offs[ip])) {
                return;
            }
        }
        pred_ids.resize(ggml_nelements(pred[nl]));
        ggml_backend_tensor_get(pred[nl], pred_ids.data(), 0, ggml_nbytes(pred[nl]));
        for (int32_t e : pred_ids) {
            if (e < 0 || e >= G.lru.n_expert) {
                return;
            }
        }
        size_t n_hit = 0;
        // the slots layer il reads (hits on the GPU now, fills in the next split) are not victims
        groups[layers[il].ig].lru.prefetch_begin(il, ids.data(), ids.size());
        const bool planned = G.lru.plan(nl, pred_ids.data(), pred_ids.size(), pred_fills, n_hit);
        groups[layers[il].ig].lru.prefetch_end();
        if (!planned) {
            return;
        }
        // the experts layer il is about to read must still be resident (victims come from the LRU head)
        {
            const group & Gc = groups[layers[il].ig];
            for (int32_t e : ids) {
                GGML_ASSERT(Gc.lru.slot_map[il][e] >= 0 && "MoE prediction evicted an expert of the running layer");
            }
        }
        pst.launched++;
        pst.predicted  += pred_ids.size();
        pst.prefetched += pred_fills.size();
        pred_layer = nl;
        pred_fill_experts.clear();
        if (pred_fills.empty()) {
            return;
        }
        std::vector<llama_exps_prefetch::upload_job> pj;
        size_t slot_bytes = 0;
        for (size_t ip = 0; ip < L.experts.size(); ++ip) {
            const ggml_tensor * w    = L.experts[ip];
            ggml_tensor       * bank = G.banks[ip];
            const size_t expert_size = w->nb[2];
            slot_bytes += expert_size;
            for (size_t i = 0; i < pred_fills.size();) {
                size_t n = 1;
                while (i + n < pred_fills.size() && pred_fills[i + n].expert == pred_fills[i].expert + (int32_t) n && pred_fills[i + n].slot == pred_fills[i].slot + (int32_t) n) {
                    n++;
                }
                GGML_ASSERT(pred_fills[i].slot + (int64_t) n <= bank->ne[2]);
                const size_t src_off = (size_t) pred_fills[i].expert*expert_size;
                const size_t dst_off = (size_t) pred_fills[i].slot*expert_size;
                llama_exps_prefetch::add_upload(pj, fds[ip], offs[ip] + src_off, (char *) bank->data + dst_off, n*expert_size, (const char *) w->data + src_off);
                pst.bytes += n*expert_size;
                i += n;
            }
        }
        for (const auto & f : pred_fills) {
            pred_fill_experts.push_back(f.expert);
            if (!G.filled[f.slot]) {
                G.filled[f.slot] = true;
                G.filled_bytes += slot_bytes;
            }
        }
        pred_busy = true;
        pred_th = std::thread([jobs = std::move(pj)]() {
            llama_exps_prefetch::upload_bg(jobs);
        });
    }

    // called by copy(il) once layer il's real ids are known: wait for its prefetch and score the prediction
    void pred_resolve(int32_t il) {
        pred_wait();
        if (pred_layer != il) {
            return;
        }
        pred_layer = -1;
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t e = ids[i];
            bool dup = false;
            for (size_t j = 0; j < i && !dup; ++j) {
                dup = ids[j] == e;
            }
            if (dup) {
                continue;
            }
            pst.actual++;
            pst.pred_in_actual  += std::find(pred_ids.begin(), pred_ids.end(), e) != pred_ids.end();
            pst.prefetched_used += std::find(pred_fill_experts.begin(), pred_fill_experts.end(), e) != pred_fill_experts.end();
        }
    }

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        {
            const char * v = getenv("LLAMA_DSV41_MOE_OVERLAP");
            overlap_level = v ? atoi(v) : 0;
            if (overlap_level != 0 && !llama_exps_prefetch::upload_enabled()) {
                fprintf(stderr, "llama_moe_cache: LLAMA_DSV41_MOE_OVERLAP needs the pread uploader, disabled\n");
                overlap_level = 0;
            }
            const char * nt = getenv("LLAMA_DSV41_MOE_OVERLAP_NT");
            overlap_max_tokens = nt != nullptr && atoi(nt) != 0 ? max_batch : 1;

            // LLAMA_DSV41_MOE_GPURES=1: GPU-resident slot maps and event-chained misses (see gr_layer)
            const char * gr = getenv("LLAMA_DSV41_MOE_GPURES");
            gpures = gr != nullptr && atoi(gr) != 0;
            if (gpures) {
                const char * why = nullptr;
                if (overlap_level < 2) {
                    why = "needs LLAMA_DSV41_MOE_OVERLAP=2";
                } else if (llama_moe_cache_predict_mode() != 0) {
                    why = "does not support LLAMA_DSV41_MOE_PREDICT"; // a second agent changing residency would change the victims
                } else if (warm_enabled()) {
                    why = "does not support LLAMA_DSV41_MOE_WARM";
                } else if (model.hparams.no_alloc) {
                    why = "no_alloc";
                } else {
                    for (const group & g : groups) {
                        if (gr_backend == nullptr) {
                            gr_backend = devices[g.id].backend;
                        } else if (devices[g.id].backend != gr_backend) {
                            why = "needs all cached layers on one device";
                        }
                    }
                    ggml_backend_dev_t dev = gr_backend ? ggml_backend_get_device(gr_backend) : nullptr;
                    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
                    auto init       = reg ? (dsv41_init_fn)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_dsv41_init")        : nullptr;
                    gr_set_base     = reg ? (dsv41_base_fn)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_dsv41_set_base")    : nullptr;
                    gr_gpu_value    = reg ? (dsv41_value_fn)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_dsv41_gpu_value")   : nullptr;
                    gr_host_signal  = reg ? (dsv41_signal_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_dsv41_host_signal") : nullptr;
                    if (why == nullptr && (!init || !gr_set_base || !gr_gpu_value || !gr_host_signal || !init(gr_backend))) {
                        why = "needs the Metal backend's shared events";
                    }
                }
                if (why != nullptr) {
                    fprintf(stderr, "llama_moe_cache: LLAMA_DSV41_MOE_GPURES %s, disabled\n", why);
                    gpures = false;
                }
            }
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init(model.layers.size(), n_expert, g.n_slots);
            g.filled.assign(g.n_slots, false);
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size()) + (gpures ? 2*g.layers.size() : 0);
            n_tensors_host  += g.layers.size()*(overlap_level != 0 ? 3 : 1);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                n_cached_layers++;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (overlap_level != 0) {
                    l.slot_map_h = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_h, "moe_cache.slot_map_h-%d", il);
                    l.slot_map_m = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_m, "moe_cache.slot_map_m-%d", il);
                    // the device copies keep a private region (ggml-backend): the fill map is written while the previous split still runs
                    ggml_set_output(l.slot_map_h);
                    ggml_set_output(l.slot_map_m);
                    layer_of_h[l.slot_map_h] = il;
                    layer_of_m[l.slot_map_m] = il;
                    buf_host_size += 2*GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                }
                if (gpures) {
                    // device weights in the arena: the lookups read them in place, no scheduler split, no copy callback
                    l.map_d = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.map_d, "moe_cache.map_d-%d", il);
                    l.map_f = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.map_f, "moe_cache.map_f-%d", il);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                // an arena larger than the device's maximum buffer size (Metal maxBufferLength, 38.9 GiB on M5 Pro) is
                // split over several buffers behind a multi-buffer wrapper: the banks live in the sub-buffers, so those
                // are tested (the wrapper itself is never a shared Metal buffer)
                const is_shared_fn is_shared = get_is_shared(d.backend);
                d.sub_bufs.clear();
                // LLAMA_DSV41_MOE_SUBBUF_CHECK=0: old check of the wrapper only (a multi-buffer arena is cleared and runs without overlap)
                static const bool subbuf_check = [] { const char * v = getenv("LLAMA_DSV41_MOE_SUBBUF_CHECK"); return v == nullptr || atoi(v) != 0; }();
                for (ggml_tensor * t = subbuf_check ? ggml_get_first_tensor(d.ctx.get()) : nullptr; t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    if (t->view_src == nullptr && t->buffer != nullptr &&
                            std::find(d.sub_bufs.begin(), d.sub_bufs.end(), t->buffer) == d.sub_bufs.end()) {
                        d.sub_bufs.push_back(t->buffer);
                    }
                }
                if (d.sub_bufs.empty()) {
                    d.sub_bufs.push_back(d.buf.get());
                }
                bool all_shared = true;
                for (ggml_backend_buffer_t b : d.sub_bufs) {
                    // Shared Metal slots are written completely before their GPU map is published.
                    if (is_shared == nullptr || !is_shared(b)) {
                        ggml_backend_buffer_clear(b, 0);
                        all_shared = false;
                    }
                }
                fprintf(stderr, "llama_moe_cache: arena of %s in %zu buffer(s), %s\n", ggml_backend_name(d.backend), d.sub_bufs.size(),
                    all_shared ? "all shared" : "not all shared");
                if (!all_shared && overlap_level != 0) {
                    fprintf(stderr, "llama_moe_cache: the arena of %s is not a shared buffer, LLAMA_DSV41_MOE_OVERLAP disabled\n", ggml_backend_name(d.backend));
                    overlap_level = 0;
                }
                if (!all_shared && gpures) {
                    fprintf(stderr, "llama_moe_cache: the arena of %s is not a shared buffer, LLAMA_DSV41_MOE_GPURES disabled\n", ggml_backend_name(d.backend));
                    gpures = false;
                }
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            if (overlap_level != 0) {
                fprintf(stderr, "llama_moe_cache: overlap level %d (async arena fill), ubatches of up to %lld tokens\n", overlap_level, (long long) overlap_max_tokens);
            }
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            if (gpures) {
                for (const layer & l : layers) {
                    if (l.map_d != nullptr) {
                        memset(l.map_d->data, 0xff, ggml_nbytes(l.map_d)); // all slots are -1
                        memset(l.map_f->data, 0xff, ggml_nbytes(l.map_f));
                    }
                }
                fprintf(stderr, "llama_moe_cache: GPU-resident slot maps, event-chained misses (LLAMA_DSV41_MOE_GPURES)\n");
            }

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = resident_map(il);
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
        }
        // also on stderr: llama-bench silences the llama log
        for (const device & d : devices) {
            if (d.host_bytes > 0) {
                fprintf(stderr, "llama_moe_cache: arena %s %.2f MiB for %.2f MiB of host experts, %d cached layers, max_batch %d\n",
                    ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, n_cached_layers, (int) max_batch);
            }
        }
        log_arena("startup");
        if (const char * ml = getenv("LLAMA_DSV41_MOE_MLOCK"); ml && atoi(ml) != 0) {
            // keep the arena resident (pageable otherwise between command buffers; Strata MAC.md). Pages are wired as they are touched.
            for (const device & d : devices) {
                for (ggml_backend_buffer_t b : d.sub_bufs) {
                    void * base = ggml_backend_buffer_get_base(b);
                    const size_t sz = ggml_backend_buffer_get_size(b);
                    const int rc = mlock(base, sz);
                    fprintf(stderr, "llama_moe_cache: mlock arena %.0f MiB: %s\n", sz/1024.0/1024.0, rc == 0 ? "ok" : strerror(errno));
                }
            }
        }
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            fprintf(stderr, "llama_moe_cache: group %zu: %2zu layers, %s, %5d slots (%.1f%%, %.1f slots/layer)\n", ig,
                g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                (double) g.n_slots/g.layers.size());
            std::string ls;
            for (int32_t l : g.layers) {
                ls += " " + std::to_string(l);
            }
            fprintf(stderr, "llama_moe_cache: group %zu layers:%s\n", ig, ls.c_str());
        }
        if (gpures) {
            gr_th = std::thread([this] { gr_loop(); });
        }
    }

    // the slot map the LFU updates: the host map, or in GPU-resident mode the device map (shared memory). The old copy
    // callback paths publish this array, so both modes keep one residency state
    int32_t * resident_map(int32_t il) const {
        const layer & l = layers[il];
        return (int32_t *) (gpures && l.map_d != nullptr ? l.map_d->data : l.slot_map->data);
    }

    // bytes of the slots in use (the buffer itself may be larger after shrink())
    size_t active_bytes() const {
        size_t res = 0;
        for (const group & g : groups) {
            size_t slot = 0;
            for (const ggml_tensor * t : g.ref) {
                slot += t->nb[2];
            }
            res += slot*(size_t) std::max(g.n_slots, 0);
        }
        return res;
    }

    // LLAMA_DSV41_MOE_CACHE_MIB=auto: the arena is allocated at the maximum size (untouched pages cost no memory) and, once the
    // KV cache and compute buffers are known, only the first slots are used: every group keeps the same share of its size.
    // Must be called before the first graph is computed (no slot is in use).
    void shrink(size_t target) {
        const size_t cur = active_bytes();
        if (cur == 0 || target >= cur) {
            return;
        }
        const double f = (double) target/(double) cur;
        for (group & g : groups) {
            if (g.n_slots == 0) {
                continue;
            }
            const int32_t n = std::max<int32_t>(2*n_expert_used, (int32_t) (g.n_slots*f));
            if (n >= g.n_slots) {
                continue;
            }
            g.n_slots = n;
            g.lru.init(layers.size(), g.lru.n_expert, n);
            g.filled.assign(n, false);
            for (int32_t il : g.layers) {
                if (layers[il].slot_map != nullptr && layers[il].slot_map->data != nullptr) {
                    g.lru.slot_map[il] = resident_map(il);
                    memset(g.lru.slot_map[il], 0xff, ggml_nbytes(layers[il].slot_map));
                    if (gpures) {
                        memset(layers[il].map_f->data, 0xff, ggml_nbytes(layers[il].map_f));
                    }
                }
            }
        }
        fprintf(stderr, "llama_moe_cache: arena limited to %.2f MiB of %.2f MiB allocated\n", active_bytes()/1024.0/1024.0, cur/1024.0/1024.0);
    }

    static bool direct_enabled() {
        static const bool on = [] {
            const char * v = getenv("LLAMA_DSV41_MOE_DIRECT");
            return v != nullptr && atoi(v) != 0;
        }();
        return on;
    }

    is_shared_fn get_is_shared(ggml_backend_t backend) {
        auto it = is_shared_of.find(backend);
        if (it == is_shared_of.end()) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            auto fn = reg ? (is_shared_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_is_metal_shared") : nullptr;
            it = is_shared_of.emplace(backend, fn).first;
        }
        return it->second;
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    void log_arena(const char * phase) const {
        size_t allocated = 0;
        size_t filled = 0;
        for (const device & d : devices) {
            allocated += d.buf_size;
        }
        for (const group & g : groups) {
            filled += g.filled_bytes;
        }
        fprintf(stderr, "llama_moe_cache: arena %s: allocated = %zu bytes, ever filled = %zu bytes (%.2f / %.2f MiB)\n",
            phase, allocated, filled, filled/1024.0/1024.0, allocated/1024.0/1024.0);
    }

    void overlap_drain() {
        if (overlap_fill_pending) {
            llama_exps_prefetch::upload_end();
            overlap_fill_pending = false;
        }
    }

    ~impl() {
        gr_shutdown();
        tok_flush();
        overlap_drain();
        pf_flush();
        warm_finish();
        save_heat();
        if (wst.started > 0) {
            fprintf(stderr, "llama_moe_cache: warm fill: started = %zu (%.2f MiB), landed = %zu, duplicates = %zu\n", wst.started, wst.bytes/1024.0/1024.0, wst.landed, wst.dup);
        }
        if (pfs.layers > 0) {
            fprintf(stderr, "llama_moe_cache: prefetch: layers = %zu, candidates = %zu, taken = %zu (%.2f MiB), recall (predicted in actual) = %.2f%%, "
                "used = %zu (%.2f%% of taken, %.2f%% of actual), cancelled = %zu, wasted started = %zu, in flight at plan = %zu, "
                "resolve wait = %.2f ms, join wait = %.2f ms, launch cpu = %.2f ms\n",
                pfs.layers, pfs.cand, pfs.taken, pfs.bytes/1024.0/1024.0, pfs.actual ? 100.0*pfs.pred_in_actual/pfs.actual : 0.0,
                pfs.used, pfs.taken ? 100.0*pfs.used/pfs.taken : 0.0, pfs.actual ? 100.0*pfs.used/pfs.actual : 0.0,
                pfs.cancelled, pfs.wasted_started, pfs.inflight_masked, pfs.resolve_wait_us/1000.0, pfs.join_wait_us/1000.0, pfs.launch_us/1000.0);
        }
        pred_wait();
        if (seed_st.layers > 0) {
            fprintf(stderr, "llama_moe_cache: prefill seeding: layers = %zu, experts = %zu (%.2f MiB), %.1f ms on the staging thread\n",
                seed_st.layers, seed_st.experts, seed_st.bytes/1024.0/1024.0, seed_st.us/1000.0);
        }
        log_arena("teardown");
        log_stats();
        if (grs.layers > 0) {
            const size_t n_fill = grs.layers - grs.hit_only;
            fprintf(stderr, "llama_moe_cache: gpures: graphs = %zu, layers = %zu, hit-only = %zu (%.1f%%), worker wait for GPU ms/graph = %.2f, "
                "host turnaround us/layer: hit-only %.1f (max %.0f), with misses %.1f (max %.0f)\n",
                grs.graphs, grs.layers, grs.hit_only, 100.0*grs.hit_only/grs.layers, grs.gpu_wait_us/1000.0/std::max<size_t>(1, grs.graphs),
                (double) grs.plan_us/std::max<size_t>(1, grs.hit_only), (double) grs.plan_max_us,
                (double) grs.fill_us/std::max<size_t>(1, n_fill), (double) grs.fill_max_us);
        }
        if (pst.launched > 0) {
            fprintf(stderr, "llama_moe_cache: predict: layers = %zu, predicted = %zu, prefetched = %zu (%.2f MiB), "
                "prediction recall = %.2f%%, prefetched used = %.2f%%, prefetch wait = %.2f ms total\n",
                pst.launched, pst.predicted, pst.prefetched, pst.bytes/1024.0/1024.0,
                pst.actual ? 100.0*pst.pred_in_actual/pst.actual : 0.0,
                pst.prefetched ? 100.0*pst.prefetched_used/pst.prefetched : 0.0, pst.wait_us/1000.0);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    // overlap mode: the hit and fill maps of layer il, false if the layer reads the plain slot map
    bool get_slot_maps(int32_t il, int64_t n_tokens, int64_t n_expert_used, ggml_tensor ** h, ggml_tensor ** m) const {
        if (!overlap_usable(il, n_tokens) || get_slot_map(il, n_tokens, n_expert_used) == nullptr) {
            return false;
        }
        *h = layers[il].slot_map_h;
        *m = layers[il].slot_map_m;
        return true;
    }

    // GPU-resident mode: the device slot map and fill map of layer il, false if the layer does not use the mode
    bool get_gpures_maps(int32_t il, int64_t n_tokens, int64_t n_expert_used, ggml_tensor ** map, ggml_tensor ** fill) const {
        if (!gpures || !overlap_usable(il, n_tokens) || get_slot_map(il, n_tokens, n_expert_used) == nullptr || layers[il].map_d == nullptr) {
            return false;
        }
        *map  = layers[il].map_d;
        *fill = layers[il].map_f;
        return true;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    // LLAMA_DSV41_MOE_SEED_STAGE (prefill arena seeding): w is a host expert tensor of a cached layer, counts [n_expert] the
    // selections of that layer over the last n_rows prompt rows. Gives slots to up to max_n of the most selected experts
    // (moe_cache_lru::seed_take) and copies their bytes from src(host tensor, e), a CPU address holding expert e (the prefill
    // staging set, same bytes as the file). Runs on the staging thread while the GPU computes the prefill: no decode graph
    // and no arena read by the GPU may run concurrently (large ubatches do not read the arena; the caller joins before a
    // small ubatch). Lossless: the arena only holds file bytes, outputs do not depend on residency. Returns bytes copied.
    size_t seed_layer(const ggml_tensor * w, const float * counts, int32_t n_rows, int32_t max_n,
            const std::function<const void *(const ggml_tensor *, int64_t)> & src) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const int32_t il = it->second.il;
        const layer & l = layers[il];
        group & g = groups[l.ig];
        if (g.lru.slot_map[il] == nullptr) {
            return 0;
        }
        const is_shared_fn is_shared = get_is_shared(devices[g.id].backend);
        for (const ggml_tensor * bank : g.banks) {
            if (is_shared == nullptr || bank->data == nullptr || bank->buffer == nullptr || !is_shared(bank->buffer)) {
                return 0;
            }
        }
        // nothing else may write a slot that the plan below hands out
        overlap_drain();
        pf_flush();
        pred_wait();
        const auto t0 = std::chrono::steady_clock::now();
        g.lru.seed_take(il, counts, n_rows, max_n, fills);
        size_t bytes = 0;
        size_t slot_bytes = 0;
        for (const ggml_tensor * t : l.experts) {
            slot_bytes += t->nb[2];
        }
        for (const auto & f : fills) {
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * t = l.experts[ip];
                const size_t es = t->nb[2];
                const void * s = src(t, f.expert);
                GGML_ASSERT(s != nullptr);
                memcpy((char *) g.banks[ip]->data + (size_t) f.slot*es, s, es);
                bytes += es;
            }
            if (!g.filled[f.slot]) {
                g.filled[f.slot] = true;
                g.filled_bytes += slot_bytes;
            }
        }
        seed_st.layers++;
        seed_st.experts += fills.size();
        seed_st.bytes   += bytes;
        seed_st.us      += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
        return bytes;
    }
    struct { size_t layers = 0, experts = 0, bytes = 0; uint64_t us = 0; } seed_st;

    // LLAMA_DSV41_PREFILL_ARENA=1: address of expert e of the host tensor w inside the arena (shared memory), nullptr if it is
    // not resident or the arena is not CPU-addressable. Read-only: no heat, no recency change, bytes identical to the file.
    const void * resident_expert(const ggml_tensor * w, int64_t e, int32_t * slot_out = nullptr) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return nullptr;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return nullptr;
        }
        const ggml_tensor * bank = g.banks[b.ip];
        const is_shared_fn is_shared = get_is_shared(devices[g.id].backend);
        if (is_shared == nullptr || bank->data == nullptr || bank->buffer == nullptr || !is_shared(bank->buffer)) {
            return nullptr;
        }
        if (slot_out) {
            *slot_out = slots[e];
        }
        return (const char *) bank->data + (size_t) slots[e]*w->nb[2];
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    // overlap mode ------------------------------------------------------------------------------------------------------

    bool overlap_usable(int32_t il, int64_t n_tokens) const {
        return overlap_level != 0 && n_tokens >= 1 && n_tokens <= overlap_max_tokens && il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 && layers[il].slot_map_h != nullptr;
    }

    bool copy_nosync(const ggml_tensor * src) const {
        return overlap_level != 0 && layer_of_m.find(src) != layer_of_m.end();
    }

    // the experts missing from the arena are read by the upload pool into the free slots: starts the reads and returns
    void overlap_start_fill(const layer & l, group & g, const std::vector<moe_cache_lru::fill> & fills, std::vector<llama_exps_prefetch::upload_job> & jobs, size_t & bytes) {
        jobs.clear();
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            GGML_ASSERT(bank->nb[2] == expert_size);
            int      fd       = -1;
            uint64_t file_off = 0;
            const bool direct = moe_fill_lookup(w->data, ggml_nbytes(w), &fd, &file_off);
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                GGML_ASSERT(fills[i].slot + (int64_t) n <= bank->ne[2]);
                const size_t src_off = (size_t) fills[i].expert*expert_size;
                const size_t dst_off = (size_t) fills[i].slot*expert_size;
                // without a file (fd -1) the pool copies from the mapping
                llama_exps_prefetch::add_upload(jobs, direct ? fd : -1, direct ? file_off + src_off : 0, (char *) bank->data + dst_off, n*expert_size, (const char *) w->data + src_off);
                bytes += n*expert_size;
                i += n;
            }
        }
        if (!jobs.empty()) {
            llama_exps_prefetch::upload_begin(jobs);
        }
    }

    // split 1 of the layer: plan, start the upload, publish the hit map
    bool copy_h(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const int32_t il = layer_of_h.at(src);
        const layer & l = layers[il];
        group & g = groups[l.ig];
        GGML_ASSERT(backend == devices[g.id].backend);
        // a graph aborted between split 1 and split 2 leaves an upload in flight: drain it before planning again
        overlap_drain();

        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        const is_shared_fn is_shared = get_is_shared(backend);
        const ggml_tensor * sel_buf_t = sel->view_src ? sel->view_src : sel;
        if (is_shared == nullptr || sel->data == nullptr || sel_buf_t->buffer == nullptr || !is_shared(sel_buf_t->buffer) ||
                dst->data == nullptr || dst->buffer == nullptr || !is_shared(dst->buffer)) {
            GGML_ABORT("LLAMA_DSV41_MOE_OVERLAP needs shared (unified memory) Metal buffers");
        }

        const auto t0 = std::chrono::steady_clock::now();
        ids.resize(ggml_nelements(sel));
        // no queued GPU work still reads a slot that is overwritten below; the ids are read directly
        ggml_backend_synchronize(backend);
        memcpy(ids.data(), sel->data, ggml_nbytes(sel));
        const auto t1 = std::chrono::steady_clock::now();

        // the background prefetch of this layer's predicted misses must land before the plan (they count as hits)
        pred_resolve(il);
        const bool pf_on = llama_moe_cache_predict_mode() == 2;
        if (pf_on) {
            pf_resolve(il);
        }

        const int32_t n_expert = g.lru.n_expert;
        overlap_before.assign(g.lru.slot_map[il], g.lru.slot_map[il] + n_expert);
        if (pf_on) {
            pf_mask(overlap_before);
        }

        // the ubatch length drives the heat decay of the LFU policy, as in copy() (1 for single-token decode, the old call)
        size_t n_hit = 0;
        int32_t n_tok = 1;
        if (n_expert_used > 0 && ids.size() % (size_t) n_expert_used == 0) {
            n_tok = (int32_t) (ids.size() / (size_t) n_expert_used);
        }
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit, n_tok)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        size_t bytes = 0;
        if (!fills.empty()) {
            overlap_start_fill(l, g, fills, jobs, bytes);
        }
        size_t slot_bytes = 0;
        for (const ggml_tensor * w : l.experts) {
            slot_bytes += w->nb[2];
        }
        for (const auto & fill : fills) {
            GGML_ASSERT(fill.slot >= 0 && fill.slot < g.n_slots);
            if (!g.filled[fill.slot]) {
                g.filled[fill.slot] = true;
                g.filled_bytes += slot_bytes;
            }
        }

        // the hit map: the experts resident before this plan (the fills only write to slots that no selected expert uses)
        GGML_ASSERT(dst->ne[0]*dst->ne[1] == n_expert);
        memcpy(dst->data, overlap_before.data(), n_expert*sizeof(int32_t));

        overlap_fill_pending = !fills.empty();
        overlap_il = il;
        overlap_pending_bytes = bytes;
        const auto t2 = std::chrono::steady_clock::now();

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        overlap_small = &st == &stats_small;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;
        st.calls  += 1;
        st.tokens += sel->ne[1];
        st.pread_calls += !fills.empty();
        st.sync_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        st.fill_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (&st == &stats_small) {
            tok_add(il, (int32_t) sel->ne[1], n_hit, fills.size(), bytes,
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        }
        return true;
    }

    // split 2 of the layer: wait for the upload, publish the fill map
    bool copy_m(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
        const int32_t il = layer_of_m.at(src);
        GGML_ASSERT(overlap_il == il);
        const layer & l = layers[il];
        group & g = groups[l.ig];

        const auto t0 = std::chrono::steady_clock::now();
        if (llama_moe_cache_predict_mode() == 2) {
            // look-ahead: queued while the demand reads of this layer are in flight and the GPU runs the hit pass, so the
            // planning work is off the critical path (the queue yields to the demand reads)
            const auto tl = std::chrono::steady_clock::now();
            pf_launch(backend, il);
            pfs.launch_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - tl).count();
        }
        if (warm_enabled()) {
            warm_step(backend, il);
        }
        if (overlap_fill_pending) {
            llama_exps_prefetch::upload_end();
            overlap_fill_pending = false;
        }
        if (llama_moe_cache_predict_mode() == 2) {
            // prefetches of this layer that were still in flight at its plan
            pf_join();
        }
        const auto t1 = std::chrono::steady_clock::now();

        // LLAMA_DSV41_MOE_PREDICT=1: read the predicted misses of the next layer while the GPU computes this one
        if (llama_moe_cache_predict_mode() == 1) {
            pred_launch(backend, il);
        }

        const int32_t n_expert = g.lru.n_expert;
        const int32_t * now = g.lru.slot_map[il];
        int32_t * out = (int32_t *) dst->data;
        GGML_ASSERT(dst->data != nullptr && dst->ne[0]*dst->ne[1] == n_expert);
        if (overlap_level >= 2) {
            // only the experts the fill brought
            for (int32_t e = 0; e < n_expert; ++e) {
                out[e] = (overlap_before[e] < 0 && now[e] >= 0) ? now[e] : -1;
            }
        } else {
            memcpy(out, now, n_expert*sizeof(int32_t));
        }
        // the wait counts where copy_h counted the layer (always stats_small for single-token decode)
        (overlap_small ? stats_small : stats_large).wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        if (overlap_small) {
            tok_cur.wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        }
        return true;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        if (overlap_level != 0) {
            if (layer_of_h.find(src) != layer_of_h.end()) {
                return copy_h(backend, src, dst, graph);
            }
            if (layer_of_m.find(src) != layer_of_m.end()) {
                return copy_m(backend, src, dst);
            }
        }
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        pf_flush();
        const int32_t il = it->second;
        const layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        // LLAMA_DSV41_MOE_DIRECT=1: on a shared (unified memory) Metal buffer the ids are read by the CPU directly after
        // the backend is idle, and the slot map is written directly while the GPU is idle: no blit command buffers
        const is_shared_fn is_shared_d = direct_enabled() ? get_is_shared(backend) : nullptr;
        const bool direct_ids = is_shared_d != nullptr && sel->data != nullptr &&
            (sel->view_src ? sel->view_src->buffer : sel->buffer) != nullptr &&
            is_shared_d(sel->view_src ? sel->view_src->buffer : sel->buffer);
        const bool direct_map = is_shared_d != nullptr && dst->data != nullptr && dst->buffer != nullptr && is_shared_d(dst->buffer);

        const auto t0 = std::chrono::steady_clock::now();
        ids.resize(ggml_nelements(sel));
        if (direct_ids) {
            // also guarantees that no queued GPU work still reads a slot that is overwritten below
            ggml_backend_synchronize(backend);
            memcpy(ids.data(), sel->data, ggml_nbytes(sel));
        } else {
            ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
            // also guarantees that no queued GPU work still reads a slot that is overwritten below
            ggml_backend_synchronize(backend);
        }
        const auto t1 = std::chrono::steady_clock::now();

        // the background prefetch of this layer's predicted misses must land before the plan and the GPU read
        pred_resolve(il);

        // LLAMA_DSV41_MOE_TRACE=path: append (il, n_ids, ids...) as int32 for every decode call (offline policy simulation)
        {
            static FILE * trace = [] {
                const char * v = getenv("LLAMA_DSV41_MOE_TRACE");
                return v && *v ? fopen(v, "wb") : (FILE *) nullptr;
            }();
            if (trace != nullptr && sel->ne[1] <= 8) {
                int32_t hdr[3] = { il, (int32_t) sel->ne[1], (int32_t) ids.size() };
                fwrite(hdr, sizeof(hdr), 1, trace);
                fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
                fflush(trace);
            }
        }

        size_t n_hit = 0;
        int32_t n_tok = 1;
        if (n_expert_used > 0 && ids.size() % (size_t) n_expert_used == 0) {
            n_tok = (int32_t) (ids.size() / (size_t) n_expert_used);
        }
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit, n_tok)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        // deepseek41 fast path: the bank is a shared Metal buffer (CPU-addressable) and the backend is synchronized, so
        // the misses of all expert tensors of the layer are pread() by the upload pool straight into the bank from the
        // GGUF file (the same bytes as the mmap) in one batch; otherwise ggml_backend_tensor_set_async from the mmap
        size_t bytes = 0;
        bool   pread_used    = false;
        bool   fallback_used = false;
        if (!fills.empty()) {
            const is_shared_fn is_shared = llama_exps_prefetch::upload_enabled() ? get_is_shared(backend) : nullptr;
            jobs.clear();
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const ggml_tensor * w    = l.experts[ip];
                ggml_tensor       * bank = g.banks[ip];
                const size_t expert_size = w->nb[2];
                GGML_ASSERT(bank->nb[2] == expert_size);
                int      fd       = -1;
                uint64_t file_off = 0;
                const bool direct = is_shared != nullptr && bank->buffer != nullptr && bank->data != nullptr && is_shared(bank->buffer) &&
                    moe_fill_lookup(w->data, ggml_nbytes(w), &fd, &file_off);
                for (size_t i = 0; i < fills.size();) {
                    size_t n = 1;
                    while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                        n++;
                    }
                    // slots < n_slots, the bank has n_slots + 1 entries
                    GGML_ASSERT(fills[i].slot + (int64_t) n <= bank->ne[2]);
                    const size_t src_off = (size_t) fills[i].expert*expert_size;
                    const size_t dst_off = (size_t) fills[i].slot*expert_size;
                    if (direct) {
                        llama_exps_prefetch::add_upload(jobs, fd, file_off + src_off, (char *) bank->data + dst_off, n*expert_size, (const char *) w->data + src_off);
                        pread_used = true;
                    } else {
                        ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + src_off, dst_off, n*expert_size);
                        fallback_used = true;
                    }
                    bytes += n*expert_size;
                    i += n;
                }
            }
            if (!jobs.empty()) {
                llama_exps_prefetch::upload(jobs);
            }
        }
        size_t slot_bytes = 0;
        for (const ggml_tensor * w : l.experts) {
            slot_bytes += w->nb[2];
        }
        for (const auto & fill : fills) {
            GGML_ASSERT(fill.slot >= 0 && fill.slot < g.n_slots);
            if (!g.filled[fill.slot]) {
                g.filled[fill.slot] = true;
                g.filled_bytes += slot_bytes;
            }
        }
        const auto t2 = std::chrono::steady_clock::now();

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;
        st.calls  += 1;
        st.tokens += sel->ne[1];
        st.pread_calls    += pread_used;
        st.fallback_calls += fallback_used;
        st.sync_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        st.fill_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (&st == &stats_small) {
            tok_add(il, (int32_t) sel->ne[1], n_hit, fills.size(), bytes,
                std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count(),
                std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
        }

        // the next copy synchronizes the backend before it changes the slot map again
        // (the LFU's array is src->data, or the GPU-resident map with LLAMA_DSV41_MOE_GPURES)
        const int32_t * map = g.lru.slot_map[il];
        GGML_ASSERT(map != nullptr && ggml_nbytes(src) == (size_t) g.lru.n_expert*sizeof(int32_t));
        if (direct_map) {
            // the GPU has been idle since the sync above (nothing was submitted by the plan or the fill)
            memcpy(dst->data, map, ggml_nbytes(src));
        } else {
            ggml_backend_tensor_set_async(backend, dst, map, 0, ggml_nbytes(src));
        }

        if (llama_moe_cache_predict_enabled() && sel->ne[1] == 1) {
            pred_launch(backend, il);
        }

        return true;
    }

    void log_stats() const {
        // on stderr: llama-bench silences the llama log
        const int32_t n_layers = std::max<int32_t>(1, n_cached_layers);
        auto log = [&](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            const double tok = std::max(1.0, (double) st.tokens/n_layers); // tokens served through the cache
            fprintf(stderr, "llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB, "
                "tokens = %.0f, misses/token = %.1f, MiB/token = %.2f, fill ms/token = %.2f, sync ms/token = %.2f, join wait ms/token = %.2f, "
                "calls = %zu (pread %zu, set_async fallback %zu)\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0,
                tok, st.misses/tok, st.bytes/1024.0/1024.0/tok, st.fill_us/1000.0/tok, st.sync_us/1000.0/tok, st.wait_us/1000.0/tok,
                st.calls, st.pread_calls, st.fallback_calls);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            fprintf(stderr, "llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

bool llama_moe_cache::get_slot_maps(int32_t il, int64_t n_tokens, int64_t n_expert_used, ggml_tensor ** h, ggml_tensor ** m) const {
    return pimpl->get_slot_maps(il, n_tokens, n_expert_used, h, m);
}

int llama_moe_cache::overlap_level() const {
    return pimpl->overlap_level;
}

bool llama_moe_cache::get_gpures_maps(int32_t il, int64_t n_tokens, int64_t n_expert_used, ggml_tensor ** map, ggml_tensor ** fill) const {
    return pimpl->get_gpures_maps(il, n_tokens, n_expert_used, map, fill);
}

int64_t llama_moe_cache::overlap_max_tokens() const {
    return pimpl->overlap_level != 0 ? pimpl->overlap_max_tokens : 0;
}

size_t llama_moe_cache::active_bytes() const {
    return pimpl->active_bytes();
}

void llama_moe_cache::shrink(size_t target) {
    pimpl->shrink(target);
}

bool llama_moe_cache::copy_nosync(const ggml_tensor * src) const {
    return pimpl->copy_nosync(src);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

void llama_moe_cache::set_graph(ggml_cgraph * gf) {
    pimpl->set_graph(gf);
}

int llama_moe_cache_predict_mode() {
    static const int mode = [] {
        const char * v = getenv("LLAMA_DSV41_MOE_PREDICT");
        return v != nullptr ? atoi(v) : 0;
    }();
    return mode;
}

bool llama_moe_cache_predict_enabled() {
    return llama_moe_cache_predict_mode() != 0;
}

int llama_moe_cache_predict_k() {
    static const int k = [] {
        const char * v = getenv("LLAMA_DSV41_PREDICT_K");
        return v != nullptr ? std::max(1, std::min(16, atoi(v))) : 6;
    }();
    return k;
}

const void * llama_moe_cache::resident_expert(const ggml_tensor * w, int64_t e) {
    return pimpl->resident_expert(w, e);
}

size_t llama_moe_cache::seed_layer(const ggml_tensor * w, const float * counts, int32_t n_rows, int32_t max_n,
        const std::function<const void *(const ggml_tensor *, int64_t)> & src) {
    return pimpl->seed_layer(w, counts, n_rows, max_n, src);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
