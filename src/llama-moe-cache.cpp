#include "llama-moe-cache.h"

#include "llama-exps-prefetch.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
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
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

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
    bool overlap_fill_pending = false;
    int32_t overlap_il = -1;
    size_t  overlap_pending_bytes = 0;
    std::vector<int32_t> overlap_before; // slot map of the layer before the plan

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

    void pred_wait() {
        if (pred_busy) {
            const auto t0 = std::chrono::steady_clock::now();
            pred_th.join();
            pst.wait_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            pred_busy = false;
        }
    }

    void set_graph(ggml_cgraph * gf) {
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
        if (!G.lru.plan(nl, pred_ids.data(), pred_ids.size(), pred_fills, n_hit)) {
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
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
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
                const is_shared_fn is_shared = get_is_shared(d.backend);
                // Shared Metal slots are written completely before their GPU map is published.
                if (is_shared == nullptr || !is_shared(d.buf.get())) {
                    ggml_backend_buffer_clear(d.buf.get(), 0);
                    if (overlap_level != 0) {
                        fprintf(stderr, "llama_moe_cache: the arena of %s is not a shared buffer, LLAMA_DSV41_MOE_OVERLAP disabled\n", ggml_backend_name(d.backend));
                        overlap_level = 0;
                    }
                }
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            if (overlap_level != 0) {
                fprintf(stderr, "llama_moe_cache: overlap level %d (async arena fill)\n", overlap_level);
            }
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
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
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            fprintf(stderr, "llama_moe_cache: group %zu: %2zu layers, %s, %5d slots (%.1f%%, %.1f slots/layer)\n", ig,
                g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                (double) g.n_slots/g.layers.size());
        }
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
        overlap_drain();
        pred_wait();
        log_arena("teardown");
        log_stats();
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

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
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
        return overlap_level != 0 && n_tokens == 1 && il >= 0 && il < (int32_t) layers.size() && layers[il].ig >= 0 && layers[il].slot_map_h != nullptr;
    }

    bool copy_nosync(const ggml_tensor * src) const {
        return overlap_level != 0 && layer_of_m.find(src) != layer_of_m.end();
    }

    // the experts missing from the arena are read by the upload pool into the free slots: starts the reads and returns
    void overlap_start_fill(ggml_backend_t backend, const layer & l, group & g, size_t & bytes) {
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
        (void) backend;
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

        const int32_t n_expert = g.lru.n_expert;
        overlap_before.assign(g.lru.slot_map[il], g.lru.slot_map[il] + n_expert);

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        size_t bytes = 0;
        if (!fills.empty()) {
            overlap_start_fill(backend, l, g, bytes);
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
        st.hits   += n_hit;
        st.misses += fills.size();
        st.bytes  += bytes;
        st.calls  += 1;
        st.tokens += sel->ne[1];
        st.pread_calls += !fills.empty();
        st.sync_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        st.fill_us += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        return true;
    }

    // split 2 of the layer: wait for the upload, publish the fill map
    bool copy_m(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst) {
        const int32_t il = layer_of_m.at(src);
        GGML_ASSERT(overlap_il == il);
        const layer & l = layers[il];
        group & g = groups[l.ig];

        const auto t0 = std::chrono::steady_clock::now();
        if (overlap_fill_pending) {
            llama_exps_prefetch::upload_end();
            overlap_fill_pending = false;
        }
        const auto t1 = std::chrono::steady_clock::now();

        // LLAMA_DSV41_MOE_PREDICT=1: read the predicted misses of the next layer while the GPU computes this one
        if (llama_moe_cache_predict_enabled()) {
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
        (stats_small).wait_us += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
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

        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
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

        // the next copy synchronizes the backend before it changes the slot map again
        if (direct_map) {
            // the GPU has been idle since the sync above (nothing was submitted by the plan or the fill)
            memcpy(dst->data, src->data, ggml_nbytes(src));
        } else {
            ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
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

bool llama_moe_cache_predict_enabled() {
    static const bool on = [] {
        const char * v = getenv("LLAMA_DSV41_MOE_PREDICT");
        return v != nullptr && atoi(v) != 0;
    }();
    return on;
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
