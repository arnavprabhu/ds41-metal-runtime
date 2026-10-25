#pragma once

// Eviction for one MoE-cache group.
//
// LLAMA_DSV41_MOE_EVICT=lru|lfu-decay (default lru). lru picks the same victim as before.
// lfu-decay keeps a heat value per (layer, expert):
//   +1 for every selected id (repeats in the ubatch count)
//   every LLAMA_DSV41_MOE_DECAY_TOKENS tokens (default 256) multiply every heat by
//   LLAMA_DSV41_MOE_DECAY (default 0.9). Half-life is about 1685 tokens.
//   Alternative, not the default: DECAY=0.5 (half-life = the period, 256 tokens).
// A group's token counter advances by the ubatch length once per cached layer, and the
// period is scaled by the number of cached layers in the group, so 256 means 256 tokens.
// Decay runs before this call's +1, so the crossing token is recorded at full weight.
// Victim: an empty slot if any, otherwise the lowest heat among slots whose expert is
// not selected by this call (hits and misses already assigned). Equal heat keeps the
// least-recently-used slot. Misses are still emitted in expert-id order.
//
// LLAMA_DSV41_MOE_SEED_PROFILE=path loads raw little-endian u32 counts [43][384]
// (Strata profile-2.00.bin). Each layer is scaled so its hottest expert starts at
// k_seed_peak (4) selections. Slots are not filled. Rows past n_layer, and the zero
// rows 40..42 of that file, stay cold. Ignored unless evict is lfu-decay.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef MOE_CACHE_ASSERT
#ifdef GGML_ASSERT
#define MOE_CACHE_ASSERT(x) GGML_ASSERT(x)
#else
#define MOE_CACHE_ASSERT(x) \
    do { \
        if (!(x)) { \
            std::fprintf(stderr, "moe_cache assert failed: %s\n", #x); \
            std::abort(); \
        } \
    } while (0)
#endif
#endif

struct moe_cache_lru {
    enum class evict_kind { lru, lfu_decay };

    // Hottest profile expert starts as this many selections. A few real hits overtake it.
    static constexpr float k_seed_peak = 4.0f;
    static constexpr int k_profile_layers  = 43;
    static constexpr int k_profile_experts = 384;
    static constexpr float k_decay_default = 0.9f;
    static constexpr int32_t k_decay_tokens_default = 256;

    int32_t n_expert = 0;
    int32_t n_layer  = 0;
    int32_t n_slots  = 0;
    int32_t n_group_layers = 0; // cached layers in this group; 0 = count non-null slot maps

    evict_kind evict = evict_kind::lru;
    float   decay = k_decay_default;
    int32_t decay_tokens = k_decay_tokens_default;
    int32_t token_acc = 0; // layer-visits toward decay_tokens * n_group_layers

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty
    std::vector<float>     heat;     // [n_layer * n_expert], kept after eviction

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;
    std::vector<int32_t> lru_rank; // [n_slots], 0 = least recently used

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_layer  = n_layer;
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        heat.assign((size_t) n_layer * (size_t) n_expert, 0.0f);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = n_slots > 0 ? 0 : -1;
        tail = n_slots > 0 ? n_slots - 1 : -1;
        seen.assign(n_expert, 0);
        seen_gen = 0;
        token_acc = 0;
        n_group_layers = 0;
        read_env();
        const char * seed = nullptr;
        if (evict == evict_kind::lfu_decay) {
            seed = std::getenv("LLAMA_DSV41_MOE_SEED_PROFILE");
            if (seed != nullptr && seed[0] != '\0') {
                if (!load_seed(seed)) {
                    seed = nullptr;
                }
            } else {
                seed = nullptr;
            }
        }
        static int logged = 0;
        if (logged++ == 0) {
            std::fprintf(stderr,
                "llama_moe_cache: evict %s, decay %.2f every %d tokens (alt LLAMA_DSV41_MOE_DECAY=0.5), seed %s\n",
                evict == evict_kind::lfu_decay ? "lfu-decay" : "lru",
                decay, decay_tokens, seed != nullptr ? seed : "off");
        }
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il
    // returns false if the ids select more distinct experts than there are slots
    // n_tokens is the ubatch length (ids hold n_tokens * n_expert_used ids)
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, int32_t n_tokens = 1) {
        fills.clear();
        n_hit = 0;
        if (!dedup(il, ids, n_ids)) {
            return false;
        }
        if (evict == evict_kind::lru) {
            return plan_lru(il, fills, n_hit);
        }
        return plan_lfu(il, ids, n_ids, n_tokens, fills, n_hit);
    }

    // ---- speculative prefetch (LLAMA_DSV41_MOE_PREDICT) ------------------------------------------------------------------
    // prefetch_begin: the slots of the experts layer pl is reading right now may not be taken as victims until the next
    // prefetch_begin/prefetch_end; prefetch_take gives a slot to expert e of layer il without counting a selection (no heat
    // bump) when it is not resident and a victim exists whose heat is below heat(e) + bonus. unmap forgets a prefetch that was
    // cancelled before any byte was read (the slot becomes empty).
    std::vector<uint32_t> prot;
    uint32_t prot_gen = 0;
    bool prot_active = false;

    void prefetch_begin(int32_t pl, const int32_t * ids, size_t n_ids) {
        if (prot.size() != (size_t) n_slots) {
            prot.assign(n_slots, 0);
            prot_gen = 0;
        }
        if (++prot_gen == 0) {
            std::fill(prot.begin(), prot.end(), 0);
            prot_gen = 1;
        }
        prot_active = true;
        for (size_t i = 0; i < n_ids; ++i) {
            const int32_t s = slot_map[pl][ids[i]];
            if (s >= 0) {
                prot[s] = prot_gen;
            }
        }
        rank_lru();
    }
    void prefetch_end() {
        prot_active = false;
    }
    bool prefetch_take(int32_t il, int32_t e, float bonus, fill & out) {
        int32_t * slots = slot_map[il];
        if (slots[e] >= 0) {
            return false;
        }
        int32_t best = -1;
        float best_h = 0.0f;
        int32_t best_r = 0;
        bool best_empty = false;
        for (int32_t s = 0; s < n_slots; ++s) {
            if (prot[s] == prot_gen) {
                continue;
            }
            const int32_t key = key_of[s];
            const bool empty = key < 0;
            const float h = empty ? 0.0f : heat[(size_t) key];
            const int32_t r = lru_rank.empty() ? 0 : lru_rank[s];
            const bool better = best < 0 || (empty && !best_empty) ||
                (empty == best_empty && (h < best_h || (h == best_h && r < best_r)));
            if (better) {
                best = s; best_h = h; best_r = r; best_empty = empty;
            }
        }
        if (best < 0) {
            return false;
        }
        if (evict == evict_kind::lfu_decay && !best_empty && heat[(size_t) il*n_expert + e] + bonus <= best_h) {
            return false;
        }
        if (key_of[best] >= 0) {
            slot_map[key_of[best] / n_expert][key_of[best] % n_expert] = -1;
        }
        key_of[best] = il*n_expert + e;
        slots[e] = best;
        prot[best] = prot_gen;
        out = { e, best };
        return true;
    }
    void unmap(int32_t il, int32_t e, int32_t slot) {
        MOE_CACHE_ASSERT(slot_map[il][e] == slot && key_of[slot] == il*n_expert + e);
        slot_map[il][e] = -1;
        key_of[slot] = -1;
    }

private:
    void read_env() {
        decay = k_decay_default;
        decay_tokens = k_decay_tokens_default;
        evict = evict_kind::lru;
        const char * mode = std::getenv("LLAMA_DSV41_MOE_EVICT");
        if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "lru") == 0) {
            evict = evict_kind::lru;
        } else if (std::strcmp(mode, "lfu-decay") == 0) {
            evict = evict_kind::lfu_decay;
        } else {
            std::fprintf(stderr, "llama_moe_cache: unknown LLAMA_DSV41_MOE_EVICT=%s, using lru\n", mode);
            evict = evict_kind::lru;
        }
        if (const char * d = std::getenv("LLAMA_DSV41_MOE_DECAY")) {
            char * end = nullptr;
            const float v = std::strtof(d, &end);
            if (end != d && v > 0.0f && v < 1.0f) {
                decay = v;
            } else {
                std::fprintf(stderr, "llama_moe_cache: ignoring LLAMA_DSV41_MOE_DECAY=%s\n", d);
            }
        }
        if (const char * t = std::getenv("LLAMA_DSV41_MOE_DECAY_TOKENS")) {
            char * end = nullptr;
            const long v = std::strtol(t, &end, 10);
            if (end != t && v >= 1 && v <= 100000000L) {
                decay_tokens = (int32_t) v;
            } else {
                std::fprintf(stderr, "llama_moe_cache: ignoring LLAMA_DSV41_MOE_DECAY_TOKENS=%s\n", t);
            }
        }
    }

    bool load_seed(const char * path) {
        if (n_expert != k_profile_experts) {
            std::fprintf(stderr, "llama_moe_cache: seed profile ignored, n_expert %d != %d\n", n_expert, k_profile_experts);
            return false;
        }
        std::FILE * f = std::fopen(path, "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "llama_moe_cache: cannot open seed profile %s\n", path);
            return false;
        }
        std::vector<uint32_t> raw((size_t) k_profile_layers * (size_t) k_profile_experts);
        const size_t n = std::fread(raw.data(), sizeof(uint32_t), raw.size(), f);
        std::fclose(f);
        if (n != raw.size()) {
            std::fprintf(stderr, "llama_moe_cache: seed profile %s: expected %zu u32, got %zu\n", path, raw.size(), n);
            return false;
        }
        const int32_t n_rows = std::min(n_layer, k_profile_layers);
        for (int32_t row = 0; row < n_rows; ++row) {
            const uint32_t * counts = raw.data() + (size_t) row * (size_t) k_profile_experts;
            uint32_t mx = 0;
            for (int e = 0; e < k_profile_experts; ++e) {
                mx = std::max(mx, counts[e]);
            }
            if (mx == 0) {
                continue;
            }
            const float scale = k_seed_peak / (float) mx;
            float * dst = heat.data() + (size_t) row * (size_t) n_expert;
            for (int e = 0; e < k_profile_experts; ++e) {
                dst[e] = (float) counts[e] * scale;
            }
        }
        return true;
    }

    int32_t group_layers() {
        if (n_group_layers > 0) {
            return n_group_layers;
        }
        int32_t n = 0;
        for (int32_t * p : slot_map) {
            if (p != nullptr) {
                n++;
            }
        }
        n_group_layers = n > 0 ? n : 1;
        return n_group_layers;
    }

    bool dedup(int32_t il, const int32_t * ids, size_t n_ids) {
        MOE_CACHE_ASSERT(il >= 0 && il < n_layer);
        MOE_CACHE_ASSERT(slot_map[il] != nullptr);
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            MOE_CACHE_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        return uniq.size() <= (size_t) n_slots;
    }

    // byte-identical to the original recency plan
    bool plan_lru(int32_t il, std::vector<fill> & fills, size_t & n_hit) {
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

    void note_tokens(int32_t n_tokens) {
        if (n_tokens < 0) {
            n_tokens = 0;
        }
        token_acc += n_tokens;
        const int32_t period = decay_tokens * group_layers();
        if (period <= 0 || token_acc < period || heat.empty()) {
            return;
        }
        int n_decay = token_acc / period;
        token_acc -= n_decay * period;
        float factor = 1.0f;
        for (int i = 0; i < n_decay; ++i) {
            factor *= decay;
        }
        if (factor == 1.0f) {
            return;
        }
        for (float & h : heat) {
            h *= factor;
        }
    }

    void rank_lru() {
        lru_rank.assign(n_slots, 0);
        int32_t r = 0;
        for (int32_t s = head; s >= 0; s = next[s]) {
            lru_rank[s] = r++;
        }
    }

    // empty slots first, then lowest heat, then least recently used. skip this call's experts.
    int32_t pick_victim(int32_t il) const {
        int32_t best = -1;
        float best_h = 0.0f;
        int32_t best_r = 0;
        bool best_empty = false;
        for (int32_t s = 0; s < n_slots; ++s) {
            if (prot_active && prot[s] == prot_gen) {
                continue;
            }
            const int32_t key = key_of[s];
            const bool empty = key < 0;
            if (!empty) {
                const int32_t layer = key / n_expert;
                const int32_t e = key % n_expert;
                if (layer == il && seen[e] == seen_gen) {
                    continue;
                }
            }
            const float h = empty ? 0.0f : heat[(size_t) key];
            const int32_t r = lru_rank[s];
            const bool better = best < 0 ||
                (empty && !best_empty) ||
                (empty == best_empty && (h < best_h || (h == best_h && r < best_r)));
            if (better) {
                best = s;
                best_h = h;
                best_r = r;
                best_empty = empty;
            }
        }
        return best;
    }

    bool plan_lfu(int32_t il, const int32_t * ids, size_t n_ids, int32_t n_tokens, std::vector<fill> & fills, size_t & n_hit) {
        int32_t * slots = slot_map[il];
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        note_tokens(n_tokens);
        float * row = heat.data() + (size_t) il * (size_t) n_expert;
        for (size_t i = 0; i < n_ids; ++i) {
            row[ids[i]] += 1.0f;
        }
        rank_lru();
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = pick_victim(il);
            if (s < 0) {
                return false;
            }
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
};
