// Synthetic checks for moe_cache_lru. Build:
//   c++ -std=c++17 -Isrc tests/test-moe-cache-evict.cpp -o build/test-moe-cache-evict
#include "llama-moe-cache-policy.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_fails = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_fails++; \
    } \
} while (0)

struct Harness {
    int32_t n_layer;
    int32_t n_expert;
    int32_t n_slots;
    std::vector<std::vector<int32_t>> maps;
    moe_cache_lru lru;

    Harness(int32_t layers, int32_t experts, int32_t slots, moe_cache_lru::evict_kind kind)
        : n_layer(layers), n_expert(experts), n_slots(slots), maps(layers, std::vector<int32_t>(experts, -1)) {
        lru.init(layers, experts, slots);
        lru.evict = kind;
        lru.decay = 0.9f;
        lru.decay_tokens = 1000000; // decay tests set their own period
        lru.n_group_layers = layers;
        for (int32_t i = 0; i < layers; ++i) {
            lru.slot_map[i] = maps[i].data();
        }
    }

    bool plan(int32_t il, const std::vector<int32_t> & ids, int32_t n_tokens = 1) {
        std::vector<moe_cache_lru::fill> fills;
        size_t n_hit = 0;
        const bool ok = lru.plan(il, ids.data(), ids.size(), fills, n_hit, n_tokens);
        last_fills = std::move(fills);
        last_hits = n_hit;
        return ok;
    }

    std::vector<moe_cache_lru::fill> last_fills;
    size_t last_hits = 0;
};

static void test_lru_evicts_oldest() {
    Harness h(1, 8, 3, moe_cache_lru::evict_kind::lru);
    CHECK(h.plan(0, {0}));
    CHECK(h.plan(0, {1}));
    CHECK(h.plan(0, {2}));
    CHECK(h.plan(0, {0})); // refresh 0
    CHECK(h.plan(0, {3}));
    CHECK(h.maps[0][0] >= 0);
    CHECK(h.maps[0][1] < 0); // oldest unreused
    CHECK(h.maps[0][2] >= 0);
    CHECK(h.maps[0][3] >= 0);
    CHECK(h.last_fills.size() == 1);
    CHECK(h.last_fills[0].expert == 3);
}

static void test_lfu_keeps_hot_against_lru() {
    // Two slots. Expert 0 is used twice, then expert 1 once. LRU's head is expert 0.
    // A new expert must evict expert 1 (low heat), not expert 0.
    Harness h(1, 8, 2, moe_cache_lru::evict_kind::lfu_decay);
    CHECK(h.plan(0, {0}));
    CHECK(h.plan(0, {0}));
    CHECK(h.plan(0, {1}));
    CHECK(h.lru.head >= 0);
    CHECK(h.lru.key_of[h.lru.head] % h.n_expert == 0);
    CHECK(h.plan(0, {2}));
    CHECK(h.maps[0][0] >= 0);
    CHECK(h.maps[0][1] < 0);
    CHECK(h.maps[0][2] >= 0);
}

static void test_protects_current_selection() {
    Harness h(1, 8, 2, moe_cache_lru::evict_kind::lfu_decay);
    CHECK(h.plan(0, {0}));
    CHECK(h.plan(0, {1}));
    h.lru.heat[0] = 0.0f;   // selected, coldest
    h.lru.heat[1] = 100.0f; // not selected
    CHECK(h.plan(0, {0, 2}));
    CHECK(h.maps[0][0] >= 0);
    CHECK(h.maps[0][1] < 0);
    CHECK(h.maps[0][2] >= 0);
}

static void test_multiple_misses_keep_new_fills() {
    Harness h(1, 8, 2, moe_cache_lru::evict_kind::lfu_decay);
    CHECK(h.plan(0, {0}));
    CHECK(h.plan(0, {1}));
    CHECK(h.plan(0, {3, 2})); // two misses, unsorted on purpose
    CHECK(h.maps[0][2] >= 0);
    CHECK(h.maps[0][3] >= 0);
    CHECK(h.maps[0][0] < 0);
    CHECK(h.maps[0][1] < 0);
    CHECK(h.last_fills.size() == 2);
    CHECK(h.last_fills[0].expert == 2);
    CHECK(h.last_fills[1].expert == 3);
    CHECK(h.last_fills[0].slot != h.last_fills[1].slot);
}

static void test_empty_before_occupied() {
    Harness h(1, 8, 2, moe_cache_lru::evict_kind::lfu_decay);
    CHECK(h.plan(0, {0}));
    h.lru.heat[0] = 0.0f;
    const int32_t slot0 = h.maps[0][0];
    CHECK(h.plan(0, {1}));
    CHECK(h.maps[0][0] == slot0);
    CHECK(h.maps[0][1] >= 0);
    CHECK(h.maps[0][1] != slot0);
}

static void test_tie_breaks_by_lru() {
    Harness h(1, 8, 2, moe_cache_lru::evict_kind::lfu_decay);
    CHECK(h.plan(0, {0}));
    CHECK(h.plan(0, {1}));
    h.lru.heat[0] = 5.0f;
    h.lru.heat[1] = 5.0f;
    // expert 0 is the head (loaded first, expert 1 touched after)
    CHECK(h.lru.key_of[h.lru.head] % h.n_expert == 0);
    CHECK(h.plan(0, {2}));
    CHECK(h.maps[0][0] < 0);
    CHECK(h.maps[0][1] >= 0);
    CHECK(h.maps[0][2] >= 0);
}

static void test_heat_survives_eviction() {
    Harness h(1, 8, 2, moe_cache_lru::evict_kind::lfu_decay);
    CHECK(h.plan(0, {0}));
    CHECK(h.plan(0, {1}));
    h.lru.heat[0] = 1.0f;
    h.lru.heat[1] = 50.0f;
    CHECK(h.plan(0, {2}));
    CHECK(h.maps[0][0] < 0);
    CHECK(h.lru.heat[0] == 1.0f);
    CHECK(h.plan(0, {0})); // bring it back; heat was retained, then +1
    CHECK(h.maps[0][0] >= 0);
    CHECK(h.lru.heat[0] == 2.0f);
}

static void test_decay_and_repeat_counts() {
    Harness h(1, 4, 4, moe_cache_lru::evict_kind::lfu_decay);
    h.lru.decay = 0.5f;
    h.lru.decay_tokens = 4;
    h.lru.n_group_layers = 1;
    CHECK(h.plan(0, {0, 0})); // two occurrences, one token
    CHECK(h.lru.heat[0] == 2.0f);
    CHECK(h.lru.token_acc == 1);
    CHECK(h.plan(0, {1}, 3)); // acc becomes 4, decay, then +1
    CHECK(h.lru.token_acc == 0);
    CHECK(std::fabs(h.lru.heat[0] - 1.0f) < 1e-6f); // 2 * 0.5
    CHECK(std::fabs(h.lru.heat[1] - 1.0f) < 1e-6f);
    // two periods at once, group of 2 layers: period = 2 * 2 = 4, 8 tokens -> ^2
    Harness g(2, 4, 4, moe_cache_lru::evict_kind::lfu_decay);
    g.lru.decay = 0.9f;
    g.lru.decay_tokens = 2;
    g.lru.n_group_layers = 2;
    g.lru.heat[0] = 100.0f;
    CHECK(g.plan(1, {1}, 8));
    CHECK(g.lru.token_acc == 0); // 8 / (2*2) = 2 decays, remainder 0
    CHECK(std::fabs(g.lru.heat[0] - 81.0f) < 1e-4f); // 100 * 0.9 * 0.9
    CHECK(g.lru.heat[1 * 4 + 1] == 1.0f);
}

static void test_capacity() {
    Harness h(1, 8, 2, moe_cache_lru::evict_kind::lfu_decay);
    CHECK(h.plan(0, {0}));
    const int32_t slot = h.maps[0][0];
    CHECK(!h.plan(0, {1, 2, 3}));
    CHECK(h.maps[0][0] == slot);
    CHECK(h.maps[0][1] < 0);
    CHECK(h.lru.heat[1] == 0.0f); // failed plan does not add heat
    CHECK(h.plan(0, {0, 1})); // still works after a rejected call
    CHECK(h.maps[0][0] == slot);
    CHECK(h.maps[0][1] >= 0);
}

static void test_seed_is_heat_only() {
    const char * path = "build/test-moe-seed.bin";
    std::FILE * f = std::fopen(path, "wb");
    CHECK(f != nullptr);
    if (f == nullptr) {
        return;
    }
    std::vector<uint32_t> raw(43 * 384, 0);
    raw[0] = 10;       // layer 0 expert 0
    raw[1] = 5;        // layer 0 expert 1
    raw[2 * 384 + 3] = 7;
    CHECK(std::fwrite(raw.data(), sizeof(uint32_t), raw.size(), f) == raw.size());
    std::fclose(f);

    setenv("LLAMA_DSV41_MOE_EVICT", "lfu-decay", 1);
    setenv("LLAMA_DSV41_MOE_SEED_PROFILE", path, 1);
    moe_cache_lru lru;
    lru.init(40, 384, 4);
    unsetenv("LLAMA_DSV41_MOE_SEED_PROFILE");
    unsetenv("LLAMA_DSV41_MOE_EVICT");

    CHECK(std::fabs(lru.heat[0] - moe_cache_lru::k_seed_peak) < 1e-5f);
    CHECK(std::fabs(lru.heat[1] - moe_cache_lru::k_seed_peak * 0.5f) < 1e-5f);
    CHECK(std::fabs(lru.heat[2 * 384 + 3] - moe_cache_lru::k_seed_peak) < 1e-5f);
    for (int32_t s = 0; s < lru.n_slots; ++s) {
        CHECK(lru.key_of[s] < 0);
    }
}

int main() {
    unsetenv("LLAMA_DSV41_MOE_EVICT");
    unsetenv("LLAMA_DSV41_MOE_SEED_PROFILE");
    unsetenv("LLAMA_DSV41_MOE_DECAY");
    unsetenv("LLAMA_DSV41_MOE_DECAY_TOKENS");

    moe_cache_lru def;
    def.init(1, 2, 1);
    CHECK(def.evict == moe_cache_lru::evict_kind::lru);

    test_lru_evicts_oldest();
    test_lfu_keeps_hot_against_lru();
    test_protects_current_selection();
    test_multiple_misses_keep_new_fills();
    test_empty_before_occupied();
    test_tie_breaks_by_lru();
    test_heat_survives_eviction();
    test_decay_and_repeat_counts();
    test_capacity();
    test_seed_is_heat_only();

    if (g_fails != 0) {
        std::fprintf(stderr, "%d checks failed\n", g_fails);
        return 1;
    }
    std::fprintf(stderr, "moe cache eviction checks passed\n");
    return 0;
}
