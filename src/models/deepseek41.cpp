#include "llama-moe-cache.h"
#include "models.h"
#include "llama-impl.h"

#include "llama-kv-cache-dsv4.h"
#include "llama-exps-prefetch.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include <unistd.h>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

// ref: deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/model.py, engram.py
// compressed layers attend over [window | all compressed rows] with top-k as a mask. with flash attention the default
// is sparse: fused indexer, FA bounded to window + top-k, decode-sized ubatches gather only the selected rows (same
// keys and mask, different key order). LLAMA_DSV41_SPARSE=0 restores the reference path

static constexpr int64_t DSV41_ENGRAM_DEAD = -1;

static bool dsv41_env_on(const char * name) {
    const char * v = getenv(name);
    return !v || atoi(v) != 0;
}

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    llama_model_deepseek4::load_arch_hparams(ml);

    hparams.n_embd_out_impl = hparams.n_embd; // no hc_head
    if (hparams.indexer_top_k == 0) {
        throw std::runtime_error("DeepSeek-V4.1 indexer top_k must be > 0");
    }

    // each compressed layer reads the KV / top-k of the latest source layer at or before it
    // ISTA-DASLab GGUFs spell several keys after the HF config (kv_source_layer_ids, ...) and carry the Engram hash
    // constants as integer tensors: such a file must have every key it needs, nothing is silently disabled
    const std::string arch = ml.get_arch_name();
    const auto akey = [&](const char * suffix) { return arch + "." + suffix; };

    std::vector<uint32_t> kv_src;
    std::vector<uint32_t> idx_src;
    const bool ista = !ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYERS, kv_src, false);
    if (ista) {
        ml.get_arr(akey("kv_source_layer_ids"),    kv_src);
        ml.get_arr(akey("index_source_layer_ids"), idx_src);
    } else {
        ml.get_arr(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS, idx_src);
    }

    const uint32_t n_layer = hparams.n_layer();
    const auto out_of_range = [&](const std::vector<uint32_t> & v) {
        return std::any_of(v.begin(), v.end(), [&](uint32_t il) { return il >= n_layer; });
    };
    if (out_of_range(kv_src) || out_of_range(idx_src)) {
        throw std::runtime_error("DeepSeek-V4.1 kv/index source layer out of range");
    }
    hparams.dsv41_kv_src.fill(-1);
    hparams.dsv41_topk_src.fill(-1);
    int32_t last_kv  = -1;
    int32_t last_idx = -1;
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t ratio = hparams.dsv4_compress_ratios[il];
        const bool is_kv  = std::find(kv_src.begin(),  kv_src.end(),  il) != kv_src.end();
        const bool is_idx = std::find(idx_src.begin(), idx_src.end(), il) != idx_src.end();
        if ((is_kv || is_idx) && ratio == 0) {
            throw std::runtime_error(format("DeepSeek-V4.1 source layer %u has compress ratio 0", il));
        }
        last_kv  = is_kv  ? (int32_t) il : last_kv;
        last_idx = is_idx ? (int32_t) il : last_idx;
        if (ratio == 0) {
            continue;
        }
        if (ratio > 2 || last_kv < 0 || last_idx < last_kv ||
                hparams.dsv4_compress_ratios[last_kv] != ratio || hparams.dsv4_compress_ratios[last_idx] != ratio) {
            throw std::runtime_error(format("DeepSeek-V4.1 layer %u: ratio %u without a matching kv/index source", il, ratio));
        }
        hparams.dsv41_kv_src[il]   = last_kv;
        hparams.dsv41_topk_src[il] = last_idx;
    }

    if (ista) {
        // required: without them candidate masking would silently be off
        uint32_t cs = 0;
        ml.get_key(akey("attention.candidate_source_layer_id"), cs);
        ml.get_key(akey("attention.candidate_topk_blocks"),     hparams.dsv41_cand_topk_blocks);
        ml.get_key(akey("attention.candidate_block_size"),      hparams.dsv41_cand_block_size);
        if (cs >= n_layer) {
            throw std::runtime_error(format("DeepSeek-V4.1 candidate_source_layer_id %u out of range", cs));
        }
        hparams.dsv41_cand_src = (int32_t) cs;
    } else {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, hparams.dsv41_cand_src, false);
    }
    if (hparams.dsv41_cand_src >= 0) {
        if (!ista) {
            ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, hparams.dsv41_cand_topk_blocks);
            ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   hparams.dsv41_cand_block_size);
        }
        const int32_t cs = hparams.dsv41_cand_src;
        if ((uint32_t) cs >= n_layer || hparams.dsv41_topk_src[cs] != cs || hparams.dsv41_cand_block_size == 0 ||
                hparams.dsv41_cand_topk_blocks == 0) {
            throw std::runtime_error("DeepSeek-V4.1 invalid candidate selection (source must be an index source, block size and top-k blocks > 0)");
        }
        for (uint32_t il = cs + 1; il < n_layer; ++il) {
            if (hparams.dsv41_topk_src[il] == (int32_t) il && hparams.dsv4_compress_ratios[il] != hparams.dsv4_compress_ratios[cs]) {
                throw std::runtime_error("DeepSeek-V4.1 candidate consumers must share the source's compress ratio");
            }
        }
    }

    // long-prompt layer skip premise: from layer Y on, every layer has compressed attention fed by Y alone (Y owns the
    // compressor state and the index keys; later index sources only add queries from their own hidden states). Then the
    // rows of a prompt that precede its last (n_layer - Y)*(window - 1) + window ones need layers < Y and the compressor
    // of Y, nothing else (see llama_context prompt skip)
    hparams.dsv41_yoco = -1;
    for (uint32_t il = 0; il < n_layer; ++il) {
        bool ok = hparams.dsv41_kv_src[il] == (int32_t) il;
        for (uint32_t j = il; ok && j < n_layer; ++j) {
            ok = hparams.dsv4_compress_ratios[j] != 0 && hparams.dsv41_kv_src[j] == (int32_t) il;
        }
        if (ok) {
            hparams.dsv41_yoco = (int32_t) il;
            break;
        }
    }

    std::vector<uint32_t> e_layers;
    if (ista && !ml.get_arr(LLM_KV_PLE_LAYERS, e_layers, false)) {
        bool present = false;
        ml.get_key(akey("engram.present"), present);
        if (present) {
            load_engram_ista(ml, e_layers);
        }
    } else {
        ml.get_arr(LLM_KV_PLE_LAYERS, e_layers, false);
    }
    if (!e_layers.empty()) {
        if (!engram_per_layer) {
            ml.get_key(LLM_KV_PLE_NGRAM_SIZE,             engram_max_ngram);
            ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM,        engram_n_head);
            ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, engram_head_dim);
            ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS,      engram_mult);
            ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,           engram_offsets);
            ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES,       engram_primes);
            ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID,        engram_pad_id);
            ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,           engram_token_map);
        }

        const size_t n_e    = e_layers.size();
        const size_t n_cols = engram_n_cols();
        if (engram_max_ngram < 2 || engram_n_head == 0 || engram_head_dim == 0 ||
                engram_mult.size() != n_e*engram_max_ngram || engram_primes.size() != n_e*n_cols ||
                engram_offsets.size() != n_e*n_cols || engram_pad_id >= engram_token_map.size()) {
            throw std::runtime_error("DeepSeek-V4.1 engram metadata is inconsistent");
        }
        for (size_t e = 0; e < n_e; ++e) {
            if (e_layers[e] >= n_layer) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram layer %u is out of range", e_layers[e]));
            }
        }
        for (size_t c = 0; c < n_e*n_cols; ++c) {
            // a bucket count of 0 traps in the hash; row ids are int32
            if (engram_primes[c] < 2 || engram_offsets[c] + engram_primes[c] > (uint64_t) INT32_MAX) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram head %zu: bucket range out of range", c));
            }
        }
        engram_layers.assign(e_layers.begin(), e_layers.end());
        if (const char * v = getenv("LLAMA_DSV41_ENGRAM_OFF"); v && atoi(v) != 0) {
            LLAMA_LOG_WARN("%s: LLAMA_DSV41_ENGRAM_OFF is set: Engram is disabled, outputs are NOT the model's\n", __func__);
        }
        hparams.dsv41_tok_hist = true;
        hparams.dsv41_hist_len = engram_max_ngram > 0 ? (uint32_t) engram_max_ngram - 1 : 0;
    }

    {
        std::string ls;
        for (int32_t l : engram_layers) {
            ls += (ls.empty() ? "" : ",") + std::to_string(l);
        }
        const auto join = [](const std::vector<uint32_t> & v) {
            std::string r;
            for (uint32_t x : v) {
                r += (r.empty() ? "" : ",") + std::to_string(x);
            }
            return r;
        };
        LLAMA_LOG_INFO("%s: format = %s\n", __func__, ista ? "ISTA (HF-style keys, Engram constants as tensors)" : "llama.cpp (ple.* keys)");
        LLAMA_LOG_INFO("%s: kv_source_layers = [%s], index_source_layers = [%s]\n", __func__, join(kv_src).c_str(), join(idx_src).c_str());
        LLAMA_LOG_INFO("%s: long-prompt skip layer (yoco) = %d\n", __func__, hparams.dsv41_yoco);
        LLAMA_LOG_INFO("%s: candidate: source_layer = %d, top_k_blocks = %u, block_size = %u\n", __func__,
                hparams.dsv41_cand_src, hparams.dsv41_cand_topk_blocks, hparams.dsv41_cand_block_size);
        LLAMA_LOG_INFO("%s: engram: layers = [%s], max_ngram = %u, n_head = %u, head_dim = %u, pad_id = %u (mapped %d), "
                "token_map = %zu entries, tables = %s\n", __func__, ls.c_str(), engram_max_ngram, engram_n_head, engram_head_dim,
                engram_pad_id, engram_pad_id < engram_token_map.size() ? engram_token_map[engram_pad_id] : -1, engram_token_map.size(),
                engram_layers.empty() ? "none" : engram_per_layer ? "per layer (blk.N.engram_embd)" : "shared (per_layer_token_embd)");
        const size_t n_cols = engram_n_cols();
        for (size_t e = 0; e < engram_layers.size(); ++e) {
            std::string m;
            for (size_t o = 0; o < engram_max_ngram; ++o) {
                m += (o ? "," : "") + std::to_string(engram_mult[e*engram_max_ngram + o]);
            }
            LLAMA_LOG_INFO("%s: engram layer %d: multipliers = [%s], primes[0] = %" PRIu64 ", primes[%zu] = %" PRIu64
                    ", offsets[0] = %" PRIu64 ", offsets[%zu] = %" PRIu64 ", rows = %" PRIu64 "\n", __func__, engram_layers[e], m.c_str(),
                    engram_primes[e*n_cols], n_cols - 1, engram_primes[e*n_cols + n_cols - 1], engram_offsets[e*n_cols], n_cols - 1,
                    engram_offsets[e*n_cols + n_cols - 1], engram_offsets[e*n_cols + n_cols - 1] + engram_primes[e*n_cols + n_cols - 1]);
        }
    }
}

// ISTA: scalar keys under HF names, the hash constants as I32/I64 tensors of the first shard:
//   engram_token_map [n_vocab] I32, engram_primes / engram_offsets [cols, n_e] I64, engram_multipliers [ngram, n_e] I64
// offsets are relative to each layer's own table (blk.N.engram_embd, engram.num_embeddings rows)
void llama_model_deepseek41::load_engram_ista(llama_model_loader & ml, std::vector<uint32_t> & e_layers) {
    const std::string arch = ml.get_arch_name();
    const auto akey = [&](const char * suffix) { return arch + "." + suffix; };

    std::vector<uint32_t> rows;
    ml.get_arr(akey("engram.layer_ids"),      e_layers);
    ml.get_arr(akey("engram.num_embeddings"), rows);
    ml.get_key(akey("engram_max_ngram_size"), engram_max_ngram);
    ml.get_key(akey("engram_n_heads"),        engram_n_head);
    ml.get_key(akey("engram_head_dim"),       engram_head_dim);
    ml.get_key(akey("engram_pad_token_id"),   engram_pad_id);
    uint32_t cvocab = 0;
    ml.get_key(akey("engram_compressed_vocab_size"), cvocab);

    const size_t n_e    = e_layers.size();
    const size_t n_cols = engram_n_cols();
    if (n_e == 0 || rows.size() != n_e || engram_max_ngram < 2 || n_cols == 0) {
        throw std::runtime_error("DeepSeek-V4.1 ISTA engram metadata is inconsistent (layer_ids / num_embeddings / ngram / heads)");
    }

    // tensor -> values, checking type and shape
    const auto read = [&](const char * name, ggml_type type, int64_t ne0, int64_t ne1) { // ne0 < 0: any
        const auto * w = ml.get_weight(name);
        if (!w) {
            throw std::runtime_error(format("DeepSeek-V4.1 ISTA engram: tensor %s is missing", name));
        }
        const ggml_tensor * t = w->tensor;
        ne0 = ne0 < 0 ? t->ne[0] : ne0;
        if (t->type != type || t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::runtime_error(format("DeepSeek-V4.1 ISTA engram: %s is %s [%" PRId64 ", %" PRId64 "], expected %s [%" PRId64 ", %" PRId64 "]",
                    name, ggml_type_name(t->type), t->ne[0], t->ne[1], ggml_type_name(type), ne0, ne1));
        }
        return ml.consume_weight_data(name);
    };
    const auto as_u64 = [](const std::vector<uint8_t> & d) {
        std::vector<int64_t> v(d.size()/sizeof(int64_t));
        memcpy(v.data(), d.data(), v.size()*sizeof(int64_t));
        std::vector<uint64_t> r(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] < 0) {
                throw std::runtime_error("DeepSeek-V4.1 ISTA engram: negative hash constant");
            }
            r[i] = (uint64_t) v[i];
        }
        return r;
    };

    {
        const auto d = read("engram_token_map.weight", GGML_TYPE_I32, -1, 1); // checked against the vocab in load_arch_tensors
        engram_token_map.resize(d.size()/sizeof(int32_t));
        memcpy(engram_token_map.data(), d.data(), d.size());
    }
    engram_primes  = as_u64(read("engram_primes.weight",      GGML_TYPE_I64, n_cols,           n_e));
    engram_offsets = as_u64(read("engram_offsets.weight",     GGML_TYPE_I64, n_cols,           n_e));
    engram_mult    = as_u64(read("engram_multipliers.weight", GGML_TYPE_I64, engram_max_ngram, n_e));

    int32_t tm_max = -1;
    for (int32_t c : engram_token_map) {
        if (c < 0) {
            throw std::runtime_error("DeepSeek-V4.1 ISTA engram: negative token map entry");
        }
        tm_max = std::max(tm_max, c);
    }
    if ((uint32_t) tm_max + 1 != cvocab) {
        throw std::runtime_error(format("DeepSeek-V4.1 ISTA engram: token map covers %d ids, engram_compressed_vocab_size is %u", tm_max + 1, cvocab));
    }
    for (size_t e = 0; e < n_e; ++e) {
        // each layer's heads tile its own table: offsets are the running sum of the bucket counts, from 0
        uint64_t off = 0;
        for (size_t c = 0; c < n_cols; ++c) {
            if (engram_offsets[e*n_cols + c] != off) {
                throw std::runtime_error(format("DeepSeek-V4.1 ISTA engram layer %u: offset %zu is %" PRIu64 ", expected %" PRIu64 " (offsets must be per layer)",
                        e_layers[e], c, engram_offsets[e*n_cols + c], off));
            }
            off += engram_primes[e*n_cols + c];
        }
        if (off != rows[e]) {
            throw std::runtime_error(format("DeepSeek-V4.1 ISTA engram layer %u: buckets sum to %" PRIu64 ", num_embeddings is %u", e_layers[e], off, rows[e]));
        }
        for (size_t o = 0; o < engram_max_ngram; ++o) {
            // the hash is token*mult in int64 (engram.py): must not overflow
            const uint64_t m = engram_mult[e*engram_max_ngram + o];
            if (m == 0 || (m & 1) == 0 || m > (uint64_t) INT64_MAX/(uint64_t) cvocab) {
                throw std::runtime_error(format("DeepSeek-V4.1 ISTA engram layer %u: bad multiplier %" PRIu64, e_layers[e], m));
            }
        }
    }

    engram_rows.assign(rows.begin(), rows.end());
    engram_per_layer = true;
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_embd_head     = hparams.n_embd_head_k();
    const int64_t o_groups        = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult         = hparams.dsv4_hc_mult;
    const int64_t hc_dim          = hc_mult * n_embd;
    const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;
    const int64_t n_idx_embd      = hparams.indexer_head_size;
    const int64_t n_idx_head      = hparams.indexer_n_head;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    if (!engram_layers.empty() && engram_token_map.size() != (size_t) n_vocab) {
        // a token map shorter than the vocab would hash the tail as dead tokens
        throw std::runtime_error(format("DeepSeek-V4.1 engram token map has %zu entries, the vocab has %" PRId64, engram_token_map.size(), n_vocab));
    }

    if (engram_per_layer) {
        // ISTA: one lazily read table per Engram layer, from the sidecar GGUF; never in device memory
        const size_t n_cols = engram_n_cols();
        engram_embd.resize(engram_layers.size());
        for (size_t e = 0; e < engram_layers.size(); ++e) {
            const int il = engram_layers[e];
            const std::string name = tn(LLM_TENSOR_ENGRAM_EMBD, "weight", il).str();
            if (!ml.get_weight(name.c_str())) {
                throw std::runtime_error(format("DeepSeek-V4.1: Engram table %s not found: the sidecar named by %s.engram.file "
                        "(or LLAMA_DSV41_ENGRAM_FILE) must be loaded", name.c_str(), ml.get_arch_name().c_str()));
            }
            const int64_t need = (int64_t) (engram_offsets[e*n_cols + n_cols - 1] + engram_primes[e*n_cols + n_cols - 1]);
            if ((int64_t) engram_rows[e] != need) {
                throw std::runtime_error(format("DeepSeek-V4.1: Engram layer %d needs %" PRId64 " rows, num_embeddings says %" PRIu64, il, need, engram_rows[e]));
            }
            engram_embd[e] = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", il), {(int64_t) engram_head_dim, need}, TENSOR_READ_LAZY);
            if (ml.lazy.has(engram_embd[e])) {
                can_prefetch.insert(engram_embd[e]);
            }
        }
    }

    // all Engram tables in one lazily read tensor, like the qwen4exp n-gram table
    if (!engram_layers.empty() && !engram_per_layer) {
        int64_t n_rows = 0;
        for (size_t c = 0; c < engram_offsets.size(); ++c) {
            n_rows = std::max(n_rows, (int64_t) (engram_offsets[c] + engram_primes[c]));
        }
        const std::string name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * w = ml.get_weight(name.c_str())) {
            if (w->tensor->ne[1] < n_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the Engram heads (%" PRId64 ")",
                                                name.c_str(), w->tensor->ne[1], n_rows));
            }
            n_rows = w->tensor->ne[1];
        }
        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"), {(int64_t) engram_head_dim, n_rows}, TENSOR_READ_LAZY);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        if (hparams.dsv41_kv_src[i] == i) {
            layer.attn_comp_wkv  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,  "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", i), {n_embd_head}, 0);
            if (hparams.dsv4_compress_ratios[i] > 1) {
                layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, 0);
            }
            layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i), {n_embd_head, n_idx_embd}, 0);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {n_idx_embd}, 0);
        }
        if (hparams.dsv41_topk_src[i] == i) {
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_idx_head * n_idx_embd}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_idx_head}, 0);
        }

        if (engram_index(i) >= 0) {
            const int64_t n_kv_in = (int64_t) engram_n_cols() * engram_head_dim;
            if (engram_per_layer) {
                // ISTA: engram_wkv is engram_kv; the gate stays as its factors q and k (folded in build_engram)
                layer.engram_kv = create_tensor(tn(LLM_TENSOR_ENGRAM_WKV, "weight", i), {n_kv_in, n_embd * (hc_mult + 1)}, 0);
                layer.engram_q  = create_tensor(tn(LLM_TENSOR_ENGRAM_Q,   "weight", i), {n_embd, hc_mult}, 0);
                layer.engram_k  = create_tensor(tn(LLM_TENSOR_ENGRAM_K,   "weight", i), {n_embd, hc_mult}, 0);
            } else {
                layer.engram_kv   = create_tensor(tn(LLM_TENSOR_ENGRAM_KV,   "weight", i), {n_kv_in, n_embd * (hc_mult + 1)}, 0);
                layer.engram_gate = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE, "weight", i), {n_embd, hc_mult}, 0);
            }
        }

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias", i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd}, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("DeepSeek-V4.1 has no MTP graph: its DSpark draft is a separate DFLASH GGUF "
                "(convert_hf_to_gguf.py --dspark), used with --spec-type draft-dspark");
    }
    return std::make_unique<graph>(*this, params);
}

// Engram n-gram hash (engram.py NgramHashState.forward)
int llama_model_deepseek41::engram_index(int il) const {
    const auto it = std::find(engram_layers.begin(), engram_layers.end(), il);
    return it == engram_layers.end() ? -1 : (int) (it - engram_layers.begin());
}

void llama_model_deepseek41::engram_hash(int e, const llama_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) const {
    const int64_t n_gram = engram_max_ngram;
    const int64_t n_prev = n_gram - 1;
    const int64_t n_head = engram_n_head;
    const int64_t n_cols = engram_n_cols();

    const uint64_t * mult   = engram_mult.data()    + (size_t) e*n_gram;
    const uint64_t * primes = engram_primes.data()  + (size_t) e*n_cols;
    const int64_t    pad    = engram_token_map[engram_pad_id];
    const uint64_t * offs   = engram_offsets.data() + (size_t) e*n_cols;

    const auto cid = [&](llama_token t) -> int64_t {
        return t < 0 || t >= (llama_token) engram_token_map.size() ? DSV41_ENGRAM_DEAD : engram_token_map[t];
    };

    out.resize(ubatch.n_tokens*n_cols);
    std::vector<uint64_t> toks(n_gram);
    for (int64_t i = 0; i < ubatch.n_tokens; ++i) {
        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);
        int64_t c = is_embd ? DSV41_ENGRAM_DEAD : cid(ubatch.token[i]);
        bool blocked = c == DSV41_ENGRAM_DEAD;
        toks[0] = blocked ? pad : c;
        for (int64_t s = 1; s < n_gram; ++s) {
            c = ubatch.pos[i] - s < 0 ? DSV41_ENGRAM_DEAD : cid(prev[i*n_prev + (n_prev - s)]);
            blocked = blocked || c == DSV41_ENGRAM_DEAD;
            toks[s] = blocked ? pad : c;
        }

        uint64_t rolling = toks[0]*mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o]*mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1)*n_head + h;
                out[i*n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
            }
        }
    }
}

// LLAMA_DSV41_DUMP_ENGRAM=path: append each token's hashed row ids (relative to its layer's table) per Engram layer,
// one line per (token, layer): "seq pos token layer id_0 .. id_{cols-1}", for parity tests against a reference hash
static void llama_dsv41_dump_engram(const llama_model_deepseek41 & model, const llama_ubatch & ubatch, const std::vector<std::vector<int32_t>> & ids) {
    static const char * const path = getenv("LLAMA_DSV41_DUMP_ENGRAM");
    if (!path || !*path) {
        return;
    }
    // several contexts (server slots, a speculative target) may call this concurrently: all state is under the lock
    static std::mutex mtx;
    static FILE * f = nullptr;
    static bool failed = false;
    std::lock_guard<std::mutex> lock(mtx);
    if (failed) {
        return;
    }
    if (!f) {
        f = fopen(path, "a");
        if (!f) {
            LLAMA_LOG_WARN("%s: cannot open %s\n", __func__, path);
            failed = true;
            return;
        }
    }
    const int64_t n_cols = model.engram_n_cols();
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const int32_t tok = ubatch.token ? ubatch.token[i] : -1;
        const int32_t seq = ubatch.n_seq_id && ubatch.n_seq_id[i] > 0 ? ubatch.seq_id[i][0] : 0;
        for (size_t e = 0; e < ids.size(); ++e) {
            fprintf(f, "%d %d %d %d", seq, (int) ubatch.pos[i], tok, model.engram_layers[e]);
            for (int64_t c = 0; c < n_cols; ++c) {
                fprintf(f, " %d", ids[e][i*n_cols + c]);
            }
            fputc('\n', f);
        }
    }
    fflush(f);
}

// cand_pin: +1e30 (the reference uses +inf) on the block holding each query's newest compressed row, so a partly
// filled block always survives; a finite pin keeps fully masked blocks at -inf instead of NaN
class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
    llm_graph_input_dsv41(const llama_model_deepseek41 & model, const llama_kv_cache_dsv4_context * mctx) :
        model(model), mctx(mctx) {}

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::vector<ggml_tensor *> engram_ids;
    std::vector<const ggml_tensor *> engram_tab;
    ggml_tensor * engram_keep = nullptr;    // 0 for embedding (image) tokens; only in mixed ubatches
    ggml_tensor * cand_pin = nullptr;
    bool cand_csa = false;
    uint32_t n_ubatch = 0; // cparams.n_ubatch: length of the next ubatch predicted from a prompt hint

    const llama_model_deepseek41 & model;
    const llama_kv_cache_dsv4_context * mctx;

private:
    const llama_kv_cache_dsv4_context::comp_plan & cand_plan(const llama_ubatch & ubatch) const {
        return cand_csa ? mctx->get_csa_plan(ubatch) : mctx->get_hca_plan(ubatch);
    }
};

namespace {
struct engram_stats_t {
    bool on = getenv("LLAMA_DSV41_ENGRAM_STATS") != nullptr;
    uint64_t calls = 0, tok1_calls = 0, hash_us = 0, pref_us = 0, gather_us = 0, set_us = 0;
    ~engram_stats_t() {
        if (on && tok1_calls > 0) {
            fprintf(stderr, "engram stats: single-token set_input calls = %llu, hash %.1f us, readahead %.1f us, gather %.1f us, upload %.1f us (per call)\n",
                (unsigned long long) tok1_calls, (double) hash_us/tok1_calls, (double) pref_us/tok1_calls, (double) gather_us/tok1_calls, (double) set_us/tok1_calls);
        }
    }
} g_eg;
inline uint64_t eg_now() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// LLAMA_DSV41_ENGRAM_PREAD=N (multi-token ubatches): the Engram rows are pread from the GGUF file by N threads straight
// into a row buffer and converted like llm_graph_host_get_rows (same bytes, same conversion: lossless), instead of one
// madvise(WILLNEED) per row range on the main thread followed by page faults in an 8-thread gather. 0 = off (default).
int engram_pread_threads() {
    static const int n = [] { const char * v = getenv("LLAMA_DSV41_ENGRAM_PREAD"); return v ? std::max(0, atoi(v)) : 0; }();
    return n;
}

bool engram_pread_rows(const ggml_tensor * tab, const int32_t * ids, int64_t n_ids, float * dst, int n_thr) {
    int fd = -1;
    uint64_t base = 0;
    if (!llama_exps_prefetch::lookup(tab->data, ggml_nbytes(tab), &fd, &base) || fd < 0) {
        return false;
    }
    const int64_t nc = tab->ne[0];
    const size_t row_bytes = ggml_row_size(tab->type, nc);
    const ggml_to_float_t to_float = tab->type == GGML_TYPE_F32 ? nullptr : ggml_get_type_traits(tab->type)->to_float;
    std::atomic<bool> ok{true};
    auto run = [&](int64_t i0, int64_t i1) {
        std::vector<char> buf(row_bytes);
        for (int64_t i = i0; i < i1 && ok.load(std::memory_order_relaxed); ++i) {
            const int64_t r = ids[i];
            GGML_ASSERT(r >= 0 && r < tab->ne[1]);
            const uint64_t off = base + (uint64_t) r*tab->nb[1];
            size_t got = 0;
            while (got < row_bytes) {
                const ssize_t n = pread(fd, buf.data() + got, row_bytes - got, (off_t) (off + got));
                if (n <= 0) {
                    ok = false;
                    return;
                }
                got += (size_t) n;
            }
            if (to_float) {
                to_float(buf.data(), dst + i*nc, nc);
            } else {
                memcpy(dst + i*nc, buf.data(), nc*sizeof(float));
            }
        }
    };
    n_thr = (int) std::max<int64_t>(1, std::min<int64_t>(n_thr, n_ids/64 + 1));
    std::vector<std::thread> workers;
    const int64_t per = (n_ids + n_thr - 1)/n_thr;
    for (int t = 1; t < n_thr; ++t) {
        const int64_t i0 = std::min<int64_t>(n_ids, t*per);
        workers.emplace_back(run, i0, std::min<int64_t>(n_ids, i0 + per));
    }
    run(0, std::min<int64_t>(n_ids, per));
    for (auto & w : workers) {
        w.join();
    }
    return ok.load();
}

// LLAMA_DSV41_ENGRAM_AHEAD=1 (multi-token ubatches, host-gathered Engram rows): when the tokens of the next ubatch are
// known - the next ubatch of this batch, or the prompt hint (llama_set_prompt_tokens) for the first ubatch of the next
// batch - a background thread hashes them and gathers their rows into a staging buffer while the current ubatch
// computes. set_input still hashes every ubatch from the cache's token history and takes a token's rows from the
// staging buffer only if all of its row ids match the staged ones (same ids, same tables, same conversion: bitwise
// identical rows); the other tokens are gathered as before. 0 = off (default).
// LLAMA_DSV41_ENGRAM_AHEAD_THREADS: pread threads of the background gather (default LLAMA_DSV41_ENGRAM_PREAD, else 16)
bool engram_ahead_on() {
    static const bool on = [] { const char * v = getenv("LLAMA_DSV41_ENGRAM_AHEAD"); return v && atoi(v) != 0; }();
    return on;
}

int engram_ahead_threads() {
    static const int n = [] {
        const char * v = getenv("LLAMA_DSV41_ENGRAM_AHEAD_THREADS");
        const int n = v ? atoi(v) : engram_pread_threads();
        return n > 0 ? n : 16;
    }();
    return n;
}

// staged rows of positions pos0 .. pos0 + n - 1 of seq; owned by the KV cache (joined before the model is freed)
struct engram_ahead_t : llama_kv_cache_dsv4::aux_state {
    std::thread th;
    llama_seq_id seq = -1;
    llama_pos pos0 = 0;
    int64_t n = 0;
    std::vector<llama_token> win;           // tokens of positions pos0 - n_prev .. pos0 + n - 1 the job hashed
    std::vector<std::vector<int32_t>> ids;  // [Engram layer] n*n_cols staged row ids, empty = layer not staged
    std::vector<std::vector<float>>   rows; // [Engram layer] n*n_cols*head_dim
    double job_ms = 0;

    ~engram_ahead_t() override { wait(); }
    void wait() {
        if (th.joinable()) {
            th.join();
        }
    }
    void release() {
        wait();
        n = 0;
        win.clear();
        std::vector<std::vector<int32_t>>().swap(ids);
        std::vector<std::vector<float>>().swap(rows);
    }
    bool layer_staged(size_t e) const { return n > 0 && e < ids.size() && (int64_t) ids[e].size() > 0; }
};

engram_ahead_t * engram_ahead_get(const llama_kv_cache_dsv4_context * mctx, bool create) {
    auto * kv = mctx ? mctx->get_kv() : nullptr;
    if (!kv) {
        return nullptr;
    }
    auto & aux = kv->get_aux();
    if (!aux && create) {
        aux = std::make_unique<engram_ahead_t>();
    }
    return dynamic_cast<engram_ahead_t *>(aux.get());
}

// a ubatch of text tokens of one sequence at consecutive positions: seq, else -1
llama_seq_id engram_ahead_contig(const llama_ubatch & ub) {
    if (!ub.token || ub.embd || ub.is_mixed() || ub.n_tokens == 0 || !ub.pos || !ub.n_seq_id || !ub.seq_id) {
        return -1;
    }
    const llama_seq_id seq = ub.seq_id[0][0];
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        if (ub.n_seq_id[i] != 1 || ub.seq_id[i][0] != seq || ub.pos[i] != ub.pos[0] + (llama_pos) i) {
            return -1;
        }
    }
    return seq;
}

// start the background hash + gather of win (n_prev history tokens, then the n tokens at p0 ..) for the given tables
void engram_ahead_launch(engram_ahead_t & ah, const llama_model_deepseek41 & model, std::vector<const ggml_tensor *> tabs,
        llama_seq_id seq, llama_pos p0, std::vector<llama_token> win) {
    ah.wait();
    const int64_t n_prev = model.engram_max_ngram - 1;
    ah.seq  = seq;
    ah.pos0 = p0;
    ah.n    = (int64_t) win.size() - n_prev;
    ah.win  = std::move(win);
    ah.ids.resize(tabs.size());
    ah.rows.resize(tabs.size());
    for (size_t e = 0; e < tabs.size(); ++e) {
        ah.ids[e].clear(); // filled by the job; empty = not staged
    }
    ah.th = std::thread([&ah, &model, tabs = std::move(tabs), n_prev] {
        const uint64_t t0 = eg_now();
        const int64_t n = ah.n;
        std::vector<llama_token> tok(ah.win.begin() + n_prev, ah.win.end());
        std::vector<llama_pos>   pos(n);
        std::vector<llama_token> prev(n*n_prev);
        for (int64_t i = 0; i < n; ++i) {
            pos[i] = ah.pos0 + (llama_pos) i;
            for (int64_t j = 0; j < n_prev; ++j) {
                prev[i*n_prev + j] = ah.win[i + j]; // position pos[i] - (n_prev - j), as get_prev_tokens
            }
        }
        llama_ubatch ub = {};
        ub.n_tokens = (uint32_t) n;
        ub.token    = tok.data();
        ub.pos      = pos.data();
        std::vector<int32_t> ids;
        for (size_t e = 0; e < tabs.size(); ++e) {
            if (!tabs[e]) {
                continue;
            }
            model.engram_hash((int) e, ub, prev, ids);
            ah.rows[e].resize(ids.size()*tabs[e]->ne[0]);
            if (!engram_pread_rows(tabs[e], ids.data(), (int64_t) ids.size(), ah.rows[e].data(), engram_ahead_threads())) {
                llm_graph_host_get_rows(tabs[e], ids.data(), (int64_t) ids.size(), ah.rows[e].data());
            }
            ah.ids[e].swap(ids);
        }
        ah.job_ms = (eg_now() - t0)/1000.0;
    });
}
} // namespace

void llm_graph_input_dsv41::set_input(const llama_ubatch * ubatch) {
    if (!engram_ids.empty()) {
        // LLAMA_DSV41_ENGRAM_STATS: single-token calls are averaged at exit; multi-token ubatches (prefill) log one line each
        const bool eg_t = g_eg.on && ubatch->n_tokens == 1;
        const bool eg_m = g_eg.on && ubatch->n_tokens > 1;
        uint64_t eg0 = (eg_t || eg_m) ? eg_now() : 0;
        uint64_t eg_gather = 0, eg_set = 0;
        size_t eg_rows = 0;
        // a token shared by several sequences has the same history in each, so its first sequence resolves it
        std::vector<llama_token> prev;
        mctx->get_raw()->get_prev_tokens(*ubatch, model.engram_max_ngram - 1, prev);
        // hash all Engram layers first so a lazily read table gets one batched readahead before the first gather
        std::vector<std::vector<int32_t>> ids(engram_ids.size());
        std::vector<int32_t> all;
        const bool eg_pread = engram_pread_threads() > 0 && ubatch->n_tokens > 1;
        for (size_t e = 0; e < engram_ids.size(); ++e) {
            model.engram_hash((int) e, *ubatch, prev, ids[e]);
            if (eg_pread) {
                // read below by the pread gather, no readahead
            } else if (model.engram_per_layer) {
                if (model.can_prefetch.count(model.engram_table((int) e))) {
                    llama_prefetch_rows(model.engram_table((int) e), ids[e].data(), ids[e].size());
                }
            } else {
                all.insert(all.end(), ids[e].begin(), ids[e].end());
            }
        }
        if (!eg_pread && !model.engram_per_layer && model.can_prefetch.count(model.per_layer_tok_embd)) {
            llama_prefetch_rows(model.per_layer_tok_embd, all.data(), all.size());
        }
        llama_dsv41_dump_engram(model, *ubatch, ids);
        uint64_t eg1 = (eg_t || eg_m) ? eg_now() : 0;
        if (eg_t) { g_eg.pref_us += eg1 - eg0; g_eg.tok1_calls++; }

        // LLAMA_DSV41_ENGRAM_AHEAD: rows staged by the background job of the previous ubatch
        engram_ahead_t * ah = nullptr;
        llama_seq_id ah_seq = -1;
        uint64_t eg_wait = 0;
        size_t eg_staged = 0;
        double eg_job = 0;
        if (engram_ahead_on() && std::any_of(engram_tab.begin(), engram_tab.end(), [](const ggml_tensor * t) { return t != nullptr; })) {
            ah = engram_ahead_get(mctx, ubatch->n_tokens > 1);
            if (ah && ubatch->n_tokens == 1) {
                if (ah->n > 0 || ah->th.joinable()) {
                    ah->release(); // decode after the prompt: drop what was staged in vain
                }
                ah = nullptr;
            }
            if (ah) {
                const uint64_t w0 = eg_now();
                ah->wait();
                eg_wait = eg_now() - w0;
                eg_job  = ah->n > 0 ? ah->job_ms : 0.0;
                ah_seq  = engram_ahead_contig(*ubatch);
            }
        }
        const int64_t n_cols = model.engram_n_cols();

        std::vector<float> rows;
        for (size_t e = 0; e < engram_ids.size(); ++e) {
            if (engram_ids[e]->buffer) {
                if (engram_tab[e]) {
                    // host gather: same rows and conversion as the CPU GET_ROWS
                    const int64_t hd = engram_tab[e]->ne[0];
                    const int64_t nt = ubatch->n_tokens;
                    const uint64_t g0 = (eg_t || eg_m) ? eg_now() : 0;
                    // tokens whose staged row ids all match: their rows are taken from the staging buffer
                    std::vector<int64_t> miss;
                    int64_t off = 0;
                    if (ah && ah_seq >= 0 && ah_seq == ah->seq && ah->layer_staged(e)) {
                        off = ubatch->pos[0] - ah->pos0;
                        for (int64_t i = 0; i < nt; ++i) {
                            const int64_t s = off + i;
                            if (s < 0 || s >= ah->n || memcmp(ids[e].data() + i*n_cols, ah->ids[e].data() + s*n_cols, n_cols*sizeof(int32_t)) != 0) {
                                miss.push_back(i);
                            }
                        }
                    } else {
                        miss.resize(nt);
                        for (int64_t i = 0; i < nt; ++i) {
                            miss[i] = i;
                        }
                    }
                    eg_staged += nt - miss.size();
                    const float * src = nullptr;
                    if (miss.empty()) {
                        src = ah->rows[e].data() + off*n_cols*hd;
                    } else if ((int64_t) miss.size() == nt) {
                        rows.resize(ids[e].size()*hd);
                        if (!eg_pread || !engram_pread_rows(engram_tab[e], ids[e].data(), (int64_t) ids[e].size(), rows.data(), engram_pread_threads())) {
                            llm_graph_host_get_rows(engram_tab[e], ids[e].data(), (int64_t) ids[e].size(), rows.data());
                        }
                        src = rows.data();
                    } else {
                        // some tokens staged: gather the others into their slots
                        rows.resize(ids[e].size()*hd);
                        std::vector<int32_t> mids(miss.size()*n_cols);
                        for (size_t k = 0; k < miss.size(); ++k) {
                            memcpy(mids.data() + k*n_cols, ids[e].data() + miss[k]*n_cols, n_cols*sizeof(int32_t));
                        }
                        std::vector<float> mrows(mids.size()*hd);
                        if (!eg_pread || !engram_pread_rows(engram_tab[e], mids.data(), (int64_t) mids.size(), mrows.data(), engram_pread_threads())) {
                            llm_graph_host_get_rows(engram_tab[e], mids.data(), (int64_t) mids.size(), mrows.data());
                        }
                        const size_t tok_floats = (size_t) (n_cols*hd);
                        size_t k = 0;
                        for (int64_t i = 0; i < nt; ++i) {
                            if (k < miss.size() && miss[k] == i) {
                                memcpy(rows.data() + i*tok_floats, mrows.data() + k*tok_floats, tok_floats*sizeof(float));
                                ++k;
                            } else {
                                memcpy(rows.data() + i*tok_floats, ah->rows[e].data() + (off + i)*tok_floats, tok_floats*sizeof(float));
                            }
                        }
                        src = rows.data();
                    }
                    const uint64_t g1 = (eg_t || eg_m) ? eg_now() : 0;
                    ggml_backend_tensor_set(engram_ids[e], src, 0, ids[e].size()*hd*sizeof(float));
                    if (eg_t) { g_eg.gather_us += g1 - g0; g_eg.set_us += eg_now() - g1; }
                    if (eg_m) { eg_gather += g1 - g0; eg_set += eg_now() - g1; eg_rows += ids[e].size(); }
                } else {
                    ggml_backend_tensor_set(engram_ids[e], ids[e].data(), 0, ids[e].size()*sizeof(int32_t));
                }
            }
        }

        // LLAMA_DSV41_ENGRAM_AHEAD: start the gather of the next ubatch (this batch's next one, else the prompt hint)
        if (ah) {
            const int64_t n_prev = model.engram_max_ngram - 1;
            const llama_pos q0 = ah_seq >= 0 ? ubatch->pos[0] + (llama_pos) ubatch->n_tokens : 0;
            llama_pos hp0 = 0;
            const std::vector<llama_token> * hint = ah_seq >= 0 && mctx->get_kv() ? mctx->get_kv()->get_prompt_tokens(ah_seq, hp0) : nullptr;
            // token at position p of ah_seq: this ubatch, the hint, else unknown (the first rows then just miss)
            const auto tok_at = [&](llama_pos p) -> llama_token {
                if (p >= ubatch->pos[0] && p < q0) {
                    return ubatch->token[p - ubatch->pos[0]];
                }
                if (hint && p >= hp0 && p < hp0 + (llama_pos) hint->size()) {
                    return (*hint)[p - hp0];
                }
                return LLAMA_TOKEN_NULL;
            };
            std::vector<llama_token> win;
            if (ah_seq >= 0) {
                const llama_ubatch * nx = mctx->peek_next_ubatch();
                if (nx && engram_ahead_contig(*nx) == ah_seq && nx->pos[0] == q0) {
                    for (int64_t j = n_prev; j > 0; --j) {
                        win.push_back(tok_at(q0 - (llama_pos) j));
                    }
                    win.insert(win.end(), nx->token, nx->token + nx->n_tokens);
                } else if (!nx && hint && q0 >= hp0 && q0 < hp0 + (llama_pos) hint->size()) {
                    const int64_t n = std::min<int64_t>(n_ubatch > 0 ? n_ubatch : ubatch->n_tokens, hp0 + (int64_t) hint->size() - q0);
                    for (int64_t j = n_prev; j > 0; --j) {
                        win.push_back(tok_at(q0 - (llama_pos) j));
                    }
                    win.insert(win.end(), hint->begin() + (q0 - hp0), hint->begin() + (q0 - hp0) + n);
                }
            }
            const int64_t n_next = (int64_t) win.size() - n_prev;
            bool covered = false;
            if (n_next > 0 && ah->n > 0 && ah->seq == ah_seq && q0 >= ah->pos0 && q0 + n_next <= ah->pos0 + ah->n) {
                // the staged range already holds these tokens (a prompt-skip head split of a hinted range)
                covered = std::equal(win.begin(), win.end(), ah->win.begin() + (q0 - ah->pos0));
                for (size_t e = 0; covered && e < engram_tab.size(); ++e) {
                    covered = !engram_tab[e] || ah->layer_staged(e);
                }
            }
            if (n_next > 0 && !covered) {
                engram_ahead_launch(*ah, model, engram_tab, ah_seq, q0, std::move(win));
            } else if (n_next <= 0) {
                ah->release();
            }
        }

        if (eg_m) {
            fprintf(stderr, "engram stats: ubatch %u tokens: rows = %zu, hash+readahead %.2f ms, gather %.2f ms, upload %.2f ms, total %.2f ms",
                ubatch->n_tokens, eg_rows, (eg1 - eg0)/1000.0, eg_gather/1000.0, eg_set/1000.0, (eg_now() - eg0)/1000.0);
            if (ah) {
                fprintf(stderr, ", ahead: staged %zu/%zu token-layers, wait %.2f ms, job %.2f ms, next %s",
                    eg_staged, (size_t) ubatch->n_tokens*engram_tab.size(), eg_wait/1000.0, eg_job, ah->th.joinable() ? "launched" : ah->n > 0 ? "staged" : "none");
            }
            fputc('\n', stderr);
        }
    }

    // embedding (image) tokens are hashed DEAD and get no Engram contribution (gate 0)
    if (engram_keep && engram_keep->buffer) {
        std::vector<float> keep(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            keep[i] = is_embd ? 0.0f : 1.0f;
        }
        ggml_backend_tensor_set(engram_keep, keep.data(), 0, keep.size()*sizeof(float));
    }

    if (cand_pin && cand_pin->buffer) {
        const auto & plan = cand_plan(*ubatch);
        const int64_t n_blk = cand_pin->ne[0];
        const int64_t bs    = model.hparams.dsv41_cand_block_size;
        GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) ubatch->n_tokens);
        std::vector<float> pin(n_blk*ubatch->n_tokens, 0.0f);
        for (int64_t i = 0; i < (int64_t) ubatch->n_tokens; ++i) {
            const int64_t nv = plan.n_visible[i];
            if (nv > 0) {
                GGML_ASSERT((nv - 1)/bs < n_blk);
                pin[i*n_blk + (nv - 1)/bs] = 1e30f;
            }
        }
        ggml_backend_tensor_set(cand_pin, pin.data(), 0, pin.size()*sizeof(float));
    }
}

bool llm_graph_input_dsv41::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    bool res = true;
    for (auto * t : engram_ids) {
        res &= t->type == GGML_TYPE_F32 ? t->ne[1] == (int64_t) params.ubatch.n_tokens
                                        : t->ne[0] == (int64_t) model.engram_n_cols()*params.ubatch.n_tokens;
    }
    res &= (engram_keep != nullptr) == (!engram_ids.empty() && params.ubatch.embd != nullptr);
    if (engram_keep) {
        res &= engram_keep->ne[2] == (int64_t) params.ubatch.n_tokens;
    }
    if (cand_pin) {
        const auto & plan = cand_plan(params.ubatch);
        const int64_t bs = model.hparams.dsv41_cand_block_size;
        res &= plan.n_kv > 0 && cand_pin->ne[0] == (plan.n_kv + bs - 1)/bs;
        res &= cand_pin->ne[1]*cand_pin->ne[3] == (int64_t) params.ubatch.n_tokens;
    }
    return res;
}

ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model_deepseek41 & model,
        ggml_tensor * x,
        ggml_tensor * ids,
        ggml_tensor * keep,
        int il) const {
    const auto & layer = model.layers[il];
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * rows = ids;
    if (ids->type != GGML_TYPE_F32) {
        rows = ggml_get_rows(ctx0, const_cast<ggml_tensor *>(model.engram_table(model.engram_index(il))), ids);
        rows = ggml_reshape_2d(ctx0, rows, model.engram_head_dim*model.engram_n_cols(), nt);
    }
    GGML_ASSERT(rows->ne[0] == (int64_t) (model.engram_head_dim*model.engram_n_cols()) && rows->ne[1] == nt);
    cb(rows, "engram_rows", il);

    ggml_tensor * kv = build_lora_mm(layer.engram_kv, rows);
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, hc, nt, ggml_row_size(kv->type, n_embd), kv->nb[1], 0));
    ggml_tensor * value = ggml_cont(ctx0, ggml_view_3d(ctx0, kv, n_embd, 1,  nt, ggml_row_size(kv->type, n_embd), kv->nb[1],
            ggml_row_size(kv->type, hc*n_embd)));

    ggml_tensor * egate = nullptr;
    if (layer.engram_gate) {
        // quantized copy has no binary-op kernel
        egate = ggml_is_quantized(layer.engram_gate->type) ? ggml_cast(ctx0, layer.engram_gate, GGML_TYPE_F32) : layer.engram_gate;
    } else {
        // ISTA: gate = q_weight * k_weight, as the converter folds it (a BF16 x BF16 product is exact in F32)
        GGML_ASSERT(layer.engram_q && layer.engram_k);
        const auto f32 = [&](ggml_tensor * t) { return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx0, t, GGML_TYPE_F32); };
        egate = ggml_mul(ctx0, f32(layer.engram_q), f32(layer.engram_k));
        cb(egate, "engram_gate_w", il);
    }
    ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0, ggml_rms_norm(ctx0, x, norm_rms_eps), ggml_rms_norm(ctx0, key, norm_rms_eps)), egate);
    dot = ggml_scale(ctx0, ggml_sum_rows(ctx0, dot), 1.0f/sqrtf((float) n_embd));
    cb(dot, "engram_dot", il);

    // sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); sign(0) = +1 as in copysign
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * sign = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_neg(ctx0, dot)), -2.0f, 1.0f);
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, sign));
    if (keep) {
        gate = ggml_mul(ctx0, gate, keep);
    }
    cb(gate, "engram_gate", il);

    return ggml_add(ctx0, x, ggml_mul(ctx0, ggml_repeat_4d(ctx0, value, n_embd, hc, nt, 1), gate));
}

ggml_tensor * llama_model_deepseek41::graph::build_attention41(
        const llama_model_deepseek41 & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * cand_pin,
        int il,
        bool kv_only) {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t n_idx_embd       = hparams.indexer_head_size;
    const int64_t n_idx_head       = hparams.indexer_n_head;
    const int64_t nt               = cur->ne[1];
    const float   kq_scale         = 1.0f/sqrtf(float(n_embd_head));
    const int64_t ratio            = hparams.dsv4_compress_ratios[il];

    const bool    comp_rope     = ratio != 0;
    const float   freq_base_l   = comp_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float   freq_scale_l  = comp_rope ? freq_scale : 1.0f;
    const float   ext_factor_l  = comp_rope ? ext_factor : 0.0f;
    const float   attn_factor_l = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float   beta_fast_l   = comp_rope ? beta_fast : 0.0f;
    const float   beta_slow_l   = comp_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l  = comp_rope ? n_ctx_orig : 0;

    const auto rope = [&](ggml_tensor * t, ggml_tensor * pos, int64_t n_nope) {
        t = ggml_rope_ext(ctx0, t, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        return ggml_rope_set_offset(t, n_nope);
    };

    ggml_tensor * qr = build_norm(build_lora_mm(layer.wq_a, cur), layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = ggml_reshape_3d(ctx0, build_lora_mm(layer.wq_b, qr), n_embd_head, n_head, nt);
    q = rope(q, inp_pos, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * kv = build_norm(build_lora_mm(layer.wkv, cur), layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = rope(ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt), inp_pos, n_embd_head_nope);
    cb(kv, "kv", il);

    const auto & ci         = ratio == 2 ? inp_dsv4->get_csa()             : inp_dsv4->get_hca();
    const auto * comp_ctx   = ratio == 2 ? inp_dsv4->mctx->get_csa()       : inp_dsv4->mctx->get_hca();
    const auto * comp_state = ratio == 2 ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();

    const int32_t rd = ratio != 0 ? llama_dsv41_kv_read_layer(model, il, cparams.offload_kqv) : -1;

    if (ratio != 0) {
        GGML_ASSERT(hparams.dsv41_kv_src[il] >= 0 && (int64_t) comp_state->get_ratio() == ratio);

        if (hparams.dsv41_kv_src[il] == il && ci.state_pos) {
            ggml_tensor * state_kv    = build_lora_mm(layer.attn_comp_wkv, cur);
            ggml_tensor * state_score = ratio > 1 ? build_lora_mm(layer.attn_comp_wgate, cur) : ggml_fill(ctx0, state_kv, 0.0f);

            ggml_tensor * restored_kv    = comp_state->get_kv_all(ctx0, il);
            ggml_tensor * restored_score = comp_state->get_score_all(ctx0, il);
            if (ci.state_restore_src_idxs && ci.state_restore_dst_idxs) {
                restored_kv    = comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, restored_kv,    ci.state_restore_src_idxs), ci.state_restore_dst_idxs, il);
                restored_score = comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, restored_score, ci.state_restore_src_idxs), ci.state_restore_dst_idxs, il);
            }

            if (ci.state_write_idxs) {
                const int64_t n_blocks = ci.state_write_pos->ne[0];
                const int64_t n_state  = comp_state->get_n_rows();
                GGML_ASSERT(ci.state_read_idxs->ne[0] == ratio*n_blocks);

                ggml_tensor * src_kv    = ggml_concat(ctx0, ggml_view_2d(ctx0, restored_kv,    restored_kv->ne[0],    n_state, restored_kv->nb[1],    0), state_kv,    1);
                ggml_tensor * src_score = ggml_concat(ctx0, ggml_view_2d(ctx0, restored_score, restored_score->ne[0], n_state, restored_score->nb[1], 0), state_score, 1);

                ggml_tensor * v = ggml_get_rows(ctx0, src_kv,    ci.state_read_idxs);
                ggml_tensor * s = ggml_get_rows(ctx0, src_score, ci.state_read_idxs);
                v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));
                s = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, s, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));

                ggml_tensor * comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, ggml_soft_max(ctx0, s)));
                comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
                comp = build_norm(comp, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
                cb(comp, "comp_latent", il);

                // index K from the RoPE-free latent; a latent stands for the first token of its group
                ggml_tensor * ik = build_norm(build_lora_mm(layer.indexer_attn_k, comp), layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
                ik = rope(ik, ci.state_write_pos, n_idx_embd - n_embd_head_rope);
                comp = rope(comp, ci.state_write_pos, n_embd_head_nope);

                ggml_tensor * row = ggml_concat(ctx0, comp, ik, 0);
                ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, row, ci.state_write_idxs, il));
                comp_row_of[il] = row;
            }

            if (ci.state_snapshot_src_idxs && ci.state_snapshot_dst_idxs) {
                ggml_tensor * snap_kv    = ggml_concat(ctx0, restored_kv,    state_kv,    1);
                ggml_tensor * snap_score = ggml_concat(ctx0, restored_score, state_score, 1);
                ggml_build_forward_expand(gf, comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, snap_kv,    ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
                ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, snap_score, ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
            }
            ggml_build_forward_expand(gf, comp_state->cpy_kv   (ctx0, ggml_get_rows(ctx0, state_kv,    ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
            ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0, ggml_get_rows(ctx0, state_score, ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
        }

        if (rd == il && hparams.dsv41_kv_src[il] != il && comp_row_of[hparams.dsv41_kv_src[il]]) {
            // first consumer on this device: append the source's new rows to the local mirror
            ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, comp_row_of[hparams.dsv41_kv_src[il]], ci.state_write_idxs, il));
        }
    }

    if (kv_only) {
        // long-prompt skip head: only the compressed rows / index keys of this layer are wanted. q, the window K and the
        // attention are not part of the graph (nothing in it reads them)
        return nullptr;
    }

    ggml_tensor * out = nullptr;
    if (ratio == 0 || !ci.kq_mask) {
        // no compressed row visible yet: window only
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
    } else {
        const int64_t n_comp = ci.kq_mask->ne[0];

        ggml_tensor * comp_all = comp_ctx->get_k(ctx0, rd);
        GGML_ASSERT(comp_all->ne[0] == n_embd_head + n_idx_embd && n_comp <= comp_all->ne[2]);
        const int64_t n_stream = comp_all->ne[3];

        ggml_tensor * top_k = topk_of[hparams.dsv41_topk_src[il]];
        if (hparams.dsv41_topk_src[il] == il) {
            ggml_tensor * ik = ggml_view_4d(ctx0, comp_all, n_idx_embd, 1, n_comp, n_stream,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], ggml_row_size(comp_all->type, n_embd_head));
            // the fused indexer reads the strided view in place, the unfused path needs it contiguous; quantized
            // caches are dequantized straight from the strided view (CUDA has no same-type copy of quantized views)
            const bool fused_idx = sparse && cparams.fused_lid && ci.kq_mask->type == GGML_TYPE_F16;
            if (ggml_is_quantized(ik->type)) {
                ik = ggml_cast(ctx0, ik, GGML_TYPE_F32);
            } else if (!fused_idx) {
                ik = ggml_cont(ctx0, ik);
            }

            ggml_tensor * iq = ggml_reshape_3d(ctx0, build_lora_mm(layer.indexer_attn_q_b, qr), n_idx_embd, n_idx_head, nt);
            iq = rope(iq, inp_pos, n_idx_embd - n_embd_head_rope);
            iq = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nt/n_stream, n_stream, iq->nb[1], iq->nb[2], iq->nb[2]*(nt/n_stream), 0);

            ggml_tensor * iw = ggml_scale(ctx0, build_lora_mm(layer.indexer_proj, cur), 1.0f/sqrtf(float(n_idx_embd*n_idx_head)));
            iw = ggml_view_4d(ctx0, iw, iw->ne[0], nt/n_stream, 1, n_stream, iw->nb[1], iw->nb[1]*(nt/n_stream), iw->nb[1]*(nt/n_stream), 0);

            // chunk queries so the [n_ih, tokens, n_comp] scores stay near 2^26 elements (at most 64 chunks); the fused
            // indexer has no such intermediate and runs as one chunk
            const int64_t n_tok = nt/n_stream;
            GGML_ASSERT(ci.kq_mask->ne[1] == n_tok);
            const int64_t chunk = fused_idx ? n_tok :
                std::max({ (n_tok + 63)/64, (INT64_C(1) << 26)/(n_idx_head*n_comp*n_stream), INT64_C(1) });

            const int32_t cand_src = hparams.dsv41_cand_src;
            const bool    pick_blk = cand_src == il && cand_pin;              // level one: pick the candidate blocks
            const bool    use_blk  = cand_src >= 0 && cand_src < il && cand_sel; // level two: score inside them
            const int64_t bs       = hparams.dsv41_cand_block_size;
            const int64_t n_blk    = pick_blk || use_blk ? (n_comp + bs - 1)/bs : 0;
            GGML_ASSERT(!pick_blk || cand_pin->ne[0] == n_blk);
            GGML_ASSERT(!use_blk  || cand_n_comp == n_comp);

            ggml_tensor * ik_p = fused_idx ? nullptr : ggml_permute(ctx0, ik, 0, 2, 1, 3);
            std::vector<ggml_tensor *> parts_top_k;
            std::vector<ggml_tensor *> parts_sel;
            for (int64_t t0 = 0; t0 < n_tok; t0 += chunk) {
                const int64_t nc = std::min(chunk, n_tok - t0);
                const auto rows = [&](ggml_tensor * t) {
                    return ggml_view_4d(ctx0, t, t->ne[0], nc, 1, n_stream, t->nb[1], t->nb[2], t->nb[3], t0*t->nb[1]);
                };
                ggml_tensor * score = nullptr;
                if (fused_idx) {
                    score = ggml_lightning_indexer(ctx0, iq, ik, iw, ci.kq_mask);
                    res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
                } else {
                    ggml_tensor * iqc = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nc, n_stream, iq->nb[1], iq->nb[2], iq->nb[3], t0*iq->nb[2]);

                    ggml_tensor * kq = ggml_mul_mat(ctx0, ik_p, ggml_permute(ctx0, iqc, 0, 2, 1, 3));
                    if (!ggml_is_quantized(ik->type)) {
                        ggml_prec_set_src(kq, GGML_PREC_F32, 1);
                    }
                    kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
                    score = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), rows(iw)));
                    ggml_tensor * mask  = rows(ci.kq_mask);
                    score = ggml_add(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3)),
                            mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32));
                }
                cb(score, "idx_score", il);

                if (pick_blk) {
                    // best position per block, pin the newest block, keep the top blocks
                    ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nc*n_stream);
                    if (n_blk*bs != n_comp) {
                        s2 = ggml_concat(ctx0, s2, ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk*bs - n_comp, nc*n_stream), -INFINITY), 0);
                    }
                    ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0);
                    bmax = ggml_add(ctx0, ggml_reshape_4d(ctx0, bmax, n_blk, nc, 1, n_stream), rows(cand_pin));
                    cb(bmax, "cand_block_score", il);
                    parts_sel.push_back(ggml_top_k(ctx0, bmax, std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk)));
                } else if (use_blk) {
                    // own scores, restricted to the source's candidate blocks: 0 inside them, -inf elsewhere
                    ggml_tensor * bsel = ggml_cont(ctx0, rows(cand_sel));
                    const int64_t k_b = bsel->ne[0];
                    ggml_tensor * bm    = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blk, nc*n_stream, 1), -INFINITY);
                    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_b, nc*n_stream, 1), 0.0f);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nc*n_stream, 1));
                    bm = ggml_reshape_2d(ctx0, ggml_repeat_4d(ctx0, bm, bs, n_blk, nc*n_stream, 1), bs*n_blk, nc*n_stream);
                    bm = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nc*n_stream, bm->nb[1], 0)), n_comp, nc, 1, n_stream);
                    cb(bm, "cand_mask", il);
                    score = ggml_add(ctx0, score, bm);
                }

                parts_top_k.push_back(ggml_top_k(ctx0, score, std::min<int64_t>(n_comp, hparams.indexer_top_k)));
            }

            const auto join = [&](std::vector<ggml_tensor *> v) {
                while (v.size() > 1) {
                    std::vector<ggml_tensor *> w;
                    for (size_t i = 0; i < v.size(); i += 2) {
                        w.push_back(i + 1 < v.size() ? ggml_concat(ctx0, v[i], v[i + 1], 1) : v[i]);
                    }
                    v.swap(w);
                }
                return v[0];
            };
            if (pick_blk) {
                cand_sel    = join(parts_sel);
                cand_n_comp = n_comp;
            }
            top_k = ggml_cont(ctx0, join(parts_top_k));
            cb(top_k, "idx_top_k", il);
            topk_of[il] = top_k;
        }
        GGML_ASSERT(top_k && "DeepSeek-V4.1: top-k source layer produced no top-k");

        // a quantized K cache stores the window Hadamard-rotated (as build_raw_attention)
        ggml_tensor * k_rot = inp_attn->self_k_rot;
        if (k_rot) {
            q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
            kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
        }
        ggml_build_forward_expand(gf, q);
        ggml_build_forward_expand(gf, kv);
        ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

        ggml_tensor * raw_k    = inp_attn->mctx->get_k(ctx0, il);
        ggml_tensor * raw_mask = inp_attn->get_kq_mask();

        const int32_t topk_src = hparams.dsv41_topk_src[il];
        const int64_t k_sel    = top_k->ne[0];
        const int64_t n_raw    = raw_k->ne[2];

        // few queries per stream: gather each query's top-k rows and attend over [window | selected], one FA stream
        // per query, instead of a full n_comp mask per layer. same keys and mask as dense, different key order
        const bool gather = sparse && cparams.flash_attn && raw_mask->type == GGML_TYPE_F16 && ci.kq_mask->type == GGML_TYPE_F16 &&
                ((n_stream == 1 && nt <= 32) || nt == n_stream);

        if (gather) {
            // built once per (storage, top-k) pair
            ggml_tensor *& comp_k = gather_k_of[{rd, topk_src}];
            if (!comp_k) {
                if (n_stream == 1) {
                    ggml_tensor * src_rows = ggml_view_3d(ctx0, comp_all, n_embd_head, n_comp, 1, comp_all->nb[2], comp_all->nb[3], 0);
                    comp_k = ggml_get_rows(ctx0, src_rows, ggml_reshape_1d(ctx0, top_k, k_sel*nt));
                } else {
                    ggml_tensor * src_rows = ggml_view_3d(ctx0, comp_all, n_embd_head, n_comp, n_stream, comp_all->nb[2], comp_all->nb[3], 0);
                    comp_k = ggml_get_rows(ctx0, src_rows, ggml_reshape_2d(ctx0, top_k, k_sel, n_stream));
                }
                comp_k = ggml_reshape_4d(ctx0, comp_k, n_embd_head, 1, k_sel, nt); // F32 (get_rows dequantizes)
                if (k_rot) {
                    // quantized K cache: the window is stored rotated, the compressed rows are not
                    comp_k = llama_mul_mat_hadamard(ctx0, comp_k, k_rot);
                }
                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type); // exact: rows are raw_k->type values
                }
                cb(comp_k, "comp_k_sel", il);
            }

            if (n_stream == 1 && nt > 1) {
                if (ggml_is_quantized(raw_k->type)) {
                    // no REPEAT kernel for quantized types: byte-exact copies by doubling concats
                    ggml_tensor * rep = nullptr;
                    ggml_tensor * pw  = raw_k;
                    for (int64_t n = nt; n > 0; n >>= 1) {
                        if (n & 1) {
                            rep = rep ? ggml_concat(ctx0, rep, pw, 3) : pw;
                        }
                        if (n > 1) {
                            pw = ggml_concat(ctx0, pw, pw, 3);
                        }
                    }
                    raw_k = rep;
                } else {
                    raw_k = ggml_repeat_4d(ctx0, raw_k, n_embd_head, 1, n_raw, nt);
                }
            }
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

            // mask: window row, then comp mask at the selected rows (-inf where top-k picked an invisible row);
            // one per top-k source, shared by its layers
            ggml_tensor *& kq_mask = gather_mask_of[topk_src];
            if (!kq_mask) {
                ggml_tensor * sel_mask = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, ci.kq_mask, 1, n_comp, nt),
                        ggml_reshape_2d(ctx0, top_k, k_sel, nt));
                sel_mask = ggml_cast(ctx0, ggml_reshape_4d(ctx0, sel_mask, k_sel, 1, 1, nt), GGML_TYPE_F16);
                kq_mask = ggml_concat(ctx0, ggml_reshape_4d(ctx0, raw_mask, raw_mask->ne[0], 1, 1, nt), sel_mask, 0);
                cb(kq_mask, "comp_kq_mask_sel", il);
            }

            out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks, nullptr, 0, kq_scale, il);
        } else {
            // dense attention over [window | all compressed rows], selection applied as a mask
            ggml_tensor * comp_k = ggml_view_4d(ctx0, comp_all, n_embd_head, 1, n_comp, n_stream,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], 0);
            if (k_rot) {
                // compressed rows are stored unrotated: rotate in F32, then cast to the cache type (dequantized from
                // the strided view: CUDA has no same-type quantized copy)
                comp_k = comp_k->type != GGML_TYPE_F32 ? ggml_cast(ctx0, comp_k, GGML_TYPE_F32) : ggml_cont(ctx0, comp_k);
                comp_k = llama_mul_mat_hadamard(ctx0, comp_k, k_rot);
                if (comp_k->type != raw_k->type) {
                    comp_k = ggml_cast(ctx0, comp_k, raw_k->type);
                }
            } else if (ggml_is_quantized(comp_k->type)) {
                // quantized cache without rotation (LLAMA_ATTN_ROT_DISABLE=1): via F32 for the same reason
                comp_k = ggml_cast(ctx0, ggml_cast(ctx0, comp_k, GGML_TYPE_F32), raw_k->type);
            } else {
                comp_k = comp_k->type != raw_k->type ? ggml_cast(ctx0, comp_k, raw_k->type) : ggml_cont(ctx0, comp_k);
            }
            cb(comp_k, "comp_k", il);
            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

            ggml_tensor * sel_mask = build_top_k_mask(ci.kq_mask, top_k, "comp_top_k_mask", il);
            ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_mask, sel_mask, 0);
            cb(kq_mask, "comp_kq_mask", il);

            // at most window + top-k visible columns per query: lets CUDA FA gather only those
            const int64_t n_kv_max = sparse ? std::min<int64_t>(raw_mask->ne[0], hparams.n_swa) + k_sel : 0;
            out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks, nullptr, n_kv_max, kq_scale, il);
        }
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, "attn_comp", il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_permute(ctx0, ggml_reshape_3d(ctx0, out, n_head/n_groups*n_embd_head, n_groups, nt), 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_cont_2d(ctx0, ggml_permute(ctx0, oa, 0, 2, 1, 3), o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek41::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & model = static_cast<const llama_model_deepseek41 &>(model_base);

    static const bool sparse_env = dsv41_env_on("LLAMA_DSV41_SPARSE");
    sparse = sparse_env;

    topk_of.assign(n_layer, nullptr);
    comp_row_of.assign(n_layer, nullptr);

    ggml_tensor * cur;

    // LLAMA_DSV41_HOST_GATHER=0: in-graph GET_ROWS on the host tables (one CPU split each) instead of
    // gathering token and Engram rows in set_input (bitwise the same rows)
    static const bool host_gather = dsv41_env_on("LLAMA_DSV41_HOST_GATHER");

    ggml_tensor * inp = build_inp_embd(model.tok_embd, 1.0f, host_gather);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_kq_mask);

    // long-prompt layer skip (prompt head ubatch): layers before the skip layer in full, the skip layer only to fill its
    // compressed rows and index keys; the rest of the layers read nothing else of these rows (llama_context prompt skip)
    const int32_t yoco = hparams.dsv41_yoco;
    const bool    head = yoco >= 0 && inp_dsv4->mctx->is_prompt_head();
    inp_dsv4->head = head;

    // LLAMA_DSV41_ENGRAM_OFF=1 (debugging only): no Engram contribution at all
    static const bool engram_off = [] {
        const char * v = getenv("LLAMA_DSV41_ENGRAM_OFF");
        return v && atoi(v) != 0;
    }();

    auto inp41 = std::make_unique<llm_graph_input_dsv41>(model, inp_dsv4->mctx);
    inp41->n_ubatch = cparams.n_ubatch;
    // embedding input has no token ids: it skips Engram, like the reference's masked (image) positions
    for (size_t e = 0; ubatch.token && !engram_off && e < model.engram_layers.size(); ++e) {
        const int32_t il = model.engram_layers[e];
        if (head && il > yoco) {
            break; // layers are ascending: none of the remaining ones runs
        }
        const ggml_tensor * tab = model.engram_table((int) e);
        ggml_tensor * ids = nullptr;
        if (host_gather && llm_graph_host_rows_supported(tab) && tab->ne[0] == (int64_t) model.engram_head_dim) {
            // host-resident table: rows are gathered in set_input
            ids = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, (int64_t) model.engram_n_cols()*tab->ne[0], n_tokens);
            ggml_set_name(ids, ("engram_rows_in-" + std::to_string(il)).c_str());
        } else {
            tab = nullptr;
            ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) model.engram_n_cols()*n_tokens);
            ggml_set_name(ids, ("engram_ids-" + std::to_string(il)).c_str());
        }
        ggml_set_input(ids);
        inp41->engram_ids.push_back(ids);
        inp41->engram_tab.push_back(tab);
    }
    ggml_tensor * engram_keep = nullptr;
    if (!inp41->engram_ids.empty() && ubatch.token && ubatch.embd) {
        engram_keep = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(engram_keep);
        ggml_set_name(engram_keep, "engram_keep");
        inp41->engram_keep = engram_keep;
    }
    ggml_tensor * cand_pin = nullptr;
    if (hparams.dsv41_cand_src >= 0) {
        inp41->cand_csa = hparams.dsv4_compress_ratios[hparams.dsv41_cand_src] == 2;
        const auto & ci = inp41->cand_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
        if (ci.kq_mask) {
            const int64_t bs = hparams.dsv41_cand_block_size;
            cand_pin = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, (ci.kq_mask->ne[0] + bs - 1)/bs, ci.kq_mask->ne[1], 1, ci.kq_mask->ne[3]);
            ggml_set_input(cand_pin);
            ggml_set_name(cand_pin, "cand_pin");
            inp41->cand_pin = cand_pin;
        }
    }
    const auto * inp41_ptr = inp41.get();
    res->add_input(std::move(inp41));

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // single-pass mHC: the pre-mix used to collapse the stream lags one sublayer
    ggml_tensor * pre_prev = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int e = model.engram_index(il);
        if (e >= 0 && !inp41_ptr->engram_ids.empty()) {
            inpL = build_engram(model, inpL, inp41_ptr->engram_ids[e], engram_keep, il);
            cb(inpL, "engram_out", il);
        }

        // DSpark target features: mean over the hc copies of the layer input
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * attn_pre = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il, true);

        // the first layer's attention collapses with the identity pre-mix: copy 0
        cur = pre_prev ? build_hc_pre(inpL, pre_prev, il) : ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (head && il == yoco) {
            build_attention41(model, inp_dsv4, cur, inp_pos, cand_pin, il, true);
            break;
        }

        cur = build_attention41(model, inp_dsv4, cur, inp_pos, cand_pin, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        pre_prev = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il, true);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);
        ggml_tensor * ffn_in = cur;

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // F32 router inputs: expert selection is near-tie sensitive
        ggml_tensor * logits = build_lora_mm(layer.ffn_gate_inp, cur);
        ggml_prec_set_acc(logits, GGML_PREC_F32);
        if (!ggml_is_quantized(layer.ffn_gate_inp->type)) {
            ggml_prec_set_src(logits, GGML_PREC_F32, 1);
        }
        cb(logits, "ffn_moe_logits", il);

        // image (embedding) tokens use the vision router bias, as DeepSeek-V4
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        // decode-only next-layer expert predictor for the MoE slot arena (LLAMA_DSV41_MOE_PREDICT=1): layer il+1's router
        // (its ffn_norm, gate_inp, sqrt-softplus and selection bias) applied to this layer's FFN input. The cache reads it
        // after this layer's ids and prefetches the predicted misses while the GPU computes; no output depends on it.
        if (llama_moe_cache_predict_enabled() && n_tokens == 1 && il + 1 < n_layer) {
            const auto & nl = model.layers[il + 1];
            if (nl.ffn_gate_inp && nl.ffn_exp_probs_b && nl.ffn_norm && nl.ffn_gate_exps) {
                ggml_tensor * px = build_norm(ffn_in, nl.ffn_norm, nullptr, LLM_NORM_RMS, il);
                ggml_tensor * pl = ggml_mul_mat(ctx0, nl.ffn_gate_inp, px);
                ggml_prec_set_acc(pl, GGML_PREC_F32);
                ggml_tensor * pp = ggml_add(ctx0, ggml_sqrt(ctx0, ggml_softplus(ctx0, pl)), nl.ffn_exp_probs_b);
                ggml_tensor * pred = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, pp, llama_moe_cache_predict_k()));
                ggml_format_name(pred, "dsv41_moe_pred-%d", il + 1);
                ggml_set_output(pred);
                ggml_build_forward_expand(gf, pred);
            }
        }

        // MoE cache overlap: the shared expert is built between the two lookups of the layer (it needs no routed expert), so
        // that it runs while the missing experts are read (single-token decode; LLAMA_DSV41_MOE_OVERLAP_NT=1: up to 16 tokens)
        ggml_tensor * ffn_shexp = nullptr;
        if (moe_cache != nullptr && n_tokens >= 1 && n_tokens <= moe_cache->overlap_max_tokens()) {
            moe_overlap_hook = [&]() {
                ffn_shexp = build_ffn(cur,
                        layer.ffn_up_shexp, nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr,
                        layer.ffn_down_shexp, nullptr, nullptr,
                        nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(ffn_shexp, "ffn_shexp", il);
                ggml_build_forward_expand(gf, ffn_shexp);
            };
        }

        // the reference normalizes only when it routes to more than one expert
        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm && hparams.n_expert_used() > 1,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il, logits);
        cb(moe_out, "ffn_moe_out", il);

        moe_overlap_hook = nullptr;
        if (ffn_shexp == nullptr) {
            ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);
        }

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_out", il);
    }

    // no hc_head: collapse with the last FFN's pre-mix
    if (inp_out_ids) {
        inpL     = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens), inp_out_ids), n_embd, hc, inp_out_ids->ne[0]);
        pre_prev = ggml_get_rows(ctx0, pre_prev, inp_out_ids);
    }
    cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
