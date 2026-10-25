// llama-skipcheck: last-row logits of a long prompt plus teacher-forced continuation logits, for comparing runs with
// and without LLAMA_DSV41_PROMPT_SKIP (one process per mode: the flag is read once).
//   SKIPCHK_NTOK=N      use the first N prompt tokens (default: all)
//   SKIPCHK_OUT=file    binary output: int32 rows, int32 n_vocab, float32 logits[rows][n_vocab]
//   SKIPCHK_GEN=K       greedy continuation of K tokens, written to SKIPCHK_TOKS (text, one id per line)
//   SKIPCHK_FORCE=file  teacher-force the tokens of this file instead of sampling
//   SKIPCHK_HINT=0      do not call llama_set_prompt_total (reference run without a hint)
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static int decode_range(llama_context * ctx, llama_batch_ext * batch, const std::vector<llama_token> & toks, int p0, int p1, bool out_last) {
    llama_batch_ext_clear(batch);
    for (int i = p0; i < p1; ++i) {
        const int32_t idx = llama_batch_ext_add_token(batch, 0, toks[i]);
        const llama_pos pos = i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    if (out_last) {
        llama_batch_ext_set_output_logits(batch, p1 - p0 - 1, true);
    }
    return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    llama_backend_init();
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true);
    if (const char * v = getenv("SKIPCHK_NTOK")) {
        const size_t n = (size_t) atoll(v);
        if (n < toks.size()) {
            toks.resize(n);
        }
    }
    const int n_prompt = (int) toks.size();
    const int n_batch  = (int) llama_n_batch(ctx);
    fprintf(stderr, "skipcheck: %d prompt tokens, n_batch %d\n", n_prompt, n_batch);

    const char * hint = getenv("SKIPCHK_HINT");
    if (!hint || atoi(hint) != 0) {
        llama_set_prompt_total(ctx, 0, n_prompt);
    }
    // DeepSeek-V4.1 Engram prefetch (LLAMA_DSV41_ENGRAM_AHEAD=1): the coming tokens
    llama_set_prompt_tokens(ctx, 0, toks.data(), n_prompt, 0);

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    std::vector<float> rows;
    const auto grab = [&]() {
        const float * l = llama_get_logits_ith(ctx, -1);
        rows.insert(rows.end(), l, l + n_vocab);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (l[i] > l[best]) best = i;
        }
        return best;
    };

    const int64_t t0 = ggml_time_us();
    for (int p = 0; p < n_prompt; p += n_batch) {
        const int e = std::min(n_prompt, p + n_batch);
        if (decode_range(ctx, batch, toks, p, e, e == n_prompt)) {
            fprintf(stderr, "decode failed at %d\n", p);
            return 1;
        }
    }
    llama_synchronize(ctx);
    fprintf(stderr, "skipcheck: prompt %.2f s\n", (ggml_time_us() - t0) / 1e6);
    int next = grab();

    std::vector<llama_token> forced;
    if (const char * f = getenv("SKIPCHK_FORCE")) {
        std::ifstream in(f);
        int t;
        while (in >> t) forced.push_back(t);
    }
    const int n_gen = getenv("SKIPCHK_GEN") ? atoi(getenv("SKIPCHK_GEN")) : (int) forced.size();
    std::vector<llama_token> gen;
    // SKIPCHK_BATCH=N (with SKIPCHK_FORCE): feed the forced tokens N at a time with logits on every row (batched path
    // instead of single-token decode; same rows and semantics), to locate decode-only numeric differences
    const int n_chunk = getenv("SKIPCHK_BATCH") ? atoi(getenv("SKIPCHK_BATCH")) : 0;
    if (n_chunk > 0 && !forced.empty()) {
        std::vector<llama_token> all(n_prompt, 0);
        all.insert(all.end(), forced.begin(), forced.end());
        llama_set_prompt_tokens(ctx, 0, forced.data(), std::min<int>(n_gen, (int) forced.size()), n_prompt);
        for (int k0 = 0; k0 < n_gen; k0 += n_chunk) {
            const int k1 = std::min(n_gen, k0 + n_chunk);
            llama_batch_ext_clear(batch);
            for (int k = k0; k < k1; ++k) {
                const int32_t idx = llama_batch_ext_add_token(batch, 0, all[n_prompt + k]);
                const llama_pos pos = n_prompt + k;
                llama_batch_ext_set_pos(batch, idx, &pos);
                llama_batch_ext_set_output_logits(batch, k - k0, true);
            }
            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) {
                fprintf(stderr, "batched gen decode failed\n");
                return 1;
            }
            for (int k = k0; k < k1; ++k) {
                const float * l = llama_get_logits_ith(ctx, k - k0);
                rows.insert(rows.end(), l, l + n_vocab);
                gen.push_back(all[n_prompt + k]);
            }
        }
    }
    for (int k = (n_chunk > 0 && !forced.empty()) ? n_gen : 0; k < n_gen; ++k) {
        const llama_token feed = k < (int) forced.size() ? forced[k] : next;
        gen.push_back(feed);
        std::vector<llama_token> one(n_prompt + k + 1, 0);
        one[n_prompt + k] = feed;
        if (decode_range(ctx, batch, one, n_prompt + k, n_prompt + k + 1, true)) {
            fprintf(stderr, "gen decode failed\n");
            return 1;
        }
        next = grab();
    }
    if (const char * f = getenv("SKIPCHK_TOKS")) {
        std::ofstream o(f);
        for (auto t : gen) o << t << "\n";
    }
    if (const char * f = getenv("SKIPCHK_OUT")) {
        std::ofstream o(f, std::ios::binary);
        int32_t hdr[2] = { (int32_t) (rows.size() / n_vocab), n_vocab };
        o.write((const char *) hdr, sizeof(hdr));
        o.write((const char *) rows.data(), rows.size() * sizeof(float));
    }
    llama_batch_ext_free(batch);
    return 0;
}
