#pragma once

// Cost-aware draft-length controller for deterministic (n-gram / lookup) speculative drafts.
//
// Idea from Strata include/strata/spec/{controller,draft_policy}.hpp (MIT) and Splash runtime/model/VerifyBudget.*
// (Apache-2.0): choose the verify width k per step by maximising expected emitted tokens per measured millisecond,
//   k* = argmax_k  E[tokens | k] / T(k+1),   E[tokens | k] = 1 + sum_{j=1..k} prod_{i<=j} a_i
// and skip drafting when even the best k does not beat plain decode (k = 0) by min_gain.
//
// Exactness: the draft is a point mass, so ANY prefix of it is again a valid draft and the sample-and-compare
// acceptance (target sample at each row, accept while equal to the draft) stays distribution-exact. The controller
// only decides how many rows are verified; it never touches sampling.
//
// T(n) is the measured wall time of a pure-generation n-row step (this includes the arena misses of the batch's
// expert union, which is what makes wide verifies expensive on this SSD-bound engine). T(1) is measured from
// ordinary 1-token steps; T(n>1) from verify steps (per-n EMA, else T1 * (1 + beta*(n-1)) with beta learned).
//
// Env: LLAMA_SPEC_CTRL=1 enables, LLAMA_SPEC_CTRL_MIN_GAIN (default 0.05), LLAMA_SPEC_CTRL_TRACE=1.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct spec_cost_ctrl {
    static constexpr int N_MAX = 65;

    bool    enabled  = false;
    bool    trace    = false;
    double  min_gain = 0.05;

    // measured step cost
    double  t1_ms = 0.0;           // EMA of 1-row step
    int     t1_n  = 0;
    double  prior_beta = 0.5;
    double  gamma = 0.85;
    double  beta  = 0.5;           // extra cost per extra row, in units of T1 (measured: 4 rows 2.1x, 9 rows 5.5x)
    int     beta_n = 0;
    double  t_ms[N_MAX] = {};      // EMA per row count
    int     t_n [N_MAX] = {};

    // acceptance: decayed beta-binomial counts
    double  s1 = 0, f1 = 0;        // first draft position
    double  sc = 0, fc = 0;        // later positions, conditional on the previous one being accepted
    double  prior_a1 = 0.8, prior_c = 0.9, prior_w = 3.0;
    double  decay = 0.92;
    int     skipped = 0;

    // stats
    uint64_t n_rounds = 0, n_skipped = 0, n_cut = 0;

    void init_from_env() {
        const char * e = getenv("LLAMA_SPEC_CTRL");
        enabled = e && atoi(e) > 0;
        if ((e = getenv("LLAMA_SPEC_CTRL_MIN_GAIN"))) min_gain = atof(e);
        if ((e = getenv("LLAMA_SPEC_CTRL_TRACE")))    trace = atoi(e) > 0;
        if ((e = getenv("LLAMA_SPEC_CTRL_PRIOR_A1"))) prior_a1 = atof(e);
    }

    double a1() const { return (s1 + prior_a1*prior_w) / (s1 + f1 + prior_w); }
    double c () const { return (sc + prior_c *prior_w) / (sc + fc + prior_w); }

    // extra rows get cheaper as the batch's expert union saturates (measured: 4 rows 2.1-2.5x, 9 rows 5.5x one row),
    // so the cost model is concave in n
    double model(int n) const {
        const double t1 = t1_n > 0 ? t1_ms : 100.0;
        return t1 * (1.0 + beta*std::pow((double) (n - 1), gamma));
    }

    double cost(int n) const {   // ms of an n-row pure generation step
        if (n < 1) n = 1;
        if (n == 1 && t1_n > 0) return t1_ms;
        if (n < N_MAX && t_n[n] > 0) {
            const double w = t_n[n] / (t_n[n] + 2.0);   // observed EMA, shrunk toward the model until it has samples
            return w*t_ms[n] + (1.0 - w)*model(n);
        }
        return model(n);
    }

    // expected emitted tokens for a k-token draft
    double expect(int k) const {
        double e = 1.0, p = 1.0;
        for (int j = 1; j <= k; ++j) {
            p *= (j == 1 ? a1() : c());
            e += p;
        }
        return e;
    }

    // number of draft tokens to verify out of a draft of length n_draft (0 = do not speculate this round)
    int choose(int n_draft) {
        if (!enabled || n_draft <= 0) return n_draft;
        n_rounds++;
        // no cost data yet (cold start): allow a short probe so that the cost table fills
        if (t1_n < 3) return std::min(n_draft, 4);
        const double base = 1.0 / cost(1);
        double best = base*(1.0 + min_gain);
        int    best_k = 0;
        for (int k = 1; k <= n_draft && k + 1 < N_MAX; ++k) {
            const double v = expect(k) / cost(k + 1);
            if (v > best) { best = v; best_k = k; }
        }
        if (best_k == 0) {
            n_skipped++;
            // no information arrives while skipping: drift the counts back toward the optimistic prior so a probe
            // happens again after a few rounds (bounded duty cycle of failed probes)
            s1 *= 0.85; f1 *= 0.85;
            sc *= 0.85; fc *= 0.85;
            // same for the cost belief: a verify step that was slow (cold arena) must not lock speculation out forever
            beta += 0.15*(prior_beta - beta);
            for (int n = 2; n < N_MAX; ++n) {
                if (t_n[n] > 0 && t1_n > 0) {
                    t_ms[n] += 0.15*(t1_ms*(1.0 + prior_beta*std::pow((double) (n - 1), gamma)) - t_ms[n]);
                }
            }
        } else if (best_k < n_draft) {
            n_cut++;
        }
        if (trace) {
            fprintf(stderr, "spec-ctrl: draft=%d -> k=%d (a1=%.2f c=%.2f T1=%.1f T(k+1)=%.1f gain=%.3f)\n", n_draft, best_k, a1(), c(),
                    cost(1), cost(best_k + 1), best_k ? best/base : 1.0);
        }
        return best_k;
    }

    // result of a verified draft of k tokens: m accepted
    void observe_accept(int k, int m) {
        if (!enabled || k <= 0) return;
        s1 *= decay; f1 *= decay; sc *= decay; fc *= decay;
        if (m >= 1) s1 += 1; else f1 += 1;
        // positions 2..k were evaluated up to the first failure
        if (m >= 1) {
            const int evaluated = std::min(k, m + 1) - 1;   // positions 2..min(k,m+1)
            sc += std::max(0, std::min(m, k) - 1);
            if (m < k && evaluated >= 1) fc += 1;
        }
    }

    // measured wall time of a pure generation step with n rows
    void observe_time(int n, double ms) {
        if (!enabled || n < 1 || n >= N_MAX || ms <= 0) return;
        const double a = 0.2;
        t_ms[n] = t_n[n] ? (1 - a)*t_ms[n] + a*ms : ms;
        t_n[n]++;
        if (n == 1) {
            t1_ms = t1_n ? (1 - a)*t1_ms + a*ms : ms;
            t1_n++;
        } else if (t1_n > 0) {
            const double b = (ms/t1_ms - 1.0) / std::pow((double) (n - 1), gamma);
            beta = (beta*3.0 + b) / 4.0;   // prior weight 3: one cold-arena step cannot decide by itself
            beta = std::max(0.05, std::min(2.0, beta));
            beta_n++;
        }
    }
};
