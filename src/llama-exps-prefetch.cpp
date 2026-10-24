#include "llama-exps-prefetch.h"

#include "llama-impl.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__) || defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define LLAMA_EXPS_PREFETCH_POSIX
#endif
#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace llama_exps_prefetch {

static int env_int(const char * name, int def) {
    const char * v = getenv(name);
    return v && *v ? atoi(v) : def;
}

bool enabled() {
#ifdef LLAMA_EXPS_PREFETCH_POSIX
    static const bool on = env_int("LLAMA_DSV41_EXPS_PREFETCH", 0) == 1;
    return on;
#else
    return false;
#endif
}

#ifdef LLAMA_EXPS_PREFETCH_POSIX

#if defined(__APPLE__)
using mincore_vec_t = char;
#else
using mincore_vec_t = unsigned char;
#endif

static constexpr size_t PAGE      = 16384;
static constexpr size_t JOB_BYTES = 1u << 20;
static constexpr size_t MERGE_GAP = 64u << 10;

struct mapping {
    const void * owner;
    const char * base;
    size_t       len;
    int          fd;
};

// one page-aligned piece of a mapping, at most JOB_BYTES
struct job {
    int          fd;
    const char * addr;
    size_t       off;
    size_t       len;
    int          kind;
};

struct layer_ud {
    const void * owner;
    int          il;
};

struct layer_exps {
    std::vector<std::pair<const char *, size_t>> ranges; // data, nbytes of each expert tensor
};

// [0] = decode (n_tokens == 1), [1] = batch, [2] = next-layer prefetch
struct stats {
    std::atomic<uint64_t> calls[3]          = {};
    std::atomic<uint64_t> experts[3]        = {};
    std::atomic<uint64_t> pages_checked[3]  = {};
    std::atomic<uint64_t> pages_resident[3] = {};
    std::atomic<uint64_t> bytes_issued[3]   = {};
    std::atomic<uint64_t> tokens[2]         = {};
    std::atomic<uint64_t> op_us             = 0;
    std::atomic<uint64_t> jobs_done         = 0;
    std::atomic<uint64_t> advise_us         = 0;
    std::atomic<uint64_t> queue_hwm         = 0;
};

struct state {
    std::mutex mtx; // mappings, layers, uds
    std::vector<mapping> mappings;
    std::map<std::pair<const void *, int>, layer_exps> layers;
    std::deque<layer_ud> uds;

    // advise pool: jobs of the current layer go before the next-layer jobs
    std::mutex              qmtx;
    std::condition_variable qcv;
    std::condition_variable qidle;
    std::deque<job>         q_cur;
    std::deque<job>         q_next;
    int                     n_busy = 0;
    std::vector<std::thread> workers;

    stats st;

    static void advise(int fd, size_t off, size_t len) {
#if defined(__APPLE__)
        struct radvisory ra = { (off_t) off, (int) len };
        fcntl(fd, F_RDADVISE, &ra);
#else
        posix_fadvise(fd, (off_t) off, (off_t) len, POSIX_FADV_WILLNEED);
#endif
    }

    // advise only the runs of pages that are not in the page cache yet (in-flight pages count as resident)
    void process(const job & j) {
        const size_t np = (j.len + PAGE - 1) / PAGE;
        mincore_vec_t vec[JOB_BYTES / PAGE];
        if (mincore((void *) j.addr, j.len, vec) != 0) {
            advise(j.fd, j.off, j.len);
            st.pages_checked[j.kind] += np;
            st.bytes_issued[j.kind]  += j.len;
            return;
        }
        size_t n_res = 0;
        size_t run0  = SIZE_MAX;
        size_t run1  = 0;
        auto flush = [&]() {
            if (run0 != SIZE_MAX) {
                const size_t b = run0 * PAGE;
                const size_t e = std::min(run1 * PAGE, j.len);
                advise(j.fd, j.off + b, e - b);
                st.bytes_issued[j.kind] += e - b;
            }
        };
        for (size_t i = 0; i < np; ++i) {
            if (vec[i] & 1) {
                n_res++;
                continue;
            }
            if (run0 != SIZE_MAX && (i - run1) * PAGE < MERGE_GAP) {
                run1 = i + 1;
            } else {
                flush();
                run0 = i;
                run1 = i + 1;
            }
        }
        flush();
        st.pages_checked[j.kind]  += np;
        st.pages_resident[j.kind] += n_res;
    }

    [[noreturn]] void worker() {
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lk(qmtx);
                qcv.wait(lk, [&] { return !q_cur.empty() || !q_next.empty(); });
                auto & q = !q_cur.empty() ? q_cur : q_next;
                j = q.front();
                q.pop_front();
                n_busy++;
            }
            const auto t0 = std::chrono::steady_clock::now();
            process(j);
            st.advise_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
            st.jobs_done++;
            {
                std::lock_guard<std::mutex> lk(qmtx);
                n_busy--;
                if (n_busy == 0 && q_cur.empty() && q_next.empty()) {
                    qidle.notify_all();
                }
            }
        }
    }

    void start_pool() {
        // called with qmtx held
        if (!workers.empty()) {
            return;
        }
        const int n = std::max(1, env_int("LLAMA_DSV41_PREFETCH_THREADS", 4));
        for (int i = 0; i < n; ++i) {
            workers.emplace_back([this] { worker(); });
            workers.back().detach();
        }
    }

    void submit(std::vector<job> & jobs, bool next) {
        if (jobs.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lk(qmtx);
        start_pool();
        auto & q = next ? q_next : q_cur;
        q.insert(q.end(), jobs.begin(), jobs.end());
        const uint64_t depth = q_cur.size() + q_next.size();
        if (depth > st.queue_hwm) {
            st.queue_hwm = depth;
        }
        qcv.notify_all();
    }

    void drop_fd_and_wait(const std::vector<int> & fds) {
        std::unique_lock<std::mutex> lk(qmtx);
        auto has_fd = [&](const job & j) { return std::find(fds.begin(), fds.end(), j.fd) != fds.end(); };
        q_cur.erase(std::remove_if(q_cur.begin(), q_cur.end(), has_fd), q_cur.end());
        q_next.erase(std::remove_if(q_next.begin(), q_next.end(), has_fd), q_next.end());
        qidle.wait(lk, [&] { return n_busy == 0 && q_cur.empty() && q_next.empty(); });
    }
};

// leaked on purpose: the detached workers may outlive static destruction
static state & get() {
    static state * s = new state();
    return *s;
}

void register_mapping(const void * owner, const void * addr, size_t len, int fd) {
    const int fd2 = dup(fd);
    if (fd2 < 0) {
        return;
    }
    fcntl(fd2, F_SETFD, FD_CLOEXEC);
    auto & s = get();
    std::lock_guard<std::mutex> lk(s.mtx);
    s.mappings.push_back({ owner, (const char *) addr, len, fd2 });
}

static void print_stats(const stats & st) {
    static const char * names[3] = { "decode", "batch", "next-layer" };
    for (int k = 0; k < 3; ++k) {
        const uint64_t calls = st.calls[k];
        if (calls == 0) {
            continue;
        }
        const uint64_t chk = st.pages_checked[k];
        const uint64_t res = st.pages_resident[k];
        const double   mib = st.bytes_issued[k] / 1048576.0;
        const uint64_t tok = k < 2 ? (uint64_t) st.tokens[k] : 0;
        fprintf(stderr, "%s: exps-prefetch %-10s calls=%llu experts/call=%.1f pages=%llu resident=%.1f%% issued=%.1f MiB (%.2f MiB/call%s)\n",
            __func__, names[k], (unsigned long long) calls, (double) st.experts[k] / calls, (unsigned long long) chk,
            chk ? 100.0 * res / chk : 0.0, mib, mib / calls,
            tok ? (", " + std::to_string(mib / tok) + " MiB/token").c_str() : "");
    }
    fprintf(stderr, "%s: exps-prefetch op=%.1f ms jobs=%llu advise=%.1f ms (pool time) queue_hwm=%llu\n", __func__,
        st.op_us / 1000.0, (unsigned long long) st.jobs_done, st.advise_us / 1000.0, (unsigned long long) st.queue_hwm);
}

void unregister_owner(const void * owner) {
    auto & s = get();
    std::vector<int> fds;
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        for (const auto & m : s.mappings) {
            if (m.owner == owner) {
                fds.push_back(m.fd);
            }
        }
        if (fds.empty()) {
            return;
        }
        s.mappings.erase(std::remove_if(s.mappings.begin(), s.mappings.end(), [&](const mapping & m) { return m.owner == owner; }), s.mappings.end());
        for (auto it = s.layers.begin(); it != s.layers.end();) {
            it = it->first.first == owner ? s.layers.erase(it) : std::next(it);
        }
    }
    s.drop_fd_and_wait(fds);
    for (int fd : fds) {
        close(fd);
    }
    if (env_int("LLAMA_DSV41_PREFETCH_STATS", 0) == 1) {
        print_stats(s.st);
    }
}

void * layer_userdata(const void * owner, int il, ggml_tensor * const * exps, int n_exps) {
    auto & s = get();
    std::lock_guard<std::mutex> lk(s.mtx);
    layer_exps & le = s.layers[{ owner, il }];
    le.ranges.clear();
    for (int i = 0; i < n_exps; ++i) {
        if (exps[i] != nullptr) {
            le.ranges.emplace_back((const char *) exps[i]->data, ggml_nbytes(exps[i]));
        }
    }
    for (auto & ud : s.uds) {
        if (ud.owner == owner && ud.il == il) {
            return &ud;
        }
    }
    s.uds.push_back({ owner, il });
    return &s.uds.back();
}

// splits [p, p + len) into page-aligned jobs of the mapping that holds it
static void add_range(const std::vector<mapping> & maps, std::vector<job> & jobs, const char * p, size_t len, int kind) {
    const char * a = (const char *) ((uintptr_t) p & ~(uintptr_t) (PAGE - 1));
    len = (size_t) ((p + len) - a);
    for (const auto & m : maps) {
        if (a >= m.base && a + len <= m.base + m.len) {
            for (size_t d = 0; d < len; d += JOB_BYTES) {
                jobs.push_back({ m.fd, a + d, (size_t) (a - m.base) + d, std::min(JOB_BYTES, len - d), kind });
            }
            return;
        }
    }
}

void op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto & s  = get();
    auto * ud = (const layer_ud *) userdata;

    *(int32_t *) dst->data = 0;

    const ggml_tensor * ids = dst->src[0];
    const int64_t n_used   = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];

    const ggml_tensor * w0 = dst->src[1];
    const int64_t n_expert = w0->ne[2];

    std::vector<uint8_t> sel(n_expert, 0);
    int64_t n_sel = 0;
    for (int64_t t = 0; t < n_tokens; ++t) {
        const char * row = (const char *) ids->data + t*ids->nb[1];
        for (int64_t k = 0; k < n_used; ++k) {
            const int32_t e = *(const int32_t *) (row + k*ids->nb[0]);
            if (e >= 0 && e < n_expert && !sel[e]) {
                sel[e] = 1;
                n_sel++;
            }
        }
    }

    const int kind = n_tokens == 1 ? 0 : 1;

    std::vector<std::pair<const char *, size_t>> next;
    std::vector<mapping> maps;
    bool first_layer = false;
    {
        std::lock_guard<std::mutex> lk(s.mtx);
        maps = s.mappings;
        auto it0 = s.layers.lower_bound({ ud->owner, INT_MIN });
        first_layer = it0 != s.layers.end() && it0->first.first == ud->owner && it0->first.second == ud->il;
        static const bool next_on  = env_int("LLAMA_DSV41_PREFETCH_NEXT", 1) != 0;
        static const int  next_min = env_int("LLAMA_DSV41_PREFETCH_NEXT_MIN", 32);
        if (next_on && n_tokens >= next_min) {
            auto it = s.layers.upper_bound({ ud->owner, ud->il });
            if (it != s.layers.end() && it->first.first == ud->owner) {
                next = it->second.ranges;
            }
        }
    }

    // compute order of build_moe_ffn: src[1..] are given as up, gate, (gate_up,) down
    std::vector<job> jobs;
    for (int i = 1; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        const ggml_tensor * w = dst->src[i];
        for (int64_t e = 0; e < w->ne[2]; ++e) {
            if (!sel[e]) {
                continue;
            }
            int64_t e1 = e;
            while (e1 + 1 < w->ne[2] && sel[e1 + 1]) {
                e1++;
            }
            add_range(maps, jobs, (const char *) w->data + e*w->nb[2], (e1 - e + 1)*w->nb[2], kind);
            e = e1;
        }
    }
    s.submit(jobs, false);

    s.st.calls[kind]++;
    if (first_layer) {
        s.st.tokens[kind] += n_tokens;
    }
    s.st.experts[kind] += n_sel;

    if (!next.empty()) {
        jobs.clear();
        for (const auto & r : next) {
            add_range(maps, jobs, r.first, r.second, 2);
        }
        s.submit(jobs, true);
        s.st.calls[2]++;
    }

    s.st.op_us += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
}

#else

void register_mapping(const void *, const void *, size_t, int) {}
void unregister_owner(const void *) {}
void * layer_userdata(const void *, int, ggml_tensor * const *, int) { return nullptr; }
void op(ggml_tensor * dst, int, int, void *) { *(int32_t *) dst->data = 0; }

#endif

} // namespace llama_exps_prefetch
