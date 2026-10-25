// llama-ds41-latency: Metal host <-> GPU latency microbenchmark (no model, no ggml)
//
// Decides whether an event-chained, GPU-resident decode token is worth building: one DS41 decode token makes ~40+
// GPU -> CPU -> GPU round trips (MoE arena plan per cached layer). Measured here:
//   (a) us per dependent tiny dispatch inside one command buffer (chain of N kernels, each reads the previous output),
//       for a serial encoder, a concurrent encoder with buffer barriers (what ggml-metal uses) and one encoder per dispatch
//   (b) us per command-buffer round trip (encode + commit + wait of a trivial command buffer), for each wait method:
//       waitUntilCompleted, status polling, shared-event signaledValue polling, waitUntilSignaledValue, completion
//       handler + semaphore; split into submit (commit -> GPUStartTime), GPU, and wake (GPUEndTime -> host return)
//   (c) CPU -> GPU -> CPU latency through MTLSharedEvents: a command buffer that is already committed waits on an event
//       (encodeWaitForEvent); the CPU writes a shared buffer and sets signaledValue; a kernel copies the value and the
//       command buffer signals a second event that the CPU polls (or waits on)
//
// usage: llama-ds41-latency [-n iters] [-c chain_len] [-s spin_budget_us] [-w busy_dispatches]
//
// Single-file Objective-C (manual retain/release, like ggml-metal). Prints a table and a decision line.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char * k_src =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "// one dependent step: every thread reads the previous value of its element and writes the next one\n"
    "kernel void k_step(device float * buf [[buffer(0)]], uint tid [[thread_position_in_grid]]) {\n"
    "    buf[tid] = buf[tid] * 0.5f + 1.0f;\n"
    "}\n"
    "// copy the CPU-written word to the output word (event round trip)\n"
    "kernel void k_echo(device const uint * in [[buffer(0)]], device uint * out [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n"
    "    if (tid == 0) { out[0] = in[0] + 1u; }\n"
    "}\n";

static mach_timebase_info_data_t g_tb;

static double now_s(void) {
    return (double) mach_absolute_time() * g_tb.numer / g_tb.denom * 1e-9;
}

static int cmp_dbl(const void * a, const void * b) {
    const double x = *(const double *) a, y = *(const double *) b;
    return x < y ? -1 : x > y ? 1 : 0;
}

struct stat5 { double med, p10, p90, mean, min; };

static struct stat5 stats(double * v, int n) {
    struct stat5 s = {0};
    if (n <= 0) {
        return s;
    }
    qsort(v, n, sizeof(double), cmp_dbl);
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += v[i];
    s.med = v[n/2]; s.p10 = v[n/10]; s.p90 = v[(n*9)/10]; s.mean = sum/n; s.min = v[0];
    return s;
}

// ------------------------------------------------------------------------------------------------
// (a) dependent dispatches in one command buffer
// ------------------------------------------------------------------------------------------------

enum chain_mode { CHAIN_SERIAL, CHAIN_CONCURRENT_BARRIER, CHAIN_ENCODER_PER_DISPATCH };

static double chain_gpu_us(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps, id<MTLBuffer> buf, int n, int threads, enum chain_mode mode) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> enc = nil;
        for (int i = 0; i < n; ++i) {
            if (enc == nil || mode == CHAIN_ENCODER_PER_DISPATCH) {
                if (enc) {
                    [enc endEncoding];
                }
                enc = mode == CHAIN_CONCURRENT_BARRIER ? [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                                                       : [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                [enc setBuffer:buf offset:0 atIndex:0];
            }
            if (mode == CHAIN_CONCURRENT_BARRIER && i > 0) {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            }
            [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads < 32 ? threads : 32, 1, 1)];
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "chain: command buffer failed\n");
            exit(1);
        }
        return ([cb GPUEndTime] - [cb GPUStartTime])*1e6;
    }
}

// ------------------------------------------------------------------------------------------------
// (b) command buffer round trips
// ------------------------------------------------------------------------------------------------

enum wait_mode { WAIT_BLOCK, WAIT_SPIN_STATUS, WAIT_SPIN_EVENT, WAIT_EVENT_BLOCK, WAIT_HANDLER, WAIT_COUNT };
static const char * wait_names[WAIT_COUNT] = {
    "waitUntilCompleted", "spin status", "spin event value", "waitUntilSignaledValue", "completion handler+sem",
};

struct rt_res {
    double total[4096], submit[4096], gpu[4096], wake[4096];
    int n, spin_miss;
};

// one round trip: encode `work` dependent dispatches (0 = empty command buffer with only the event signal), commit, wait
static void round_trip(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps, id<MTLBuffer> buf, id<MTLSharedEvent> ev, uint64_t * ev_val,
                       int work, enum wait_mode mode, int spin_us, struct rt_res * r) {
    @autoreleasepool {
        dispatch_semaphore_t sem = mode == WAIT_HANDLER ? dispatch_semaphore_create(0) : nil;
        const double t0 = now_s();
        id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
        if (work > 0) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf offset:0 atIndex:0];
            for (int i = 0; i < work; ++i) {
                [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            [enc endEncoding];
        }
        const uint64_t v = ++(*ev_val);
        [cb encodeSignalEvent:ev value:v];
        if (sem) {
            [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void) c; dispatch_semaphore_signal(sem); }];
        }
        [cb commit];
        const double tc = now_s();
        bool spun = true;
        switch (mode) {
            case WAIT_BLOCK:
                [cb waitUntilCompleted];
                break;
            case WAIT_SPIN_STATUS: {
                spun = false;
                while ((now_s() - tc)*1e6 < spin_us) {
                    const MTLCommandBufferStatus s = [cb status];
                    if (s == MTLCommandBufferStatusCompleted || s == MTLCommandBufferStatusError) { spun = true; break; }
                }
                if (!spun) [cb waitUntilCompleted];
            } break;
            case WAIT_SPIN_EVENT: {
                spun = false;
                while ((now_s() - tc)*1e6 < spin_us) {
                    if ([ev signaledValue] >= v) { spun = true; break; }
                }
                if (!spun) [ev waitUntilSignaledValue:v timeoutMS:10000];
            } break;
            case WAIT_EVENT_BLOCK:
                [ev waitUntilSignaledValue:v timeoutMS:10000];
                break;
            case WAIT_HANDLER:
                dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
                break;
            default: break;
        }
        const double t1 = now_s();
        // the timestamps need the command buffer to be completed (event modes may return before)
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "round trip: command buffer failed\n");
            exit(1);
        }
        const double g0 = [cb GPUStartTime], g1 = [cb GPUEndTime];
        if (sem) dispatch_release(sem);
        if (r->n < 4096) {
            r->total [r->n] = (t1 - t0)*1e6;
            r->submit[r->n] = (g0 - tc)*1e6;
            r->gpu   [r->n] = (g1 - g0)*1e6;
            r->wake  [r->n] = (t1 - g1)*1e6;
            r->n++;
        }
        r->spin_miss += !spun;
    }
}

// ------------------------------------------------------------------------------------------------
// (b2) back-to-back busy command buffers: the next one is encoded and committed as soon as the wait for the previous
//      one returns; the GPU idle gap GPUStartTime(k) - GPUEndTime(k-1) is the bubble one host round trip costs
// ------------------------------------------------------------------------------------------------

static int pingpong(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps, id<MTLBuffer> buf, id<MTLSharedEvent> ev, uint64_t * ev_val,
                    int work, enum wait_mode mode, int spin_us, int iters, double * gaps) {
    // the command buffers are kept and their timestamps read at the end, so nothing but the wait sits between two commits
    id<MTLCommandBuffer> * cbs = calloc(iters + 1, sizeof(id<MTLCommandBuffer>));
    for (int k = 0; k <= iters; ++k) {
        @autoreleasepool {
            dispatch_semaphore_t sem = mode == WAIT_HANDLER ? dispatch_semaphore_create(0) : nil;
            id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
            [cb retain];
            cbs[k] = cb;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:buf offset:0 atIndex:0];
            for (int i = 0; i < work; ++i) {
                [enc dispatchThreads:MTLSizeMake(32, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            [enc endEncoding];
            const uint64_t v = ++(*ev_val);
            [cb encodeSignalEvent:ev value:v];
            if (sem) {
                [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void) c; dispatch_semaphore_signal(sem); }];
            }
            [cb commit];
            const double tc = now_s();
            bool done = false;
            switch (mode) {
                case WAIT_BLOCK: [cb waitUntilCompleted]; break;
                case WAIT_SPIN_STATUS:
                    while (!done && (now_s() - tc)*1e6 < spin_us) {
                        const MTLCommandBufferStatus s = [cb status];
                        done = s == MTLCommandBufferStatusCompleted || s == MTLCommandBufferStatusError;
                    }
                    if (!done) [cb waitUntilCompleted];
                    break;
                case WAIT_SPIN_EVENT:
                    while (!done && (now_s() - tc)*1e6 < spin_us) {
                        done = [ev signaledValue] >= v;
                    }
                    if (!done) [ev waitUntilSignaledValue:v timeoutMS:10000];
                    break;
                case WAIT_EVENT_BLOCK: [ev waitUntilSignaledValue:v timeoutMS:10000]; break;
                case WAIT_HANDLER: dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER); break;
                default: break;
            }
            if (sem) dispatch_release(sem);
        }
    }
    int n = 0;
    [cbs[iters] waitUntilCompleted];
    for (int k = 0; k <= iters; ++k) {
        [cbs[k] waitUntilCompleted];
        if ([cbs[k] status] != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "pingpong: command buffer failed\n");
            exit(1);
        }
        if (k > 0 && n < 4096) {
            gaps[n++] = ([cbs[k] GPUStartTime] - [cbs[k - 1] GPUEndTime])*1e6;
        }
    }
    for (int k = 0; k <= iters; ++k) {
        [cbs[k] release];
    }
    free(cbs);
    return n;
}

// ------------------------------------------------------------------------------------------------
// (c) CPU -> GPU -> CPU through shared events, the command buffer is committed before the CPU signals
// ------------------------------------------------------------------------------------------------

enum ev_wait { EVW_SPIN_EVENT, EVW_SPIN_MEMORY, EVW_BLOCK_EVENT, EVW_BLOCK_CB, EVW_COUNT };
static const char * evw_names[EVW_COUNT] = { "spin event value", "spin output word", "waitUntilSignaledValue", "waitUntilCompleted" };

struct ev_res {
    double rt[4096], to_gpu_end[4096], from_gpu_end[4096];
    int n, timeouts, stale;
};

static void event_trip(id<MTLCommandQueue> q, id<MTLComputePipelineState> ps, id<MTLBuffer> in, id<MTLBuffer> out,
                       id<MTLSharedEvent> ev_go, id<MTLSharedEvent> ev_done, uint64_t * val, enum ev_wait mode, int pre_us,
                       struct ev_res * r) {
    @autoreleasepool {
        const uint64_t v = ++(*val);
        volatile uint32_t * pin  = (volatile uint32_t *) [in contents];
        volatile uint32_t * pout = (volatile uint32_t *) [out contents];
        pout[0] = 0;

        id<MTLCommandBuffer> cb = [q commandBufferWithUnretainedReferences];
        [cb encodeWaitForEvent:ev_go value:v];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:in offset:0 atIndex:0];
        [enc setBuffer:out offset:0 atIndex:1];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cb encodeSignalEvent:ev_done value:v];
        [cb commit];

        // let the command buffer reach the GPU and park on the event
        const double tp = now_s();
        while ((now_s() - tp)*1e6 < pre_us) { }

        const uint32_t magic = (uint32_t) (v*2654435761u) | 1u;
        pin[0] = magic;
        atomic_thread_fence(memory_order_seq_cst);
        const double ts = now_s();
        ev_go.signaledValue = v;

        bool ok = true;
        switch (mode) {
            case EVW_SPIN_EVENT:
                while ([ev_done signaledValue] < v) {
                    if ((now_s() - ts) > 2.0) { ok = false; break; }
                }
                break;
            case EVW_SPIN_MEMORY:
                // the GPU write may only become visible at the end of the encoder / command buffer
                while (pout[0] != magic + 1u) {
                    if ((now_s() - ts) > 0.05) { ok = false; break; }
                }
                break;
            case EVW_BLOCK_EVENT:
                ok = [ev_done waitUntilSignaledValue:v timeoutMS:2000];
                break;
            case EVW_BLOCK_CB:
                [cb waitUntilCompleted];
                break;
            default: break;
        }
        const double t1 = now_s();
        [cb waitUntilCompleted];
        if ([cb status] != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "event trip: command buffer failed (status %d)\n", (int) [cb status]);
            exit(1);
        }
        if (pout[0] != magic + 1u) {
            // the kernel read a stale input word: the CPU write was not visible after the event wait
            r->stale++;
        }
        if (!ok) {
            r->timeouts++;
            return;
        }
        const double g1 = [cb GPUEndTime];
        if (r->n < 4096) {
            r->rt          [r->n] = (t1 - ts)*1e6;
            r->to_gpu_end  [r->n] = (g1 - ts)*1e6;
            r->from_gpu_end[r->n] = (t1 - g1)*1e6;
            r->n++;
        }
    }
}

int main(int argc, char ** argv) {
    int iters = 200, chain_max = 1024, spin_us = 2000, busy = 256, pre_us = 300, idle_us = 200;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) chain_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) spin_us = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) busy = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) pre_us = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) idle_us = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-n iters=200] [-c chain_max=1024] [-s spin_budget_us=2000] [-w busy_dispatches=256] "
                    "[-p park_us=300] [-i idle_between_round_trips_us=200]\n", argv[0]);
            return 1;
        }
    }
    if (iters > 4096) iters = 4096;
    mach_timebase_info(&g_tb);

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            fprintf(stderr, "no Metal device\n");
            return 1;
        }
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:k_src] options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "library: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLFunction> f_step = [lib newFunctionWithName:@"k_step"];
        id<MTLFunction> f_echo = [lib newFunctionWithName:@"k_echo"];
        id<MTLComputePipelineState> ps_step = [dev newComputePipelineStateWithFunction:f_step error:&err];
        id<MTLComputePipelineState> ps_echo = [dev newComputePipelineStateWithFunction:f_echo error:&err];
        if (!ps_step || !ps_echo) {
            fprintf(stderr, "pipeline: %s\n", [[err localizedDescription] UTF8String]);
            return 1;
        }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        id<MTLBuffer> buf  = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bin  = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [dev newBufferWithLength:4096 options:MTLResourceStorageModeShared];
        memset([buf contents], 0, 1 << 20);
        id<MTLSharedEvent> ev      = [dev newSharedEvent]; // (b), (b2): end-of-command-buffer signal
        id<MTLSharedEvent> ev_go   = [dev newSharedEvent]; // (c): CPU -> GPU
        id<MTLSharedEvent> ev_done = [dev newSharedEvent]; // (c): GPU -> CPU
        uint64_t ev_val = 0, evc_val = 0;

        printf("device: %s, iters %d, spin budget %d us, busy CB %d dispatches, park %d us, idle between round trips %d us\n",
               [[dev name] UTF8String], iters, spin_us, busy, pre_us, idle_us);

        // ---------------- (a)
        printf("\n(a) dependent tiny dispatches in one command buffer (GPU time per dispatch, slope between N=1 and N)\n");
        printf("%-34s %8s %8s %12s %14s\n", "mode", "threads", "N", "gpu_us(med)", "us/dispatch");
        static const char * cnames[3] = { "serial encoder", "concurrent + buffer barrier", "one encoder per dispatch" };
        const int thr_list[2] = { 32, 4096 };
        double a_us_barrier = 0;
        for (int m = 0; m < 3; ++m) {
            for (int ti = 0; ti < 2; ++ti) {
                double base[64];
                const int reps = 15;
                for (int k = 0; k < reps; ++k) base[k] = chain_gpu_us(q, ps_step, buf, 1, thr_list[ti], (enum chain_mode) m);
                const double b1 = stats(base, reps).med;
                for (int n = 64; n <= chain_max; n *= 4) {
                    double v[64];
                    for (int k = 0; k < reps; ++k) v[k] = chain_gpu_us(q, ps_step, buf, n, thr_list[ti], (enum chain_mode) m);
                    const double med = stats(v, reps).med;
                    const double per = (med - b1) / (n - 1);
                    printf("%-34s %8d %8d %12.1f %14.3f\n", cnames[m], thr_list[ti], n, med, per);
                    if (m == CHAIN_CONCURRENT_BARRIER && ti == 0) a_us_barrier = per;
                }
            }
        }

        // ---------------- (b)
        printf("\n(b) command buffer round trip (encode + commit + wait), us; submit = GPUStartTime - commit, wake = return - GPUEndTime\n");
        printf("%-26s %-6s %9s %9s %9s %9s %9s %9s %10s\n", "wait", "work", "total_med", "total_p10", "total_p90", "submit", "gpu", "wake", "spin_miss");
        struct rt_res * rr = calloc(1, sizeof(struct rt_res));
        double b_block_empty = 0, b_best_empty = 1e30, b_block_busy_wake = 0, b_best_busy_wake = 1e30;
        const char * b_best_name = "";
        const int works[3] = { 0, 1, busy };
        for (int wi = 0; wi < 3; ++wi) {
            for (int m = 0; m < WAIT_COUNT; ++m) {
                memset(rr, 0, sizeof(*rr));
                for (int k = 0; k < 10; ++k) round_trip(q, ps_step, buf, ev, &ev_val, works[wi], (enum wait_mode) m, spin_us, rr);
                memset(rr, 0, sizeof(*rr));
                for (int k = 0; k < iters; ++k) {
                    round_trip(q, ps_step, buf, ev, &ev_val, works[wi], (enum wait_mode) m, spin_us, rr);
                    if (idle_us > 0) usleep(idle_us); // independent samples: the GPU and the host threads go idle in between
                }
                const struct stat5 t = stats(rr->total, rr->n);
                const struct stat5 s = stats(rr->submit, rr->n);
                const struct stat5 g = stats(rr->gpu, rr->n);
                const struct stat5 w = stats(rr->wake, rr->n);
                char wl[16];
                snprintf(wl, sizeof(wl), "%d", works[wi]);
                printf("%-26s %-6s %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %10d\n", wait_names[m], wl, t.med, t.p10, t.p90, s.med, g.med, w.med, rr->spin_miss);
                if (wi == 0) {
                    if (m == WAIT_BLOCK) b_block_empty = t.med;
                    if (t.med < b_best_empty) { b_best_empty = t.med; b_best_name = wait_names[m]; }
                }
                if (wi == 2) {
                    if (m == WAIT_BLOCK) b_block_busy_wake = w.med;
                    if (w.med < b_best_busy_wake) b_best_busy_wake = w.med;
                }
            }
        }
        free(rr);

        // ---------------- (b2)
        printf("\n(b2) back-to-back command buffers of %d dispatches, next commit right after the wait returns: GPU idle gap (us)\n", busy);
        printf("%-26s %9s %9s %9s %9s\n", "wait", "gap_med", "gap_p10", "gap_p90", "gap_mean");
        double * gaps = calloc(4096, sizeof(double));
        double b2_block = 0, b2_best = 1e30;
        const char * b2_best_name = "";
        for (int m = 0; m < WAIT_COUNT; ++m) {
            pingpong(q, ps_step, buf, ev, &ev_val, busy, (enum wait_mode) m, spin_us, 10, gaps);
            const int n = pingpong(q, ps_step, buf, ev, &ev_val, busy, (enum wait_mode) m, spin_us, iters, gaps);
            const struct stat5 g = stats(gaps, n);
            printf("%-26s %9.1f %9.1f %9.1f %9.1f\n", wait_names[m], g.med, g.p10, g.p90, g.mean);
            if (m == WAIT_BLOCK) b2_block = g.med;
            if (g.med < b2_best) { b2_best = g.med; b2_best_name = wait_names[m]; }
        }
        free(gaps);

        // ---------------- (c)
        printf("\n(c) CPU -> GPU -> CPU via MTLSharedEvent, command buffer committed and parked on encodeWaitForEvent (%d us before the signal)\n", pre_us);
        printf("    rt = CPU sets signaledValue -> CPU sees the result; to_gpu_end = signal -> GPUEndTime; from_gpu_end = GPUEndTime -> CPU sees it\n");
        printf("%-26s %9s %9s %9s %12s %14s %9s %6s\n", "cpu wait", "rt_med", "rt_p10", "rt_p90", "to_gpu_end", "from_gpu_end", "timeouts", "stale");
        struct ev_res * er = calloc(1, sizeof(struct ev_res));
        double c_best = 1e30;
        for (int m = 0; m < EVW_COUNT; ++m) {
            memset(er, 0, sizeof(*er));
            for (int k = 0; k < 10; ++k) event_trip(q, ps_echo, bin, bout, ev_go, ev_done, &evc_val, (enum ev_wait) m, pre_us, er);
            memset(er, 0, sizeof(*er));
            for (int k = 0; k < iters; ++k) {
                event_trip(q, ps_echo, bin, bout, ev_go, ev_done, &evc_val, (enum ev_wait) m, pre_us, er);
            }
            const struct stat5 t = stats(er->rt, er->n);
            const struct stat5 a = stats(er->to_gpu_end, er->n);
            const struct stat5 b = stats(er->from_gpu_end, er->n);
            printf("%-26s %9.1f %9.1f %9.1f %12.1f %14.1f %9d %6d\n", evw_names[m], t.med, t.p10, t.p90, a.med, b.med, er->timeouts, er->stale);
            if (er->n > 0 && t.med < c_best) c_best = t.med;
        }
        free(er);

        // ---------------- decision
        // the cost a host round trip adds to a decode token is the GPU bubble between two command buffers (b2, blocking wait,
        // as decode does today); an event-chained token replaces it with the CPU<->GPU event round trip (c)
        printf("\nsummary: dependent dispatch %.2f us (concurrent+barrier, 32 thr); empty CB round trip %.1f us blocking, best %.1f us (%s); "
               "wake after a busy CB %.1f us blocking, best %.1f us; GPU bubble per round trip %.1f us blocking, best %.1f us (%s); "
               "event CPU<->GPU round trip best %.1f us\n",
               a_us_barrier, b_block_empty, b_best_empty, b_best_name, b_block_busy_wake, b_best_busy_wake, b2_block, b2_best, b2_best_name, c_best);
        printf("decision: host round trip (GPU bubble, blocking wait) %.1f us %s 40 us -> event-chained GPU-resident decode token %s; "
               "~40 round trips/token: %.2f ms/token now, %.2f ms/token with the best polling wait, %.2f ms/token with event chaining\n",
               b2_block, b2_block >= 40.0 ? ">=" : "<", b2_block >= 40.0 ? "WORTH building" : "not worth building",
               40*b2_block*1e-3, 40*b2_best*1e-3, 40*c_best*1e-3);

        [ev release]; [ev_go release]; [ev_done release];
        [buf release]; [bin release]; [bout release];
        [q release];
        [ps_step release]; [ps_echo release];
        [f_step release]; [f_echo release];
        [lib release];
        [dev release];
    }
    return 0;
}
