#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy;  // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;

    // GGML_METAL_OP_PROFILE=1: per-op GPU time attribution (see ggml_metal_prof_*)
    struct ggml_metal_prof * prof;
};


// ---------------------------------------------------------------------------------------------------------
// GGML_METAL_OP_PROFILE=1
//
// The graph is encoded on the calling thread into many small command buffers, one per run of consecutive nodes
// that fall in the same bucket (MUL_MAT/<type>, MUL_MAT_ID/<type>, FLASH_ATTN_EXT, GET_ROWS, <op name>).
// GPUStartTime/GPUEndTime of every command buffer are read once it has completed (at the next synchronize or
// when a later graph_compute finds it finished). Command buffers execute in queue order, so the per-bucket GPU
// time is the sum of (end - start) of its command buffers, and the gaps between them are tracked separately.
// Nodes are never fused across a run boundary, so profiling mode perturbs fusion and concurrency slightly.
// ---------------------------------------------------------------------------------------------------------

#define GGML_METAL_PROF_MAX_BUCKETS 256
#define GGML_METAL_PROF_DECODE_MAX_TOKENS 16
#define GGML_METAL_PROF_MAX_RUN_NODES 32

struct ggml_metal_prof_bucket {
    char     name[40];
    uint64_t calls;  // graph nodes
    uint64_t runs;   // command buffers
    uint64_t bytes;  // src0 bytes read (estimate)
    double   gpu_s;
};

struct ggml_metal_prof_cls {
    struct ggml_metal_prof_bucket b[GGML_METAL_PROF_MAX_BUCKETS];
    int      nb;
    uint64_t graphs;
    uint64_t tokens;      // tokens of graphs that produce logits
    uint64_t cbs;
    double   cpu_s;       // wall time inside ggml_metal_graph_compute
    double   encode_s;    // part of cpu_s spent encoding nodes
    double   gpu_sum_s;   // sum of per-cb GPU durations
    double   gap_intra_s; // idle between consecutive cbs of one graph
    double   gap_inter_s; // idle between the last cb of one graph and the first of the next
};

struct ggml_metal_prof_pending {
    id<MTLCommandBuffer> cb;
    int      cls;
    int      bucket;
    bool     first;
};

struct ggml_metal_prof {
    struct ggml_metal_prof_cls cls[2]; // 0 = decode (<= 16 tokens), 1 = prefill
    struct ggml_metal_prof_pending * pend;
    int    n_pend, cap_pend;
    double prev_end;
    double sync_wait_s;
    uint64_t n_sync;
};

static int ggml_metal_prof_bucket_id(struct ggml_metal_prof_cls * c, const char * name) {
    for (int i = 0; i < c->nb; ++i) {
        if (strcmp(c->b[i].name, name) == 0) {
            return i;
        }
    }
    int i = c->nb < GGML_METAL_PROF_MAX_BUCKETS ? c->nb++ : GGML_METAL_PROF_MAX_BUCKETS - 1;
    if (i == c->nb - 1 && c->b[i].name[0] == 0) {
        snprintf(c->b[i].name, sizeof(c->b[i].name), "%s", name);
    }
    return i;
}

static bool ggml_metal_prof_is_noop(const struct ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

static void ggml_metal_prof_node_name(const struct ggml_tensor * t, char * buf, size_t n) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            snprintf(buf, n, "%s/%s", ggml_op_name(t->op), ggml_type_name(t->src[0]->type));
            break;
        default:
            snprintf(buf, n, "%s", ggml_op_desc(t));
            break;
    }
}

static uint64_t ggml_metal_prof_node_bytes(const struct ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
            return ggml_nbytes(t->src[0]);
        case GGML_OP_MUL_MAT_ID: {
            // experts selected: ne20 per token, at most all of them
            const uint64_t sel = (uint64_t) t->src[2]->ne[0] * t->src[2]->ne[1];
            const uint64_t all = (uint64_t) t->src[0]->ne[2];
            return t->src[0]->nb[2] * (sel < all ? sel : all);
        }
        case GGML_OP_GET_ROWS:
            return (uint64_t) t->src[1]->ne[0] * t->src[1]->ne[1] * t->src[1]->ne[2] * t->src[0]->nb[1];
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_nbytes(t->src[1]) + (t->src[2] ? ggml_nbytes(t->src[2]) : 0);
        default:
            return t->src[0] ? ggml_nbytes(t->src[0]) : 0;
    }
}

static void ggml_metal_prof_account(struct ggml_metal_prof * p, struct ggml_metal_prof_pending * e) {
    const double t0 = [e->cb GPUStartTime];
    const double t1 = [e->cb GPUEndTime];
    struct ggml_metal_prof_cls * c = &p->cls[e->cls];
    const double d = t1 > t0 ? t1 - t0 : 0.0;

    c->b[e->bucket].gpu_s += d;
    c->gpu_sum_s += d;
    if (p->prev_end > 0.0 && t0 > p->prev_end) {
        if (e->first) c->gap_inter_s += t0 - p->prev_end; else c->gap_intra_s += t0 - p->prev_end;
    }
    if (t1 > p->prev_end) {
        p->prev_end = t1;
    }
}

// account (and release) the pending command buffers; if all == false only the leading completed ones
static void ggml_metal_prof_drain(struct ggml_metal_prof * p, bool all) {
    int i = 0;
    while (i < p->n_pend) {
        if (!all && [p->pend[i].cb status] != MTLCommandBufferStatusCompleted) {
            break;
        }
        if ([p->pend[i].cb status] == MTLCommandBufferStatusCompleted) {
            ggml_metal_prof_account(p, &p->pend[i]);
        }
        [p->pend[i].cb release];
        ++i;
    }
    if (i > 0) {
        memmove(p->pend, p->pend + i, (p->n_pend - i) * sizeof(p->pend[0]));
        p->n_pend -= i;
    }
}

static void ggml_metal_prof_print(struct ggml_metal_prof * p) {
    static const char * names[2] = { "decode", "prefill" };
    for (int k = 0; k < 2; ++k) {
        struct ggml_metal_prof_cls * c = &p->cls[k];
        if (c->graphs == 0) {
            continue;
        }
        fprintf(stderr, "ggml_metal_profile: class=%s graphs=%" PRIu64 " tokens=%" PRIu64 " cbs=%" PRIu64
                " cpu_graph_compute_ms=%.3f encode_ms=%.3f gpu_sum_ms=%.3f gap_intra_ms=%.3f gap_inter_ms=%.3f\n",
                names[k], c->graphs, c->tokens, c->cbs, c->cpu_s*1e3, c->encode_s*1e3, c->gpu_sum_s*1e3,
                c->gap_intra_s*1e3, c->gap_inter_s*1e3);
        for (int i = 0; i < c->nb; ++i) {
            if (c->b[i].calls == 0) {
                continue;
            }
            fprintf(stderr, "ggml_metal_profile: class=%s bucket=%s calls=%" PRIu64 " runs=%" PRIu64 " gpu_ms=%.3f src0_bytes=%" PRIu64 "\n",
                    names[k], c->b[i].name, c->b[i].calls, c->b[i].runs, c->b[i].gpu_s*1e3, c->b[i].bytes);
        }
    }
    fprintf(stderr, "ggml_metal_profile: sync_wait_ms=%.3f syncs=%" PRIu64 "\n", p->sync_wait_s*1e3, p->n_sync);
}

static enum ggml_status ggml_metal_prof_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    struct ggml_metal_prof * p = ctx->prof;
    const int64_t t_start = ggml_time_us();

    ggml_metal_prof_drain(p, false);

    // classify the graph: decode-sized if the largest mat-mul batch is <= 16 tokens
    int n_tok = 1;
    for (int i = 0; i < gf->n_nodes; ++i) {
        const struct ggml_tensor * t = gf->nodes[i];
        if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[1]) {
            const int n = (int) (t->op == GGML_OP_MUL_MAT ? t->src[1]->ne[1] : t->src[1]->ne[2]);
            if (n > n_tok) n_tok = n;
        }
    }
    const int k = n_tok <= GGML_METAL_PROF_DECODE_MAX_TOKENS ? 0 : 1;
    struct ggml_metal_prof_cls * c = &p->cls[k];
    c->graphs++;
    if (gf->nodes[gf->n_nodes - 1]->name[0] && strcmp(gf->nodes[gf->n_nodes - 1]->name, "result_output") == 0) {
        c->tokens += n_tok;
    }

    ggml_metal_device_rsets_keep_alive(ctx->dev);

    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

    int64_t t_enc = 0;
    bool first = true;

    @autoreleasepool {
        ctx->gf = gf;

        int i0 = 0;
        while (i0 < gf->n_nodes) {
            // the run starts at the first dispatching node; no-op nodes before it are carried along
            int i1 = i0;
            int key = -1;
            char name[40];
            int n_disp = 0;
            uint64_t bytes = 0;
            while (i1 < gf->n_nodes && i1 - i0 < GGML_METAL_PROF_MAX_RUN_NODES) {
                const struct ggml_tensor * t = gf->nodes[i1];
                if (!ggml_metal_prof_is_noop(t)) {
                    ggml_metal_prof_node_name(t, name, sizeof(name));
                    const int id = ggml_metal_prof_bucket_id(c, name);
                    if (key < 0) {
                        key = id;
                    } else if (id != key) {
                        break;
                    }
                    n_disp++;
                    bytes += ggml_metal_prof_node_bytes(t);
                }
                i1++;
            }
            if (key < 0) {
                // only no-ops left
                key = ggml_metal_prof_bucket_id(c, "NOOP");
            }

            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];
            [cmd_buf enqueue];

            const int64_t te0 = ggml_time_us();
            ggml_metal_op_t ctx_op = ggml_metal_op_init(
                ctx->dev, cmd_buf, gf, ctx->finfo, i0, i1, ctx->use_concurrency, false, ctx->debug_graph);
            for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                const int res = ggml_metal_op_encode(ctx_op, idx);
                if (res == 0) {
                    break;
                }
                idx += res - 1;
            }
            ggml_metal_op_free(ctx_op);
            [cmd_buf commit];
            t_enc += ggml_time_us() - te0;

            c->b[key].calls += n_disp;
            c->b[key].runs++;
            c->b[key].bytes += bytes;
            c->cbs++;

            if (p->n_pend == p->cap_pend) {
                p->cap_pend = p->cap_pend ? 2*p->cap_pend : 4096;
                p->pend = realloc(p->pend, p->cap_pend * sizeof(p->pend[0]));
            }
            p->pend[p->n_pend++] = (struct ggml_metal_prof_pending) { cmd_buf, k, key, first };
            first = false;

            ctx->cmd_buf_last = cmd_buf;
            i0 = i1;
        }
    }

    c->encode_s += t_enc * 1e-6;
    c->cpu_s    += (ggml_time_us() - t_start) * 1e-6;

    return GGML_STATUS_SUCCESS;
}

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
        if (queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

        {
            const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        res->use_graph_optimize = true;

        if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;

        res->prof = nil;
        {
            const char * val = getenv("GGML_METAL_OP_PROFILE");
            if (val && atoi(val) != 0) {
                res->prof = calloc(1, sizeof(struct ggml_metal_prof));
                fprintf(stderr, "ggml_metal_profile: enabled (per-run command buffers, single-threaded encode)\n");
            }
        }

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->buf_refs     = [[NSMutableArray alloc] init];

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    if (ctx->prof) {
        if (ctx->cmd_buf_last) {
            [ctx->cmd_buf_last waitUntilCompleted];
        }
        ggml_metal_prof_drain(ctx->prof, true);
        ggml_metal_prof_print(ctx->prof);
        free(ctx->prof->pend);
        free(ctx->prof);
        ctx->prof = nil;
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        const int64_t t_w = ctx->prof ? ggml_time_us() : 0;
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
        if (ctx->prof) {
            ctx->prof->sync_wait_s += (ggml_time_us() - t_w) * 1e-6;
            ctx->prof->n_sync++;
            ggml_metal_prof_drain(ctx->prof, false);
        }
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_dst];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ggml_metal_device_get_queue(ctx_dst->dev);
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    if (ctx->prof) {
        return ggml_metal_prof_graph_compute(ctx, gf);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        if (ctx->n_cb == 0) {
            // single-threaded encoding: the whole graph is encoded by one command buffer
            ctx->n_nodes_0      = gf->n_nodes;
            ctx->n_nodes_1      = 0;
            ctx->n_nodes_per_cb = 0;
        } else {
            ctx->n_nodes_0      = MIN(n_main, gf->n_nodes);

            // DS41 GPU-resident MoE arena (GGML_OP_DSV41_EVENT): end the first command buffer just after a wait node, so
            // that no layer's router, slot lookup and GPU->host signal straddle the two command buffers (the host reads
            // what the signal's own command buffer computed before it). Only with n_cb == 1 (the default)
            if (ctx->n_cb == 1 && ggml_metal_device_dsv41_ready(ctx->dev)) {
                for (int i = ctx->n_nodes_0; i < gf->n_nodes; ++i) {
                    const struct ggml_tensor * node = gf->nodes[i];
                    if (node->op == GGML_OP_DSV41_EVENT && ((const int32_t *) node->op_params)[0] == GGML_DSV41_EVENT_WAIT) {
                        ctx->n_nodes_0 = i + 1;
                        break;
                    }
                }
            }

            ctx->n_nodes_1      = gf->n_nodes - ctx->n_nodes_0;

            ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute == 0,
            ctx->debug_graph);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
