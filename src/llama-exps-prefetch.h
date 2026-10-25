#pragma once

// DeepSeek-V4.1 routed-expert prefetch (LLAMA_DSV41_EXPS_PREFETCH=1, default off)
//
// The routed experts stay in the mmap'd GGUF and are read by the CPU MUL_MAT_ID through page faults
// (synchronous 16 KiB faults at queue depth <= n_threads). Once the router ids reach the CPU split we know
// exactly which expert slices the layer will read, so a graph op issues F_RDADVISE for the missing pages of
// those slices from a small background pool: the reads go to the same page cache that backs the mapping
// (outputs are bit-identical), at high queue depth, while the matmuls start right away.
//
// env:
//   LLAMA_DSV41_EXPS_PREFETCH=1         enable (graph unchanged when off)
//   LLAMA_DSV41_PREFETCH_NEXT=1         also prefetch the next MoE layer's full expert tensors in prefill (default off: it thrashes the page cache, pp512 5.96 vs 12.2 tok/s)
//   LLAMA_DSV41_PREFETCH_NEXT_MIN=N     n_tokens threshold for the next-layer prefetch (default 32)
//   LLAMA_DSV41_PREFETCH_THREADS=N      advise threads (default 4)
//   LLAMA_DSV41_PREFETCH_STATS=1        print page-cache hit rate and I/O counters when the model is freed
//
// Expert uploader (used by llama_context::sched_copy_experts when op offload runs MUL_MAT_ID on Metal):
// with LLAMA_DSV41_EXPS_OFFLOAD=1 the router-selected expert runs are pread() by a pool straight into the shared
// (CPU-addressable) Metal compute buffer instead of newBufferWithBytes memcpy + blit. Same bytes, lossless.
//   LLAMA_DSV41_UPLOAD_PREAD=0          disable the pread uploader (fall back to ggml_backend_tensor_set_async)
//   LLAMA_DSV41_UPLOAD_THREADS=N        upload threads (default 12)
//   LLAMA_DSV41_UPLOAD_CHUNK_KIB=N      pread job size (default 2048)
//   LLAMA_DSV41_PREFETCH_OFFLOADED=0    skip the batch prefetch node for offloaded (n_tokens >= offload min batch) ubatches

#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;

namespace llama_exps_prefetch {

bool enabled();

// address range [addr, addr + len) maps file fd from offset 0; fd is dup'ed and owned by the registry
void register_mapping(const void * owner, const void * addr, size_t len, int fd);

// drops the owner's mappings and layers (waits for its in-flight advises), prints the stats if requested
void unregister_owner(const void * owner);

// remembers the expert tensors of layer il (for the next-layer prefetch) and returns the op userdata
void * layer_userdata(const void * owner, int il, ggml_tensor * const * exps, int n_exps);

// ggml custom op: src[0] = selected experts I32 [n_expert_used, n_tokens], src[1..] = expert weights [.., .., n_expert]
void op(ggml_tensor * dst, int ith, int nth, void * userdata);

// true when the mappings must be registered for the uploader (deepseek41 + LLAMA_DSV41_EXPS_OFFLOAD=1 + UPLOAD_PREAD != 0)
bool upload_enabled();

// file descriptor and file offset of [addr, addr + len) if it lies inside one registered mapping
bool lookup(const void * addr, size_t len, int * fd, uint64_t * file_off);
// same, but the fd was opened with F_NOCACHE (falls back to the cached fd): for reads whose bytes are kept elsewhere
// (the MoE slot arena), so they do not also occupy the page cache
bool lookup_nocache(const void * addr, size_t len, int * fd, uint64_t * file_off);

struct upload_job {
    int          fd;
    uint64_t     file_off;
    void       * dst;
    size_t       len;
    const void * src; // the same bytes in the mmap (memcpy fallback if pread fails)
};

// splits [file_off, file_off + len) into chunk-sized jobs and appends them
void add_upload(std::vector<upload_job> & jobs, int fd, uint64_t file_off, void * dst, size_t len, const void * src);

// runs the jobs on the upload pool (the caller helps) and returns when all bytes are in place
void upload(const std::vector<upload_job> & jobs);

// asynchronous upload on the same pool as upload(): upload_begin() starts the workers and returns, upload_end() helps
// and returns when all bytes are in place. The jobs vector must stay alive and unchanged until upload_end().
void upload_begin(const std::vector<upload_job> & jobs);
void upload_end();

// same as upload() on a separate pool (LLAMA_DSV41_STAGE_THREADS, default 8), callable from one background thread
// concurrently with upload() on the scheduler thread
void upload_bg(const std::vector<upload_job> & jobs);

// time the caller spent in ggml_backend_synchronize before an upload (stats only)
void upload_add_sync_us(uint64_t us);

} // namespace llama_exps_prefetch
