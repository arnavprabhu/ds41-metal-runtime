#include "common.h"
#include "dequantize.h"

// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];

constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// pad the last chunk of C elements of k and v into a an extra pad buffer
kernel void kernel_flash_attn_ext_pad(
        constant ggml_metal_kargs_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// scan the blocks of the mask that are not masked
// 0 -     masked (i.e. full of -INF, skip)
// 1 - not masked (i.e. at least one element of the mask is not -INF)
// 2 - all zero
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 || i1*Q + Q > args.ne31 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    // detailed check of the elements of the block
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}
// compress the finite entries of each KQ mask row into a list of KV indices (ascending order),
// padded with -1 up to n_kv_max_padded (a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG)
// one threadgroup per mask row; the mask remains the single source of truth for the values
kernel void kernel_flash_attn_ext_vec_idx(
        constant ggml_metal_kargs_flash_attn_ext_vec_idx & args,
        device const half * mask,
        device       int  * idx,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NLOCAL = 32; // max finite positions kept in registers per thread

    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    device const half * pm  = (device const half *) ((device const char *) mask + i1*args.nb31 + i2*args.nb32 + i3*args.nb33);
    device int * pidx = idx + (((int64_t)i3*args.ne32 + i2)*args.ne31 + i1)*args.n_kv_max_padded;

    const int n  = args.ne30;
    const int q  = n/ntg.x;
    const int r  = n%ntg.x;

    // each thread handles a contiguous slice of the mask row
    const int r0 = q*tiitg + min((int) tiitg, r);
    const int r1 = r0 + q + (tiitg < r ? 1 : 0);

    // count the finite entries in the slice and keep their positions in registers (single mask read)
    int cnt = 0;  // total finite entries in the slice
    int nloc = 0; // finite entries kept in registers
    int local[NLOCAL];
    for (int i = r0; i < r1; ++i) {
        if (isfinite((float) pm[i])) {
            if (nloc < NLOCAL) {
                local[nloc] = i;
                nloc++;
            }
            cnt++;
        }
    }

    const short sgitg = tiitg/NW;
    const short tiisg = tiitg%NW;

    threadgroup int tcount[8];

    // simd_sum is a collective: all lanes must evaluate it
    const int sg_sum = simd_sum(cnt);
    if (tiisg == 0) {
        tcount[sgitg] = sg_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total = 0;
    for (short s = 0; s < ntg.x/NW; ++s) {
        total += tcount[s];
    }

    // base offset of this thread's slice in the output list (exclusive scan within the simdgroup)
    int sg_base = 0;
    for (short s = 0; s < sgitg; ++s) {
        sg_base += tcount[s];
    }

    // exclusive prefix scan of the per-thread counts within the simdgroup
    int incl = cnt;
    for (int d = 1; d < NW; d <<= 1) {
        const int v = simd_shuffle_up(incl, d);
        if (tiisg >= d) {
            incl += v;
        }
    }
    const int base = sg_base + (incl - cnt);

    // write the finite positions in order; if the hint is violated, keep only the first n_kv_max entries
    int j = 0;
    for (; j < nloc && base + j < args.n_kv_max; ++j) {
        pidx[base + j] = local[j];
    }

    // a dense mask may have more than NLOCAL finite entries in a slice; re-read the mask to write the rest
    if (cnt > nloc && base + nloc < args.n_kv_max) {
        int j2 = 0;
        for (int i = r0; i < r1; ++i) {
            if (isfinite((float) pm[i])) {
                if (j2 >= nloc) {
                    pidx[base + j2] = i;
                }
                j2++;
                if (base + j2 >= args.n_kv_max) {
                    break;
                }
            }
        }
    }

    // pad the tail of the list with -1
    const int count = min(total, args.n_kv_max);
    for (int i = count + tiitg; i < args.n_kv_max_padded; i += ntg.x) {
        pidx[i] = -1;
    }
}

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

constant short FC_lightning_indexer_nh [[function_constant(FC_LIGHTNING_INDEXER + 0)]];

template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
kernel void kernel_lightning_indexer(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
    const     short NH    = FC_lightning_indexer_nh;
    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;

    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
    constexpr short NTG = 32*NSG;    // threads per threadgroup

    const int i_stream = tgpig.z;
    const int i_kv_0   = tgpig.x*NK;            // first key of this threadgroup
    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup

    threadgroup half sk[NK * DK16 * 16];
    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;

    for (short i = tiitg; i < NK*DK16; i += NTG) {
        const short ik  = i/DK16;
        const short i16 = i%DK16;

        half4x4 tmp;

        if (i_kv_0 + ik < args.n_kv) {
            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);

            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
        } else {
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                tmp[j] = half4(0.0h);
            }
        }

        sk4x4[i] = tmp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K tile of this simdgroup, transposed to [DK, NKPSG]
    simdgroup_half8x8 mk[DK8];

    FOR_UNROLL (short i = 0; i < DK8; ++i) {
        simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
    }

    threadgroup half4   sq4[NHPTG*DK4];
    threadgroup half  * sq = (threadgroup half *) sq4;

    threadgroup float sw [NHPTG];
    threadgroup float sqk[NSG*NHPTG*NKPSG];

    const int i_batch_0 = tgpig.y*NBPTG;
    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);

    for (short ib = 0; ib < n_batch; ++ib) {
        const int i_batch = i_batch_0 + ib;

        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;

        float score = 0.0f;

        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights, heads past NH are zero
            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
                const short ih = i/DK4;
                const short i4 = i%DK4;

                if (i_head + ih < NH) {
                    device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);

                    sq4[ih*DK4 + i4] = half4(q4[i4]);
                } else {
                    sq4[ih*DK4 + i4] = half4(0.0h);
                }
            }

            if (tiitg < NHPTG) {
                sw[tiitg] = i_head + tiitg < NH ? ((device const float *) pw)[i_head + tiitg] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            FOR_UNROLL (short i = 0; i < DK8; ++i) {
                simdgroup_half8x8 mq;

                simdgroup_load(mq, sq + 8*i, DK, 0, false);
                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
            }

            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;

            simdgroup_store(mqk, pqk, NKPSG, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
            if (tiisg < NKPSG) {
                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiisg < NKPSG) {
            const int ik = i_kv + tiisg;
            if (ik < args.n_kv) {
                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);

                pd[ik] = score + (float) pm[ik];
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_t;

template [[host_name("kernel_lightning_indexer_f32")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32>;
template [[host_name("kernel_lightning_indexer_f16")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16>;
#endif

template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0>;


// deepseek41 sparse prefill attention (GGML_METAL_DSV4_SATTN=1): 64 query heads over one KV head (K == V),
// dk = dv = 512, each query row attends to the finite entries of its mask row (window + indexer top-k), given as
// the per-row index lists of kernel_flash_attn_ext_vec_idx. One threadgroup per (row, 16 heads), 8 simdgroups that
// each own 64 of the 512 dims. Per block of 16 keys (gathered once into threadgroup memory, used as K and as V):
// partial Q.K^T per simdgroup on the matrix units, reduced in threadgroup memory, online softmax in f32 (P in
// half), O[16 x 64] += P[16 x 16] . V[16 x 64] per simdgroup. Attention sinks are folded in at the end.
#define DSA_D   512
#define DSA_NH  16
#define DSA_NSG 8
#define DSA_KB  16
#define DSA_DS  (DSA_D/DSA_NSG)

kernel void kernel_dsv4_sparse_attn_f16(
        constant ggml_metal_kargs_dsv4_sparse_attn & args,
        device const char  * q,
        device const char  * k,
        device const char  * mask,
        device const float * sinks,
        device const int   * idx,
        device       float * dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int row = tgpig.x;
    const int h0  = tgpig.y*DSA_NH;
    const int d0  = sgitg*DSA_DS;

    threadgroup half  * sk    = (threadgroup half  *) shmem;                          // [KB][D] keys/values; Q staging [NH][D]
    threadgroup float * sp    = (threadgroup float *) (shmem + DSA_KB*DSA_D*2);         // [NSG][NH][KB] partial scores
    threadgroup half  * spp   = (threadgroup half  *) (sp + DSA_NSG*DSA_NH*DSA_KB);     // [NH][KB] probabilities
    threadgroup float * sdg   = (threadgroup float *) (spp + DSA_NH*DSA_KB);            // [2][8][8] diagonal row scales
    threadgroup float * sm    = sdg + 2*64;                                             // [NH] running max
    threadgroup float * sl    = sm + DSA_NH;                                            // [NH] running sum
    threadgroup float * smk   = sl + DSA_NH;                                            // [KB] mask of the block

    // stage Q (f32 -> half) for the 16 heads
    for (int i = tiitg; i < DSA_NH*DSA_D; i += 256) {
        const int h = i / DSA_D;
        const int d = i % DSA_D;
        device const float * pq = (device const float *) (q + (uint64_t) row*args.nb01 + (uint64_t) (h0 + h)*args.nb02);
        sk[i] = (half) pq[d];
    }
    if (tiitg < DSA_NH) {
        sm[tiitg] = -INFINITY;
        sl[tiitg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8  qf[2][8];
    simdgroup_float8x8 o [2][8];
    for (short hi = 0; hi < 2; ++hi) {
        for (short di = 0; di < 8; ++di) {
            simdgroup_load(qf[hi][di], sk + (hi*8)*DSA_D + d0 + di*8, DSA_D);
            o[hi][di] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const int  * pidx_row = idx + (int64_t) row*args.n_kv_max_padded;
    device const half * pm_row   = (device const half *) (mask + (uint64_t) row*args.nb31);

    const short tk = tiitg / 16; // key of this thread in the block load
    const short tp = tiitg % 16; // 32-element part of that key

    for (int kb = 0; kb < args.n_kv_max_padded; kb += DSA_KB) {
        if (pidx_row[kb] < 0) {
            break; // lists are compacted: the rest is padding
        }

        // gather the block: 16 keys x 512 halfs, 32 halfs per thread
        {
            const int id = pidx_row[kb + tk];
            threadgroup half4 * dk4 = (threadgroup half4 *) (sk + tk*DSA_D + tp*32);
            if (id >= 0) {
                device const half4 * sk4 = (device const half4 *) (k + (uint64_t) id*args.nb11 + tp*64);
                for (short j = 0; j < 8; ++j) {
                    dk4[j] = sk4[j];
                }
            } else {
                for (short j = 0; j < 8; ++j) {
                    dk4[j] = half4(0.0h);
                }
            }
            if (tiitg < DSA_KB) {
                const int idm = pidx_row[kb + tiitg];
                smk[tiitg] = idm >= 0 ? (float) pm_row[idm] : -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // partial scores over this simdgroup's 64 dims
        {
            simdgroup_float8x8 s[2][2];
            for (short hi = 0; hi < 2; ++hi) {
                for (short ki = 0; ki < 2; ++ki) {
                    s[hi][ki] = make_filled_simdgroup_matrix<float, 8>(0.0f);
                }
            }
            for (short di = 0; di < 8; ++di) {
                for (short ki = 0; ki < 2; ++ki) {
                    simdgroup_half8x8 kt;
                    simdgroup_load(kt, sk + (ki*8)*DSA_D + d0 + di*8, DSA_D, ulong2(0, 0), true);
                    for (short hi = 0; hi < 2; ++hi) {
                        simdgroup_multiply_accumulate(s[hi][ki], qf[hi][di], kt, s[hi][ki]);
                    }
                }
            }
            for (short hi = 0; hi < 2; ++hi) {
                for (short ki = 0; ki < 2; ++ki) {
                    simdgroup_store(s[hi][ki], sp + sgitg*DSA_NH*DSA_KB + (hi*8)*DSA_KB + ki*8, DSA_KB);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // reduce the partials, online softmax per head (16 lanes per head row)
        {
            const short h  = tiitg / DSA_KB;
            const short kk = tiitg % DSA_KB;
            float sv = 0.0f;
            for (short g = 0; g < DSA_NSG; ++g) {
                sv += sp[g*DSA_NH*DSA_KB + h*DSA_KB + kk];
            }
            sv = sv*args.scale + smk[kk];

            float mx = sv;
            mx = max(mx, simd_shuffle_xor(mx, 8));
            mx = max(mx, simd_shuffle_xor(mx, 4));
            mx = max(mx, simd_shuffle_xor(mx, 2));
            mx = max(mx, simd_shuffle_xor(mx, 1));

            const float m_old = sm[h];
            const float m_new = max(m_old, mx);
            const bool  empty = m_new == -INFINITY;
            const float p     = empty ? 0.0f : exp(sv - m_new);
            const float alpha = empty ? 1.0f : exp(m_old - m_new);

            float ps = p;
            ps += simd_shuffle_xor(ps, 8);
            ps += simd_shuffle_xor(ps, 4);
            ps += simd_shuffle_xor(ps, 2);
            ps += simd_shuffle_xor(ps, 1);

            spp[h*DSA_KB + kk] = (half) p;
            if (kk < 8) {
                sdg[(h/8)*64 + (h%8)*8 + kk] = kk == (h%8) ? alpha : 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (kk == 0) {
                sl[h] = sl[h]*alpha + ps;
                sm[h] = m_new;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(alpha) O + P V
        {
            simdgroup_float8x8 dg[2];
            simdgroup_load(dg[0], sdg +  0, 8);
            simdgroup_load(dg[1], sdg + 64, 8);
            simdgroup_half8x8 pt[2][2];
            for (short hi = 0; hi < 2; ++hi) {
                for (short ki = 0; ki < 2; ++ki) {
                    simdgroup_load(pt[hi][ki], spp + (hi*8)*DSA_KB + ki*8, DSA_KB);
                }
            }
            for (short di = 0; di < 8; ++di) {
                simdgroup_half8x8 v0, v1;
                simdgroup_load(v0, sk + 0*DSA_D + d0 + di*8, DSA_D);
                simdgroup_load(v1, sk + 8*DSA_D + d0 + di*8, DSA_D);
                for (short hi = 0; hi < 2; ++hi) {
                    simdgroup_float8x8 t;
                    simdgroup_multiply(t, dg[hi], o[hi][di]);
                    simdgroup_multiply_accumulate(t, pt[hi][0], v0, t);
                    simdgroup_multiply_accumulate(t, pt[hi][1], v1, t);
                    o[hi][di] = t;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // final 1/l with the attention sink folded in
    if (tiitg < DSA_NH) {
        const short h = tiitg;
        const float m = sm[h];
        const float l = sl[h];
        float sc;
        if (args.has_sinks) {
            const float s  = sinks[h0 + h];
            const float m2 = max(m, s);
            const float l2 = l*exp(m - m2) + exp(s - m2);
            sc = l2 > 0.0f ? exp(m - m2)/l2 : 0.0f;
        } else {
            sc = l > 0.0f ? 1.0f/l : 0.0f;
        }
        for (short c = 0; c < 8; ++c) {
            sdg[(h/8)*64 + (h%8)*8 + c] = c == (h%8) ? sc : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        simdgroup_float8x8 dg[2];
        simdgroup_load(dg[0], sdg +  0, 8);
        simdgroup_load(dg[1], sdg + 64, 8);
        for (short hi = 0; hi < 2; ++hi) {
            for (short di = 0; di < 8; ++di) {
                simdgroup_float8x8 t;
                simdgroup_multiply(t, dg[hi], o[hi][di]);
                simdgroup_store(t, dst + ((int64_t) row*args.ne02 + h0 + hi*8)*DSA_D + d0 + di*8, DSA_D);
            }
        }
    }
}
