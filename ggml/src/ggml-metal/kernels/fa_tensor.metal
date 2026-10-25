#include "common.h"

#ifdef GGML_METAL_HAS_TENSOR

constant bool FC_flash_attn_ext_tensor_has_mask  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 0)]];
constant bool FC_flash_attn_ext_tensor_has_sinks [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 1)]];
constant bool FC_flash_attn_ext_tensor_has_bias  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 2)]];
constant bool FC_flash_attn_ext_tensor_has_scap  [[function_constant(FC_FLASH_ATTN_EXT_TENSOR + 3)]];

// ref: https://arxiv.org/pdf/2307.08691.pdf
template<
    typename T,                                 // Q, K, V and P types
    typename T4,
    short DK,                                   // K head size
    short DV,                                   // V head size
    short Q   = OP_FLASH_ATTN_EXT_TENSOR_NQPSG, // queries per threadgroup
    short C   = OP_FLASH_ATTN_EXT_TENSOR_NCPSG, // cache items per threadgroup
    short NSG = OP_FLASH_ATTN_EXT_TENSOR_NSG>   // number of simd groups
kernel void kernel_flash_attn_ext_tensor(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * blk,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NT = NW*NSG;
    constexpr short NQ = Q/NSG;
    constexpr short NC = C/NW; // columns per thread

    static_assert(DK % 4 == 0,  "DK must be divisible by 4");
    static_assert(Q % NSG == 0, "Q must be divisible by NSG");
    static_assert(C % NW == 0,  "C must be divisible by NW");

    const int iq3 = tgpig[2];
    const int iq2 = tgpig[1];
    const int iq1 = tgpig[0]*Q;

    const short tiitg = sgitg*NW + tiisg;

    threadgroup T     * sq = (threadgroup T     *) shmem;       // [Q, DK] queries
    threadgroup float * ss = (threadgroup float *) (sq + Q*DK); // [Q, C]  scores
    threadgroup T     * sp = (threadgroup T     *) (ss + Q*C);  // [Q, C]  probabilities
    threadgroup float * sr = (threadgroup float *) (sp + Q*C);  // [Q]     per-row scale of O
    threadgroup int   * sf = (threadgroup int   *) (sr + Q);    // [1]     last iteration (ic0 + 1) that rescaled O

    q += iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03;

    {
        const int ikv2 = iq2/(args.ne02/args.ne_12_2);
        const int ikv3 = iq3/(args.ne03/args.ne_12_3);

        k += ikv2*args.nb12 + ikv3*args.nb13;
        v += ikv2*args.nb22 + ikv3*args.nb23;
    }

    // with softcap the scale is small (scale/softcap), so it is applied to the scores to keep the precision of Q
    const float qscale = FC_flash_attn_ext_tensor_has_scap ? 1.0f : args.scale;

    // load the queries, with the scale folded in
    for (int i = tiitg; i < Q*DK/4; i += NT) {
        const int j = i/(DK/4);

        float4 q4 = 0.0f;
        if (iq1 + j < args.ne01) {
            q4 = ((device const float4 *) (q + j*args.nb01))[i%(DK/4)];
        }

        ((threadgroup T4 *) sq)[i] = (T4) (q4*qscale);
    }

    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        pm[jj] = (device const half *) (mask + (iq1 + j)*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    {
        const int nblk1 = (args.ne01 + Q - 1)/Q;
        const int nblk0 = (args.ne11 + C - 1)/C;

        blk += (((iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*nblk1 + iq1/Q)*nblk0;
    }

    float M[NQ];
    float S[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        M[jj] = -FLT_MAX/2;
        S[jj] = 0.0f;
    }

    float slope = 1.0f;

    // ALiBi
    if (FC_flash_attn_ext_tensor_has_bias) {
        const short h = iq2;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const short exph = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exph);
    }

    const int sk = args.ns10;
    const int sv = args.ns20;

    auto tq = tensor(sq, dextents<int32_t, 2>(DK, Q));
    auto ts = tensor(ss, dextents<int32_t, 2>(C,  Q));
    auto tp = tensor(sp, dextents<int32_t, 2>(C,  Q));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, C, DK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<NSG>> mm_qk;

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(Q, DV, C, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<NSG>> mm_pv;

    auto tv0 = tensor((device T *) v, dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

    // the O matrix from the paper
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    if (tiitg == 0) {
        sf[0] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the host guarantees ne11 % C == 0
    for (int ic0 = 0, ic = 0; ic < args.ne11; ++ic0, ic += C) {
        char blk_cur = 1;

        if (FC_flash_attn_ext_tensor_has_mask) {
            blk_cur = blk[ic0];

            if (blk_cur == 0) {
                continue;
            }
        }

        // Q*K^T
        {
            auto tk = tensor((device T *) (k + (uint64_t) ic*args.nb11), dextents<int32_t, 2>(DK, C), array<int, 2>({1, sk}));

            mm_qk.run(tq, tk, ts);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            float s[NC];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                s[ii] = ss[j*C + ii*NW + tiisg];
            }

            if (FC_flash_attn_ext_tensor_has_scap) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] = args.logit_softcap*precise::tanh(s[ii]*args.scale);
                }
            }

            if (FC_flash_attn_ext_tensor_has_mask && blk_cur != 2 && iq1 + j < args.ne31) {
                FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                    s[ii] += slope*(float) pm[jj][ic + ii*NW + tiisg];
                }
            }

            float m = M[jj];

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                m = max(m, s[ii]);
            }

            m = simd_max(m);

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;

            if (m > M[jj] + 8.0f) {
                ms    = exp(M[jj] - m);
                M[jj] = m;

                if (tiisg == 0) {
                    sf[0] = ic0 + 1;
                }
            }

            float sum = 0.0f;

            FOR_UNROLL (short ii = 0; ii < NC; ++ii) {
                // the sum uses the same rounded values as P*V
                const T p = (T) exp(s[ii] - M[jj]);

                sp[j*C + ii*NW + tiisg] = p;

                sum += (float) p;
            }

            S[jj] = S[jj]*ms + simd_sum(sum);

            if (tiisg == 0) {
                sr[j] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V
        if (sf[0] == ic0 + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        {
            auto tv = tensor((device T *) (v + (uint64_t) ic*args.nb21), dextents<int32_t, 2>(DV, C), array<int, 2>({1, sv}));

            mm_pv.run(tp, tv, co);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        // the sink only adds to the denominator - its rescale of O is folded into the final scale
        float ms = 1.0f;

        if (FC_flash_attn_ext_tensor_has_sinks) {
            const float s = ((device const float *) sinks)[iq2];
            const float m = max(M[jj], s);

            ms = exp(M[jj] - m);

            S[jj] = S[jj]*ms + exp(s - m);
        }

        if (tiisg == 0) {
            sr[j] = S[jj] == 0.0f ? 0.0f : ms/S[jj];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    // store to global memory - rows past ne01 are clipped by the tensor extents
    device float * pdst = (device float *) dst + ((uint64_t) iq3*args.ne2*args.ne1 + iq2 + (uint64_t) iq1*args.ne1)*DV;

    auto td = tensor(pdst, dextents<int32_t, 2>(DV, args.ne01 - iq1), array<int, 2>({1, args.ne1*DV}));

    co.store(td);
}

typedef decltype(kernel_flash_attn_ext_tensor<half, half4, 64, 64>) flash_attn_ext_tensor_t;

template [[host_name("kernel_flash_attn_ext_tensor_f16_dk64_dv64"  )]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk128_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk192_dv128")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk256_dv256")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk512_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_f16_dk576_dv512")]]  kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<half, half4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk64_dv64"  )]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 64,  64>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk128_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 128, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk192_dv128")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 192, 128>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk256_dv256")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 256, 256>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk512_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 512, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;
template [[host_name("kernel_flash_attn_ext_tensor_bf16_dk576_dv512")]] kernel flash_attn_ext_tensor_t kernel_flash_attn_ext_tensor<bfloat, bfloat4, 576, 512, OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE>;

// deepseek41 sparse prefill attention on the matrix units, v2 (GGML_METAL_DSV4_SATTN=2): 64 query heads over one KV
// head (K == V), dk = dv = 512, each query row attends to the entries of its per-row index list (window + indexer
// top-k, built by kernel_flash_attn_ext_vec_idx). One threadgroup per (row, NH heads), NSG simdgroups.
// Per block of KB listed keys:
//   S[NH x KB] = Q K^T    tiles of 16 heads x QKN keys on single-simdgroup matmul2d; K^T is gathered straight from
//                         the KV rows into the right-input cooperative tensor (the M5 holds runs of 4 consecutive
//                         reduction elements per lane, i.e. 4 dims of one key: one half4 load), no staging
//   online softmax        f32, per head row, through threadgroup memory; P in half (lazy rescale, |P| <= e^8)
//   O[NH x DS] += P V     each simdgroup owns DS = 512/NSG dims of O in a destination cooperative tensor; V is
//                         gathered into the right-input cooperative tensor (runs of 4 consecutive dims of one key)
// Two threadgroup barriers per block. Q is rounded to half once per threadgroup and kept in the threadgroup's own
// region of dst (overwritten by the f32 result at the end).
// After: Strata metal/kernels/qsa_prompt.metal (MIT; K decoded into the right-input cooperative tensor, online
// softmax) and Splash runtime/metal/kernels/common/paged_attention_tile.h (Apache-2.0; all query heads of a KV head
// fused into the matmul M dimension, running O in a destination cooperative tensor).
template<short NH, short KB, short NSG, short QKN, short QKK, short PVK>
kernel void kernel_dsv4_sparse_attn_v2(
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
    constexpr short D   = 512;
    constexpr short NT  = 32*NSG;
    constexpr short DS  = D/NSG;      // O dims per simdgroup
    constexpr short QKM = 16;         // heads per QK tile
    constexpr short NKT = KB/QKN;     // key tiles per block
    constexpr short NQT = (NH/QKM)*NKT;
    constexpr short LPR = KB < 32 ? KB : 32;  // softmax lanes per head row
    constexpr short RPP = 32/LPR;
    constexpr short CPL = KB/LPR;
    constexpr short NRS = (NH + NSG*RPP - 1)/(NSG*RPP);

    const int row = tgpig.x;
    const int h0  = tgpig.y*NH;
    const int d0  = sgitg*DS;
    const int nkv = args.n_kv_max_padded;

    threadgroup float * ss = (threadgroup float *) shmem;        // [NH][KB] scores
    threadgroup half  * sp = (threadgroup half  *) (ss + NH*KB); // [NH][KB] probabilities
    threadgroup float * sr = (threadgroup float *) (sp + NH*KB); // [NH]     O row scale
    threadgroup int   * sf = (threadgroup int   *) (sr + NH);    // [1]      block (kb + 1) that rescaled O

    // Q -> half, into this threadgroup's own part of dst
    device half * pqh = (device half *) (dst + ((int64_t) row*args.ne02 + h0)*D);
    {
        device const char * pq = q + (uint64_t) row*args.nb01 + (uint64_t) h0*args.nb02;
        for (int i = tiitg; i < NH*D/4; i += NT) {
            const int h = i/(D/4);
            const int c = i%(D/4);
            ((device half4 *) pqh)[i] = (half4) ((device const float4 *) (pq + (uint64_t) h*args.nb02))[c];
        }
    }
    if (tiitg == 0) {
        sf[0] = 0;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    auto tq = tensor(pqh, dextents<int32_t, 2>(D,  NH));
    auto tp = tensor(sp,  dextents<int32_t, 2>(KB, NH));
    auto ts = tensor(ss,  dextents<int32_t, 2>(KB, NH));

    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(QKM, QKN, QKK, false, true,  false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(NH,  DS,  PVK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroup> mm_pv;

    auto tqs = tq.template slice<QKK, QKM>(0, 0);
    auto rk0 = mm_qk.template get_right_input_cooperative_tensor<half, half, float>();
    auto tps = tp.template slice<PVK, NH>(0, 0);
    auto rv0 = mm_pv.template get_right_input_cooperative_tensor<half, half, float>();

    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tps), decltype(rv0), float>();
    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] = 0.0f;
        }
    }

    float M[NRS];
    float S[NRS];
    for (short i = 0; i < NRS; ++i) {
        M[i] = -FLT_MAX/2;
        S[i] = 0.0f;
    }

    device const int  * pidx = idx + (int64_t) row*nkv;
    device const half * pm   = (device const half *) (mask + (uint64_t) row*args.nb31);

    for (int kb = 0; kb < nkv; kb += KB) {
        if (pidx[kb] < 0) {
            break; // lists are compacted: the rest is padding
        }

        // S = Q K^T
        for (short t = sgitg; t < NQT; t += NSG) {
            const short ht = t / NKT;
            const short kt = t % NKT;

            auto acc = mm_qk.template get_destination_cooperative_tensor<decltype(tqs), decltype(rk0), float>();
            FOR_UNROLL (short i = 0; i < acc.get_capacity(); ++i) {
                if (acc.is_valid_element(i)) {
                    acc[i] = 0.0f;
                }
            }

            for (short c0 = 0; c0 < D; c0 += QKK) {
                auto rk = mm_qk.template get_right_input_cooperative_tensor<half, half, float>();
                FOR_UNROLL (short i = 0; i < rk.get_capacity(); i += 4) {
                    const auto ix = rk.get_multidimensional_index(i); // (dim, key)
                    const int  j  = kb + kt*QKN + ix[1];
                    const int  id = j < nkv ? pidx[j] : -1;
                    // clamped row, select on the value
                    const half4 w = *((device const half4 *) (k + (uint64_t) max(id, 0)*args.nb11) + (c0 + ix[0])/4);
                    const half4 v = id >= 0 ? w : half4(0.0h);
                    rk[i + 0] = v.x;
                    rk[i + 1] = v.y;
                    rk[i + 2] = v.z;
                    rk[i + 3] = v.w;
                }
                auto tqc = tq.template slice<QKK, QKM>(c0, ht*QKM);
                mm_qk.run(tqc, rk, acc);
            }

            auto tsc = ts.template slice<QKN, QKM>(kt*QKN, ht*QKM);
            acc.store(tsc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        for (short pass = 0; pass < NRS; ++pass) {
            const short h = (pass*NSG + sgitg)*RPP + tiisg/LPR;
            const short l = tiisg%LPR;

            float s[CPL];
            float m = M[pass];
            for (short c = 0; c < CPL; ++c) {
                const int j  = kb + c*LPR + l;
                const int id = j < nkv ? pidx[j] : -1;
                s[c] = h < NH && id >= 0 ? ss[h*KB + c*LPR + l]*args.scale + (float) pm[max(id, 0)] : -INFINITY;
                m = max(m, s[c]);
            }
            for (short o = LPR/2; o > 0; o /= 2) {
                m = max(m, simd_shuffle_xor(m, o));
            }

            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;
            if (m > M[pass] + 8.0f) {
                ms = exp(M[pass] - m);
                M[pass] = m;
                if (l == 0) {
                    sf[0] = kb + 1;
                }
            }

            float sum = 0.0f;
            for (short c = 0; c < CPL; ++c) {
                // the sum uses the same rounded values as P*V
                const half p = (half) exp(s[c] - M[pass]);
                if (h < NH) {
                    sp[h*KB + c*LPR + l] = p;
                }
                sum += (float) p;
            }
            for (short o = LPR/2; o > 0; o /= 2) {
                sum += simd_shuffle_xor(sum, o);
            }
            S[pass] = S[pass]*ms + sum;
            if (l == 0 && h < NH) {
                sr[h] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms) O + P V on this simdgroup's dims
        if (sf[0] == kb + 1) {
            FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) {
                    co[i] *= sr[co.get_multidimensional_index(i)[1]];
                }
            }
        }

        for (short k0 = 0; k0 < KB; k0 += PVK) {
            auto rv = mm_pv.template get_right_input_cooperative_tensor<half, half, float>();
            FOR_UNROLL (short i = 0; i < rv.get_capacity(); i += 4) {
                const auto ix = rv.get_multidimensional_index(i); // (dim, key)
                const int  j  = kb + k0 + ix[1];
                const int  id = j < nkv ? pidx[j] : -1;
                const half4 w = *((device const half4 *) (k + (uint64_t) max(id, 0)*args.nb11) + (d0 + ix[0])/4);
                const half4 v = id >= 0 ? w : half4(0.0h);
                rv[i + 0] = v.x;
                rv[i + 1] = v.y;
                rv[i + 2] = v.z;
                rv[i + 3] = v.w;
            }
            auto tpc = tp.template slice<PVK, NH>(k0, 0);
            mm_pv.run(tpc, rv, co);
        }
        // the next write of sp/ss/sr is behind the barrier after the next QK pass
    }

    // final 1/l with the attention sink folded in
    for (short pass = 0; pass < NRS; ++pass) {
        const short h = (pass*NSG + sgitg)*RPP + tiisg/LPR;
        const short l = tiisg%LPR;
        float ms = 1.0f;
        float Sx = S[pass];
        if (args.has_sinks && h < NH) {
            const float s = sinks[h0 + h];
            const float m = max(M[pass], s);
            ms = exp(M[pass] - m);
            Sx = Sx*ms + exp(s - m);
        }
        if (l == 0 && h < NH) {
            sr[h] = Sx == 0.0f ? 0.0f : ms/Sx;
        }
    }

    // also orders every read of the half Q before the f32 stores over it
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    FOR_UNROLL (short i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            co[i] *= sr[co.get_multidimensional_index(i)[1]];
        }
    }

    device float * pdst = dst + ((int64_t) row*args.ne02 + h0)*D + d0;
    auto td = tensor(pdst, dextents<int32_t, 2>(DS, NH), array<int, 2>({1, D}));
    co.store(td);
}

typedef decltype(kernel_dsv4_sparse_attn_v2<32, 64, 16, 16, 64, 32>) kernel_dsv4_sparse_attn_v2_t;

// NH = 32 heads, KB = 64 keys, 16 simdgroups, QK tiles 16 x 16 (k step 64), PV k step 32
template [[host_name("kernel_dsv4_sparse_attn_v2_f16")]] kernel kernel_dsv4_sparse_attn_v2_t kernel_dsv4_sparse_attn_v2<32, 64, 16, 16, 64, 32>;

#endif // GGML_METAL_HAS_TENSOR
