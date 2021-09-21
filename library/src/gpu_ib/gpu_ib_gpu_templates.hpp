/******************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_GPU_IB_GPU_IB_GPU_TEMPLATES_HPP_
#define LIBRARY_SRC_GPU_IB_GPU_IB_GPU_TEMPLATES_HPP_

#include "config.h"  // NOLINT(build/include_subdir)

#include <roc_shmem.hpp>

#include "util.hpp"
#include "queue_pair.hpp"
#include "context.hpp"
#include "wg_state.hpp"

template <ROC_SHMEM_OP Op>
struct OpWrap {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        static_assert(true, "Unimplemented gpu_ib collective.");
    }
};

/******************************************************************************
 ************************** TEMPLATE SPECIALIZATIONS **************************
 *****************************************************************************/
template <>
struct OpWrap<ROC_SHMEM_SUM> {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        dst[i] += src[i];
    }
};

template <>
struct OpWrap<ROC_SHMEM_MAX> {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        dst[i] = max(dst[i], src[i]);
    }
};

template <>
struct OpWrap<ROC_SHMEM_MIN> {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        dst[i] = min(dst[i], src[i]);
    }
};

template <>
struct OpWrap<ROC_SHMEM_PROD> {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        dst[i] *= src[i];
    }
};

template <>
struct OpWrap<ROC_SHMEM_AND> {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        dst[i] &= src[i];
    }
};

template <>
struct OpWrap<ROC_SHMEM_OR> {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        dst[i] |= src[i];
    }
};

template <>
struct OpWrap<ROC_SHMEM_XOR> {
    template <typename T>
    __device__ static void
    Calc(T *src,
         T *dst,
         int i) {
        dst[i] ^= src[i];
    }
};

template <typename T, ROC_SHMEM_OP Op>
__device__ void
compute_reduce(T *src,
               T *dst,
               int size,
               int wg_id,
               int wg_size) {
    for (size_t i = wg_id; i < size; i += wg_size) {
        OpWrap<Op>::Calc(src, dst, i);
    }
    __syncthreads();
}

template <typename T>
__device__ void
GPUIBContext::p(T *dest,
                T value,
                int pe) {
    putmem_nbi(dest, &value, sizeof(T), pe);
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void
GPUIBContext::internal_ring_allreduce(T *dst,
                                      const T *src,
                                      int nelems,
                                      int PE_start,
                                      int logPE_stride,
                                      int PE_size,
                                      T *pWrk,
                                      long *pSync,  // NOLINT(runtime/int)
                                      int n_seg,
                                      int seg_size,
                                      int chunk_size) {
    int off_seg, off_send, off_recv;
    int send_pe = (my_pe + 1) % num_pes;
    long wait_val;  // NOLINT(runtime/int)

    int wg_size = get_flat_block_size();
    int wg_id = get_flat_block_id();

    for (size_t i = wg_id; i < nelems; i += wg_size) {
        dst[i] = src[i];
    }
    __syncthreads();

    for (size_t seg = 0; seg < n_seg; seg++) {
        off_seg = seg * seg_size;
        for (int round = 0; round < num_pes - 1; round++) {
            off_send = (((my_pe + 1 - round + 2 * num_pes)
                        % num_pes) * chunk_size);
            off_recv = (((my_pe - round + 2 * num_pes)
                        % num_pes) * chunk_size);

            if (is_thread_zero_in_block()) {
                putmem_nbi(reinterpret_cast<void*>(&pWrk[off_send]),
                           reinterpret_cast<void*>(&dst[off_send + off_seg]),
                           chunk_size * sizeof(T),
                           send_pe);
                fence();

                wait_val = seg + 100;
                p(&pSync[round], wait_val, send_pe);

                wait_until(&pSync[round], ROC_SHMEM_CMP_EQ, wait_val);
                __threadfence();
            }
            __syncthreads();
            T *ptr = &pWrk[off_recv];
            compute_reduce<T, Op>(&pWrk[off_recv],
                                  &dst[off_seg + off_recv],
                                  chunk_size,
                                  wg_id,
                                  wg_size);
        }
        for (size_t round = num_pes - 1; round < 2 * num_pes - 2; round++) {
            if (is_thread_zero_in_block()) {
                int off_send2 = (((my_pe + 1 - round + 2 * num_pes)
                                 % num_pes) * chunk_size);
                putmem_nbi(reinterpret_cast<void*>(&dst[off_send2 + off_seg]),
                           reinterpret_cast<void*>(&dst[off_send2 + off_seg]),
                            chunk_size * sizeof(T),
                            send_pe);

                fence();
                wait_val = seg + 100;
                p(&pSync[round], wait_val, send_pe);
                wait_until(&pSync[round], ROC_SHMEM_CMP_EQ, wait_val);
            }
        }
    }
    __syncthreads();
    for (size_t i = wg_id; i < 2 * num_pes - 2; i += wg_size) {
        pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
    __syncthreads();
}


template <typename T, ROC_SHMEM_OP Op>
__device__ void
GPUIBContext::internal_direct_allreduce(T *dst,
                                        const T *src,
                                        int nelems,
                                        int PE_start,
                                        int logPE_stride,
                                        int PE_size,
                                        T *pWrk,
                                        long *pSync) {  // NOLINT(runtime/int)
    int stride = 1 << logPE_stride;
    int finish = PE_start + stride * PE_size;
    int pe = my_pe;

    // Single thread schedules the data movement for now.  If we assume
    // multi-thread support we could try this in parallel as well, but
    // it will all end up serializing further down in the runtime, so it most
    // likely isn't worth it.
    if (is_thread_zero_in_block()) {
        for (int i = 0; i < nelems; i++) {
            dst[i] = src[i];
        }

        for (int i = PE_start; i < finish; i += stride) {
            if (i != pe) {
                putmem_nbi(&pWrk[pe * nelems],
                           reinterpret_cast<const void*>(src),
                           nelems * sizeof(T),
                           i);
                fence();
                p(&pSync[pe], 1L, i);
            }
        }
    }

    // Do the compute and pSync reset in parallel.
    int wg_size = get_flat_block_size();
    int wg_id = get_flat_block_id();

    for (int i = PE_start; i < finish; i += stride) {
        if (i != pe) {
            // Wait for leader thread to see that the buffer is ready.
            if (is_thread_zero_in_block()) {
                wait_until(&pSync[i], ROC_SHMEM_CMP_EQ, 1L);
            }
            __syncthreads();

            T *ptr = &pWrk[i * nelems];
            compute_reduce<T, Op>(ptr, dst, nelems, wg_id, wg_size);
        }
    }

    __syncthreads();

    for (int i = wg_id; i < num_pes; i += wg_size) {
        pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }

    __syncthreads();
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void
GPUIBContext::to_all(T *dest,
                     const T *source,
                     int nreduce,
                     int PE_start,
                     int logPE_stride,
                     int PE_size,
                     T *pWrk,
                     long *pSync) {  // NOLINT(runtime/int)
    size_t direct_pWrk =  num_pes * nreduce;
    size_t direct_pSync =  num_pes;

    size_t ring_pSync =  2 * num_pes;

    size_t provided_pWrk = max(nreduce / 2 + 1,
                               ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    size_t provided_pSync = ROC_SHMEM_REDUCE_SYNC_SIZE;

    // TODO(bpotter):
    // We basically do a direct reduce if pWrk is big enough, else we
    // give up. In the future we will want to design algorithms to work
    // with nreduce/2 + 1 space, which would cover every case per the
    // standard.
    if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
        internal_direct_allreduce<T, Op>(dest,
                                         source,
                                         nreduce,
                                         PE_start,
                                         logPE_stride,
                                         PE_size,
                                         pWrk,
                                         pSync);
    } else {
        if (ring_pSync <= ROC_SHMEM_REDUCE_SYNC_SIZE) {
            int chunk_size = 1024;
            size_t ring_pWrk = chunk_size * num_pes;
            if (provided_pWrk < ring_pWrk) {
                ring_pWrk = max(nreduce / 2,  // NOLINT
                                ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE);
                chunk_size = ring_pWrk / num_pes;
            }
            int seg_size = ring_pWrk;
            int n_seg = nreduce / seg_size;
            if (n_seg == 0) {
                n_seg = 1;
                seg_size = nreduce;
                chunk_size = seg_size / num_pes;
            }
            internal_ring_allreduce<T, Op>(dest,
                                           source,
                                           nreduce,
                                           PE_start,
                                           logPE_stride,
                                           PE_size,
                                           pWrk,
                                           pSync,
                                           n_seg,
                                           seg_size,
                                           chunk_size);
        } else {
            GPU_DPRINTF("Unsupported reduction size for gpu_ib.\n");
        }
    }
}

template <typename T>
__device__ void
GPUIBContext::put(T *dest, const T *source, size_t nelems, int pe) {
    putmem(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ T
GPUIBContext::g(const T *source,
                int pe) {
    auto *src_const_cast = reinterpret_cast<const char*>(source);
    uint64_t L_offset = const_cast<char*>(src_const_cast) - base_heap[my_pe];
    if (ipcImpl.isIpcAvailable(my_pe, pe)) {
        T dest;
        ipcImpl.ipcCopy(&dest,
                        ipcImpl.ipc_bases[pe] + L_offset,
                        sizeof(T));
        return dest;
    } else {
        int thread_id = get_flat_block_id();
        int block_size = get_flat_block_size();
        auto *wg_state = WGState::instance();
        int offset = wg_state->get_global_buffer_id() * block_size + thread_id;

        char *base_dest = g_ret;
        char *dest = &base_dest[offset *sizeof(int64_t)];
        T ret;
        size_t nelems = sizeof(T);

        bool must_send_message = wf_coal.coalesce(pe, source, dest, nelems);
        if (!must_send_message) {
            return;
        }
        getQueuePair(pe)->get_nbi<THREAD>(base_heap[pe] + L_offset, dest,
                              nelems, pe, true);
        getQueuePair(pe)->quiet_single<THREAD>();

        ret = *(reinterpret_cast<T*> (dest));
        return ret;
    }
}

template <typename T>
__device__ void
GPUIBContext::put_nbi(T *dest,
                      const T *source,
                      size_t nelems,
                      int pe) {
    putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void
GPUIBContext::get(T *dest,
                  const T *source,
                  size_t nelems,
                  int pe) {
    getmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void
GPUIBContext::get_nbi(T *dest,
                      const T *source,
                      size_t nelems,
                      int pe) {
    getmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void
GPUIBContext::internal_put_broadcast(T *dst,
                                     const T *src,
                                     int nelems,
                                     int pe_root,
                                     int pe_start,
                                     int log_pe_stride,
                                     int pe_size,
                                     long *p_sync) {  // NOLINT(runtime/int)
    if (is_thread_zero_in_block()) {
        if (my_pe == pe_root) {
            int stride = 1 << log_pe_stride;
            int finish = pe_start + stride * pe_size;
            for (int i = pe_start; i < finish; i += stride) {
                if (i != my_pe) {
                    put_nbi(dst, src, nelems, i);
                    p(&p_sync[my_pe], 1L, i);
                }
            }
        } else {
            wait_until(&p_sync[pe_root], ROC_SHMEM_CMP_EQ, 1L);
            p_sync[pe_root] = ROC_SHMEM_SYNC_VALUE;
        }
    }

    __syncthreads();
}

template <typename T>
__device__ void
GPUIBContext::internal_get_broadcast(T *dst,
                                     const T *src,
                                     int nelems,
                                     int pe_root,
                                     long *pSync) {  // NOLINT(runtime/int)
    int64_t wait_val = 1;
    if (is_thread_zero_in_block()) {
        if (my_pe != pe_root) {
            get(dst, src, nelems, pe_root);
            amo_add(&pSync[0], wait_val, 0, pe_root);
        } else {
            // root needs to wait until everyone read the data
            long num_pes_long = num_pes;  // NOLINT(runtime/int)
            wait_until(&pSync[0],
                       ROC_SHMEM_CMP_EQ,
                       num_pes_long - 1);
            pSync[0] = ROC_SHMEM_SYNC_VALUE;
        }
    }

    __syncthreads();
}


template <typename T>
__device__ void
GPUIBContext::broadcast(T *dst,
                        const T *src,
                        int nelems,
                        int pe_root,
                        int pe_start,
                        int log_pe_stride,
                        int pe_size,
                        long *p_sync) {  // NOLINT(runtime/int)
    if (num_pes < 4) {
        internal_put_broadcast(dst,
                               src,
                               nelems,
                               pe_root,
                               pe_start,
                               log_pe_stride,
                               pe_size,
                               p_sync);
    } else {
        internal_get_broadcast(dst, src, nelems, pe_root, p_sync);
    }
}

/******************************************************************************
 ***************** SHMEM X API EXTENSION FOR BLOCK/WAVE LEVEL *****************
 *****************************************************************************/

template <typename T>
__device__ void
GPUIBContext::put_wg(T *dest,
                     const T *source,
                     size_t nelems,
                     int pe) {
    putmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void
GPUIBContext::put_wave(T *dest,
                       const T *source,
                       size_t nelems,
                       int pe) {
    putmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void
GPUIBContext::put_nbi_wg(T *dest,
                         const T *source,
                         size_t nelems,
                         int pe) {
    putmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void
GPUIBContext::put_nbi_wave(T *dest,
                           const T *source,
                           size_t nelems,
                           int pe) {
    putmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void
GPUIBContext::get_wg(T *dest,
                     const T *source,
                     size_t nelems,
                     int pe) {
    getmem_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void
GPUIBContext::get_wave(T *dest,
                       const T *source,
                       size_t nelems,
                       int pe) {
    getmem_wave(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void
GPUIBContext::get_nbi_wg(T *dest,
                         const T *source,
                         size_t nelems,
                         int pe) {
    getmem_nbi_wg(dest, source, nelems * sizeof(T), pe);
}

template <typename T>
__device__ void
GPUIBContext::get_nbi_wave(T *dest,
                           const T *source,
                           size_t nelems,
                           int pe) {
    getmem_nbi_wave(dest, source, nelems * sizeof(T), pe);
}

#endif  // LIBRARY_SRC_GPU_IB_GPU_IB_GPU_TEMPLATES_HPP_
