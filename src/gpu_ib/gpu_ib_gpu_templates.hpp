/******************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef GPU_IB_GPU_TEMPLATES_H
#define GPU_IB_GPU_TEMPLATES_H

#include "config.h"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include <roc_shmem.hpp>

#include "gpu_ib_internal.hpp"

template<ROC_SHMEM_OP Op> struct OpWrap {
    template <typename T> __device__ static void Calc(T *src, T* dst, int i)
    {  static_assert(true, "Unimplemented gpu_ib collective."); } };

/**
 * Add specializations here!
 * Seems silly to wrap this in an object, but C++ currently doesn't support
 * partially specialized template functions, so that's what we have to do.
 **/
template<> struct OpWrap<ROC_SHMEM_SUM> {
    template <typename T> static void Calc(T *src, T* dst, int i)
    { dst[i] += src[i]; } };

template<> struct OpWrap<ROC_SHMEM_MAX> {
    template <typename T> static void Calc(T *src, T* dst, int i)
    { dst[i] = max(dst[i], src[i]); } };

template<> struct OpWrap<ROC_SHMEM_MIN> {
    template <typename T> static void Calc(T *src, T* dst, int i)
    { dst[i] = min(dst[i], src[i]); } };

template<> struct OpWrap<ROC_SHMEM_PROD> {
    template <typename T> static void Calc(T *src, T* dst, int i)
    { dst[i] *= src[i]; } };

template<> struct OpWrap<ROC_SHMEM_AND> {
    template <typename T> static void Calc(T *src, T* dst, int i)
    { dst[i] &= src[i]; } };

template<> struct OpWrap<ROC_SHMEM_OR> {
    template <typename T> static void Calc(T *src, T* dst, int i)
    { dst[i] |= src[i]; } };

template<> struct OpWrap<ROC_SHMEM_XOR> {
    template <typename T> static void Calc(T *src, T* dst, int i)
    { dst[i] ^= src[i]; } };

template <typename T, ROC_SHMEM_OP Op>
__device__ void
compute_reduce(T* src, T* dst, int size, int wg_id, int wg_size)
{
    for (int i = wg_id; i < size; i += wg_size)
        OpWrap<Op>::Calc(src, dst, i);
}

template <typename T>
__device__ void
GPUIBContext::p(T *dest, T value, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_p\n");

    // GPU_IB backend will inline packets automatically
    putmem(dest, &value, 1, pe);
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void
GPUIBContext::internal_direct_allreduce(T *dst, const T *src, int nelems,
                                        int PE_start, int logPE_stride,
                                        int PE_size, T *pWrk, long *pSync)
{
    GPU_DPRINTF("Function: internal_direct_allreduce\n");

    int stride = 1 << logPE_stride;
    int finish = PE_start + stride * PE_size;
    int n_pes = num_pes;
    int pe = my_pe;

    // Single thread schedules the data movement for now.  If we assume
    // multi-thread support we could try this in parallel as well, but
    // it will all end up serializing further down in the runtime, so it most
    // likely isn't worth it.
    if (is_thread_zero_in_block()) {

        for (int i = 0; i < nelems; i++)
            dst[i] = src[i];

        for (int i = PE_start; i < finish; i += stride) {
            if (i != pe) {
                putmem_nbi(&pWrk[pe * nelems], (void *) src,
                           nelems * sizeof(T), i);
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

            T * ptr = &pWrk[i * nelems];
            compute_reduce<T, Op>(ptr, dst, nelems, wg_id, wg_size);
        }
    }

    __syncthreads();

    for (int i = wg_id; i < n_pes; i += wg_size) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }

    __syncthreads();
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void
GPUIBContext::to_all(T *dest, const T *source, int nreduce, int PE_start,
                     int logPE_stride, int PE_size, T *pWrk, long *pSync)
{
    GPU_DPRINTF("Function: gpu_ib_to_all\n");

    size_t direct_pWrk =  num_pes * nreduce;
    size_t direct_pSync =  num_pes;

    size_t provided_pWrk = max(nreduce/2 + 1, SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    size_t provided_pSync = SHMEM_REDUCE_SYNC_SIZE;

    // TODO: We basically do a direct reduce if pWrk is big enough, else we
    // give up.  In the future we will want to design algorithms to work
    // with nreduce/2 + 1 space, which would cover every case per the
    // standard.
    if (provided_pWrk >= direct_pWrk && provided_pSync >= direct_pSync) {
        internal_direct_allreduce<T, Op>(dest, source, nreduce, PE_start,
                                         logPE_stride, PE_size, pWrk, pSync);
    } else {
        // pWrk/pSync too small for direct reduction.
        // TODO: Unfortunately, if we assert then the print won't go.  Rework
        // when assert() library is better in HIP.
        GPU_DPRINTF("Unsupported reduction size for gpu_ib.\n");
    }

}

template <typename T>
__device__ void
GPUIBContext::put(T *dest, const T *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_put\n");
    putmem_nbi(dest, source, sizeof(T) * nelems, pe);

    // Too conservative, but functionally correct
    quiet();
}

template <typename T>
__device__ T
GPUIBContext::g(T *source, int pe)
{
    int block_id = get_flat_grid_id();
    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();
    int offset = block_id * block_size + thread_id;

    char *base_dest = g_ret;
    char * dest = &base_dest[offset *sizeof(int64_t)];
    T ret;

    getmem(dest, source, sizeof(T), pe);

    ret = *(reinterpret_cast<T*> (dest));
    return ret;
}

template <typename T>
__device__ void
GPUIBContext::put_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_put_nbi\n");
    putmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void
GPUIBContext::get(T *dest, const T *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_get\n");
    getmem_nbi(dest, source, sizeof(T) * nelems, pe);

    // Too conservative, but functionally correct
    quiet();
}

template <typename T>
__device__ void
GPUIBContext::get_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_get_nbi\n");
    getmem_nbi(dest, source, sizeof(T) * nelems, pe);
}

#endif // GPU_IB_GPU_TEMPLATES_H
