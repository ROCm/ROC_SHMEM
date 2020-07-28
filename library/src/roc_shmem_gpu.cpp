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

/**
 * @file roc_shmem.cpp
 * @brief Public header for ROC_SHMEM device and host libraries.
 *
 * This is the implementation for the public roc_shmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 *
 * The device-side delegation is nasty because we can't use polymorphism with
 * our current shader compiler stack.  Maybe one day.....
 *
 * TODO: Could probably autogenerate many of these functions from macros.
 *
 * TODO: Support runtime backend detection.
 *
 */
#include "config.h"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include <hip/hip_runtime.h>
#include <roc_shmem.hpp>

#include "context.hpp"
#include "backend.hpp"
#include "util.hpp"

#include "reverse_offload/ro_net_gpu_templates.hpp"
#include "gpu_ib/gpu_ib_gpu_templates.hpp"

#include <stdlib.h>

/**
 * Begin GPU Code
 **/

__constant__ Backend *gpu_handle;

__constant__ roc_shmem_ctx_t SHMEM_CTX_DEFAULT;

__device__ void
roc_shmem_wg_init()
{
    int provided;

    /*
     * Non-threaded init is allowed to select any thread mode, so don't worry
     * if provided is different.
     */
    roc_shmem_wg_init_thread(SHMEM_THREAD_WG_FUNNELED, &provided);
}

__device__ void
roc_shmem_wg_init_thread(int requested, int *provided)
{
    gpu_handle->create_wg_state();
    roc_shmem_query_thread(provided);
}

__device__ void
roc_shmem_query_thread(int *provided)
{
#ifdef USE_THREADS
    *provided = SHMEM_THREAD_MULTIPLE;
#else
    *provided = SHMEM_THREAD_WG_FUNNELED;
#endif
}

__device__ void
roc_shmem_wg_finalize()
{ gpu_handle->finalize_wg_state(); }

/* Begin Default Context Interfaces */

__device__ void
roc_shmem_putmem(void *dest, const void *source, size_t nelems, int pe)
{ roc_shmem_putmem(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

template <typename T> __device__ void
roc_shmem_put(T *dest, const T *source, size_t nelems, int pe)
{ roc_shmem_put(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

template <typename T> __device__ void
roc_shmem_p(T *dest, T value, int pe)
{ roc_shmem_p(SHMEM_CTX_DEFAULT, dest, value, pe); }

template <typename T> __device__ T
roc_shmem_g(T *source, int pe)
{ return roc_shmem_g(SHMEM_CTX_DEFAULT, source, pe); }

__device__ void
roc_shmem_getmem(void *dest, const void *source, size_t nelems, int pe)
{ roc_shmem_getmem(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

template <typename T> __device__ void
roc_shmem_get(T *dest, const T *source, size_t nelems, int pe)
{ roc_shmem_get(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

__device__ void
roc_shmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{ roc_shmem_putmem_nbi(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

template <typename T> __device__ void
roc_shmem_put_nbi(T *dest, const T *source, size_t nelems, int pe)
{ roc_shmem_put_nbi(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

__device__ void
roc_shmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{ roc_shmem_getmem_nbi(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

template <typename T> __device__ void
roc_shmem_get_nbi(T *dest, const T *source, size_t nelems, int pe)
{ roc_shmem_get_nbi(SHMEM_CTX_DEFAULT, dest, source, nelems, pe); }

__device__ void roc_shmem_fence()
{ roc_shmem_fence(SHMEM_CTX_DEFAULT); }

__device__ void roc_shmem_quiet()
{ roc_shmem_quiet(SHMEM_CTX_DEFAULT); }

template <typename T> __device__ T
roc_shmem_atomic_fetch_add(T *dest, T val, int pe)
{ return roc_shmem_atomic_fetch_add(SHMEM_CTX_DEFAULT, dest, val, pe); }

template <typename T> __device__ T
roc_shmem_atomic_fetch_cswap(T *dest, T cond, T val, int pe)
{
    return roc_shmem_atomic_fetch_cswap(SHMEM_CTX_DEFAULT,
                                        dest, cond, val, pe);
}

template <typename T> __device__ T
roc_shmem_atomic_fetch_inc(T *dest, int pe)
{ return roc_shmem_atomic_fetch_inc(SHMEM_CTX_DEFAULT, dest, pe); }

template <typename T> __device__ T
roc_shmem_atomic_fetch(T *dest, int pe)
{ return roc_shmem_atomic_fetch(SHMEM_CTX_DEFAULT, dest, pe); }

template <typename T> __device__ void
roc_shmem_atomic_add(T *dest, T val, int pe)
{ roc_shmem_atomic_add(SHMEM_CTX_DEFAULT, dest, val, pe); }

template <typename T> __device__ void
roc_shmem_atomic_cswap(T *dest, T cond, T val, int pe)
{ roc_shmem_atomic_cswap(SHMEM_CTX_DEFAULT, dest, cond, val, pe); }

template <typename T> __device__ void
roc_shmem_atomic_inc(T *dest, int pe)
{ roc_shmem_atomic_inc(SHMEM_CTX_DEFAULT, dest, pe); }

/* Begin Context Interfaces */

__device__ Context *
get_internal_ctx(roc_shmem_ctx_t ctx)
{
    return reinterpret_cast<Context *>(ctx);
}

__device__ roc_shmem_ctx_t
get_external_ctx(Context *ctx)
{
    return reinterpret_cast<roc_shmem_ctx_t>(ctx);
}

__device__ void
roc_shmem_wg_ctx_create(long option, roc_shmem_ctx_t *ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_create\n");

    /*
     * TODO: We don't really create any new contexts here.  Init creates a
     * single wg_private context per WG and a single DEFAULT_CTX for the whole
     * GPU.  We will return the user one of these two to satisfy their request.
     */

    if (option & (SHMEM_CTX_WG_PRIVATE | SHMEM_CTX_PRIVATE)) {
        /*
         * Locking policy for WG-private context is set during
         * context creation based on threading mode for the runtime.
         *
         * If the runtime is set for SHMEM_THREAD_MULTIPLE, then it would be
         * possible to decide at a per CTX level more optimized semantics
         * (e.g., SHMEM_CTX_SERIALIZED would disable intra-wg locking).
         *
         * Unfortunately, since we use the same CTX for multiple ctx_create,
         * we are stuck with the most restrictive performance mode for the
         * given thread policy.
         */
        *ctx = get_external_ctx(WGState::instance()->get_private_ctx());
    } else {
        /*
         * All SHARED contexts satisfied with the DEFAULT_CTX because it is
         * the only context with the required visibility (global).
         *
         * SERIALIZED is a missed performance oppurtunity like mentioned for
         * SHMEM_CTX_WG_PRIVATE, but ignoring it is allowable for correctness.
         */
        *ctx = SHMEM_CTX_DEFAULT;
    }

    __syncthreads();
}

__device__ void
roc_shmem_wg_ctx_destroy(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_destroy\n");

    /*
     * Delay destroying contexts until the work-group indicates that it is
     * finished in roc_shmem_wg_finalize().  Need to do this for now since
     * we are recycling contexts and don't want to destroy one prematurely.
     */
    // get_internal_ctx(ctx)->ctx_destroy();
}

__device__ void
roc_shmem_threadfence_system(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_threadfence_system\n");

    get_internal_ctx(ctx)->threadfence_system();
}

__device__ void
roc_shmem_putmem(roc_shmem_ctx_t ctx, void *dest, const void *source,
                 size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_putmem\n");

    get_internal_ctx(ctx)->putmem(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_put\n"));

    get_internal_ctx(ctx)->putmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T> __device__ void
roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_p\n");

    get_internal_ctx(ctx)->p(dest, value, pe);
}

template <typename T> __device__ T
roc_shmem_g(roc_shmem_ctx_t ctx, T *source, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_g\n");

    return get_internal_ctx(ctx)->g(source, pe);
}

__device__ void
roc_shmem_getmem(roc_shmem_ctx_t ctx, void *dest, const void *source,
                 size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_getmem\n");

    get_internal_ctx(ctx)->getmem(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems,
              int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_get\n"));

    get_internal_ctx(ctx)->get(dest, source, nelems, pe);
}

__device__ void
roc_shmem_putmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source,
                     size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_putmem_nbi\n");

    get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_put_nbi\n"));

    get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe);
}

__device__ void
roc_shmem_getmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source,
                     size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_getmem_nbi\n");

    get_internal_ctx(ctx)->getmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_get_nbi\n"));

    get_internal_ctx(ctx)->get_nbi(dest, source, nelems, pe);
}

__device__ void
roc_shmem_fence(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_fence\n");

    get_internal_ctx(ctx)->fence();
}

__device__ void
roc_shmem_quiet(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_quiet\n");

    get_internal_ctx(ctx)->quiet();
}

template <typename T, ROC_SHMEM_OP Op> __device__ void
roc_shmem_wg_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source,
                    int nreduce, int PE_start, int logPE_stride,
                    int PE_size, T *pWrk, long *pSync)
{
    GPU_DPRINTF("Function: roc_shmem_to_all\n");

    get_internal_ctx(ctx)->to_all<T, Op>(dest, source, nreduce, PE_start,
                                         logPE_stride, PE_size, pWrk, pSync);
}

template <typename T>
__device__ void
roc_shmem_wait_until(roc_shmem_ctx_t ctx, T *ptr, roc_shmem_cmps cmp,
                     T val)
{
    GPU_DPRINTF("Function: roc_shmem_wait_until\n");

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL);

    ctx_internal->wait_until(ptr, cmp, val);
}

template <typename T>
__device__ int
roc_shmem_test(roc_shmem_ctx_t ctx, T *ptr, roc_shmem_cmps cmp, T val)
{
    GPU_DPRINTF("Function: roc_shmem_testl\n");

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_TEST);

    return ctx_internal->test(ptr, cmp, val);
}

__device__ void
roc_shmem_wg_barrier_all(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_barrier_all\n");

    get_internal_ctx(ctx)->barrier_all();
}

__device__ void
roc_shmem_wg_sync_all(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_sync_all\n");

    get_internal_ctx(ctx)->sync_all();
}

__device__ int
roc_shmem_n_pes(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_n_pes\n");

    return get_internal_ctx(ctx)->num_pes;
}

__device__ int
roc_shmem_my_pe(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_my_pe\n");

    return get_internal_ctx(ctx)->my_pe;
}

__device__ uint64_t
roc_shmem_timer()
{
    GPU_DPRINTF("Function: roc_shmem_timer\n");

    return __read_clock();
}

template <typename T> __device__ T
roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch_add\n"));

    return get_internal_ctx(ctx)->amo_fetch_add(dest, val, 0, pe);
}

template <typename T> __device__ T
roc_shmem_atomic_fetch_cswap(roc_shmem_ctx_t ctx, T *dest, T cond, T val,
                             int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch_cswap\n"));

    return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T> __device__ T
roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch_inc\n"));

    return get_internal_ctx(ctx)->amo_fetch_add(dest, 1, 0, pe);
}

template <typename T> __device__ T
roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch\n"));

    return get_internal_ctx(ctx)->amo_fetch_add(dest, 0, 0, pe);
}

template <typename T> __device__ void
roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_add\n"));

    get_internal_ctx(ctx)->amo_add((void*)dest, val, 0, pe);
}

template <typename T> __device__ void
roc_shmem_atomic_cswap(roc_shmem_ctx_t ctx, T *dest, T cond, T val, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_cswap\n"));

    get_internal_ctx(ctx)->amo_cas(dest, val, cond, pe);
}

template <typename T>
__device__ void
roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_inc\n"));

    get_internal_ctx(ctx)->amo_add(dest, 1, 0, pe);
}

/**
 * Template generators
 **/
#define TEMPLATE_GEN(T, Op) \
    template __device__ void \
    roc_shmem_wg_to_all<T, Op>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
        int nreduce, int PE_start, int logPE_stride, int PE_size, T *pWrk, \
        long *pSync);

#define RMA_GEN(T) \
    template __device__ void \
    roc_shmem_put<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                  size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_put_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                  size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_p<T>(roc_shmem_ctx_t ctx, T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_get<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                  size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_get_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                  size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_put<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_put_nbi<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_p<T>(T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_get<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_get_nbi<T>(T *dest, const T *source, size_t nelems, int pe);

#define AMO_GEN(T) \
    template __device__ T \
    roc_shmem_atomic_fetch_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, \
                                  int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_cswap<T>(roc_shmem_ctx_t ctx,  T *dest, T cond, \
                                    T value, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __device__ void \
    roc_shmem_atomic_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_atomic_cswap<T>(roc_shmem_ctx_t ctx,  T *dest, T cond, \
                              T value, int pe); \
    template __device__ void \
    roc_shmem_atomic_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_add<T>(T *dest, T value, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_cswap<T>(T *dest, T cond, T value, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_inc<T>(T *dest, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch<T>(T *dest, int pe); \
    template __device__ void \
    roc_shmem_atomic_add<T>(T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_atomic_cswap<T>(T *dest, T cond, T value, int pe); \
    template __device__ void \
    roc_shmem_atomic_inc<T>(T *dest, int pe);

#define WAIT_GEN(T) \
    template __device__ void \
    roc_shmem_wait_until<T>(roc_shmem_ctx_t ctx,  T *ptr, roc_shmem_cmps cmp, \
                            T val);\
    template __device__ int \
    roc_shmem_test<T>(roc_shmem_ctx_t ctx,  T *ptr, roc_shmem_cmps cmp, \
                      T val);\
    template __device__ void \
    Context::wait_until<T>(T *ptr, roc_shmem_cmps cmp, T val); \
    template __device__ int \
    Context::test<T>(T *ptr, roc_shmem_cmps cmp, T val);

#define ARITH_TEMPLATE_GEN(T) \
    TEMPLATE_GEN(T, ROC_SHMEM_SUM) \
    TEMPLATE_GEN(T, ROC_SHMEM_MIN) \
    TEMPLATE_GEN(T, ROC_SHMEM_MAX) \
    TEMPLATE_GEN(T, ROC_SHMEM_PROD)

#define LOGIC_TEMPLATE_GEN(T) \
    TEMPLATE_GEN(T, ROC_SHMEM_OR) \
    TEMPLATE_GEN(T, ROC_SHMEM_AND) \
    TEMPLATE_GEN(T, ROC_SHMEM_XOR)

#define INT_COLL_GEN(T) \
    ARITH_TEMPLATE_GEN(T) \
    LOGIC_TEMPLATE_GEN(T)

#define FLOAT_COLL_GEN(T) \
    ARITH_TEMPLATE_GEN(T)

INT_COLL_GEN(int)
INT_COLL_GEN(short)
INT_COLL_GEN(long)
INT_COLL_GEN(long long)
FLOAT_COLL_GEN(float)
FLOAT_COLL_GEN(double)
FLOAT_COLL_GEN(long double)

/* All supported OpenSHMEM RMA types */
RMA_GEN(float) RMA_GEN(double) RMA_GEN(long double) RMA_GEN(char)
RMA_GEN(signed char) RMA_GEN(short) RMA_GEN(int) RMA_GEN(long)
RMA_GEN(long long) RMA_GEN(unsigned char) RMA_GEN(unsigned short)
RMA_GEN(unsigned int) RMA_GEN(unsigned long) RMA_GEN(unsigned long long)

/* Supported AMO types for now (Only 64-bits)  */
AMO_GEN(int64_t)
AMO_GEN(uint64_t)
//AMO_GEN(long long)
//AMO_GEN(unsigned long long)
//AMO_GEN(size_t)
//AMO_GEN(ptrdiff_t)
WAIT_GEN(float) WAIT_GEN(double) WAIT_GEN(long double) WAIT_GEN(char)
WAIT_GEN(signed char) WAIT_GEN(short) WAIT_GEN(int) WAIT_GEN(long)
WAIT_GEN(long long) WAIT_GEN(unsigned char) WAIT_GEN(unsigned short)
WAIT_GEN(unsigned int) WAIT_GEN(unsigned long) WAIT_GEN(unsigned long long)


