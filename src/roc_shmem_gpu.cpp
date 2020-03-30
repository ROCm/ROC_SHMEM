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

#include "hip/hip_runtime.h"
#include <roc_shmem.hpp>

#include "backend.hpp"
#include "util.hpp"

#include "reverse_offload/ro_net_gpu_templates.hpp"
#include "gpu_ib/gpu_ib_gpu_templates.hpp"

#include <stdlib.h>

/**
 * Begin GPU Code
 **/

__constant__ Backend *gpu_handle;

__device__
Context::Context(const Backend &handle)
    : num_pes(handle.getNumPEs()), my_pe(handle.getMyPE())
{ }

__device__
ROContext::ROContext(const Backend &b)
    : Context(b)
{
    HIP_DYNAMIC_SHARED(char, baseDynamicPtr);

    // Set the heap base to account for space used by this context
    dynamicPtr = baseDynamicPtr + sizeof(ROContext);
}

__device__
GPUIBContext::GPUIBContext(const Backend &b)
    : Context(b)
{
    HIP_DYNAMIC_SHARED(char, baseDynamicPtr);

    // Set the heap base to account for space used by this context
    dynamicPtr = baseDynamicPtr + sizeof(GPUIBContext);
}

template <typename T> __device__ void
Context::wait_until(T *ptr, roc_shmem_cmps cmp, T val)
{
    while (!test(ptr, cmp, val)) __roc_inv();
}

template <typename T> __device__ int
Context::test(T *ptr, roc_shmem_cmps cmp, T val)
{
    int ret = 0;
    volatile T * vol_ptr = (T*) ptr;
    __roc_inv();
    switch (cmp) {
        case ROC_SHMEM_CMP_EQ:
            if (*vol_ptr == val) ret = 1;
            break;
        case ROC_SHMEM_CMP_NE:
            if (*vol_ptr != val) ret = 1;
            break;
        case ROC_SHMEM_CMP_GT:
            if (*vol_ptr > val) ret = 1;
            break;
        case ROC_SHMEM_CMP_GE:
            if (*vol_ptr >= val) ret = 1;
            break;
        case ROC_SHMEM_CMP_LT:
            if (*vol_ptr < val) ret = 1;
            break;
        case ROC_SHMEM_CMP_LE:
            if (*vol_ptr <= val) ret = 1;
            break;
        default:
            break;
    }
    return ret;
}

__device__ bool
Context::wavefrontNetCoalescer(int pe, const void *source,
                                     const void *dest, size_t &size)
{
    /*
     * Coalesce contiguous messages from the same wavefront to the same pe.
     * The command must already be the same for active threads otherwise they
     * would have diverged at the function call level.
     *
     * This is probably a research problem unto itself.  There appear to be
     * interesting tradeoffs between number of instructions, LDS utalization,
     * and quality of coalescing.
     *
     * The way we do this for now is fairly low overhead and uses no LDS space
     * or 'rounds' of a tree reduction, but can miss a couple scenerios.
     * This method can identify multiple groups of coalescable packets within
     * a wavefront.  For example if threads 0-8 are coalescable and 10-63 are
     * coalescable, the logic will generate 3 messages, which is optimal.
     * However all must be contigous by thread id (e.g. if thread
     * 0 and thread 2 are coalescable, but thread 1 isn't none will be
     * coalesced). It also misses oppurtunities when threads have coalescable
     * msgs but the sizes of the msgs are different.
     *
     * First use a single round of DPP shuffle ops to see if our lower neighbor
     * thread (defined as the thread in the wavefront with our thread id - 1)
     * can be coalesced with us.  Coalescable means that the our lower
     * neighbor's source and dest addresses is our addr - size and has the
     * same PE and size.
     *
     * Every thread then shares its 'coalescability' with its lower neighbor to
     * all other active threads through a ballot predicate.
     *
     * We are responsible for crafting a packet if we aren't coalesceable with
     * our lower neighbor. Thread zero or threads with an inactive lower
     * neighbor are never coalescable. We figure out how much to coalesce by
     * counting the number of consequtive bits in higher threads that are
     * coalescable with their immediate lower neighbor.
     */
    int wv_id = get_flat_block_id() % WF_SIZE;
    uint64_t mask = __ballot(1);

    const uint64_t src = (const uint64_t) source;
    const uint64_t dst = (const uint64_t) dest;

    GPU_DPRINTF("Coal: thread_id_in_wv %d active_threads %d mask %p dst %p "
                "src %p size %u\n", wv_id, __popcll(mask), mask, dst, src,
                size);

    // Split 64 bit values into high and low for 32 bit shuffles
    uint32_t src_low = uint32_t(src & 0xFFFFFFFF);
    uint32_t src_high = uint32_t((src >> 32) & 0xFFFFFFFF);

    uint32_t dst_low = uint32_t(dst & 0xFFFFFFFF);
    uint32_t dst_high = uint32_t((dst >> 32) & 0xFFFFFFFF);

    // Shuffle msg info to upwards neighboring threads
    uint64_t lower_src_low = __shfl_up(src_low, 1);
    uint64_t lower_src_high = __shfl_up(src_high, 1);
    uint64_t lower_dst_low = __shfl_up(dst_low, 1);
    uint64_t lower_dst_high = __shfl_up(dst_high, 1);
    int lower_pe = __shfl_up(pe, 1);
    size_t lower_size =  __shfl_up((unsigned int) size, 1);

    // Reconstruct 64 bit values
    uint64_t lower_src = (lower_src_high << 32) | lower_src_low;
    uint64_t lower_dst = (lower_dst_high << 32) | lower_dst_low;

    GPU_DPRINTF("Coal: thread_id_in_wv %d got shuffle src %p dst %p\n",
                wv_id, lower_src, lower_dst);

    /*
     * I am responsible for sending a message if I am not coalescable with my
     * lower neighbor.  If I am coalescable with my lower neighbor,
     * someone else will send for me.
     */
    bool coalescable =
        (mask & (1LL << (wv_id - 1))) &&   // Ensure lower lane is active
        (lower_size == size) &&            // Ensure lower lane size is equal
        ((lower_src + size) == src) &&     // Ensure I cover lower src
        ((lower_dst + size) == dst) &&     // Ensure I cover lower dst
        (pe == lower_pe) &&                // Must be sending to the same pe
        (wv_id != 0);                      // thread zero is never coalescable

    /*
     * Share my lower neighbor coalescability status with all the active
     * threads in the wavefont.  Inactive threads will not participate in the
     * ballot and return 0 in their position.
     */
    uint64_t lowerNeighborCoal = __ballot(coalescable);

    /*
     * If I'm not coalescable, check how many workitems above me that I am
     * responsible for.  Do this by counting the number of contiguous 1's
     * greater than my thread ID from the ballot function.  I will coalesce
     * the messages for all contiguous higher threads that say they are
     * coalescable with their immediate lower neighbor.
     */
    if (!coalescable) {
        int coal_size = size;

        // Remove all lower thread IDs
        lowerNeighborCoal >>= (wv_id + 1);

        /*
         * Invert and find the first bit index set to zero (first higher thread
         * not coalescable with its lower neighbor). I'm responsible for
         * coalescing everything between my own index and this one.
         */
        uint32_t coalMsgs = __ffsll((unsigned long long) ~lowerNeighborCoal);

        if (coalMsgs) {
            ctxStats.incStat(NUM_MSG_COAL);
            coal_size += size * (coalMsgs - 1);
        }

        GPU_DPRINTF("Coal: thread [%d - %d] src %p dst %p pe %d coalescing %d "
                    "bytes\n", wv_id, wv_id + coalMsgs - 1, src, dst, pe,
                    coal_size);

        size = coal_size;
    }

    return !coalescable;
}

__device__ char *
Context::allocateDynamicShared(size_t size)
{
    if (is_thread_zero_in_block()) {
        dynamicPtr += size;
         // ROCm (as of 3.0) device-side printf doesn't handle %p format for
         // the char * data type correctly, so we need to cast to some other
         // type (e.g. void *) to make this work.
        GPU_DPRINTF("Allocating %u bytes dynamic LDS.  Heap ptr at %p.\n",
                    size, (void *) dynamicPtr);
    }

    // dynamicPtr is updated for all threads after this call and can be
    // returned per-thread.
    __syncthreads();

    return dynamicPtr - size;
}

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
roc_shmem_ctx_create(long option, roc_shmem_ctx_t *ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_create\n");

    // TODO: Rework this
    switch (gpu_handle->type) {
        case RO_BACKEND:
        {
            HIP_DYNAMIC_SHARED(ROContext, ctx_internal);

            // Only initialize the context once.
            if (is_thread_zero_in_block())
                new (ctx_internal) ROContext(*gpu_handle);

            __syncthreads();

            ctx_internal->ctx_create(option);
            (*ctx) = get_external_ctx(ctx_internal);
            break;
        }
        case GPU_IB_BACKEND:
        {
            HIP_DYNAMIC_SHARED(GPUIBContext, ctx_internal);

            // Only initialize the context once.
            if (is_thread_zero_in_block())
                new (ctx_internal) GPUIBContext(*gpu_handle);

            __syncthreads();

            ctx_internal->ctx_create(option);
            (*ctx) = get_external_ctx(ctx_internal);
            break;
        }
        default:
            break;
    }
}

__device__ void
roc_shmem_ctx_destroy(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_destroy\n");

    Context *ctx_internal = get_internal_ctx(ctx);

    if (is_thread_zero_in_block()) {
        ctx_internal->ctxStats.incStat(NUM_FINALIZE);
        gpu_handle->globalStats.accumulateStats(ctx_internal->ctxStats);
    }

    ctx_internal->ctx_destroy();
}

__device__ void
roc_shmem_threadfence_system(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_threadfence_system\n");

    Context *ctx_internal = get_internal_ctx(ctx);

    ctx_internal->threadfence_system();
}

__device__ void
roc_shmem_putmem(roc_shmem_ctx_t ctx, void *dest, const void *source,
                 size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_putmem\n");

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_PUT);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, nelems))
        return;
#endif

    ctx_internal->putmem(dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_put\n"));

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_PUT);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    size_t size = nelems * sizeof(T);
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, size))
        return;

    nelems = size / sizeof(T);
#endif

    ctx_internal->putmem(dest, source, sizeof(T) * nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_p\n");

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_P);

    /*
     * TODO: Need to handle _p a bit differently for coalescing, since the
     * owner of a coalesced message needs val from all absorbed messages.
     */
     ctx_internal->p(dest, value, pe);
}

template <typename T>
__device__ T
roc_shmem_g(roc_shmem_ctx_t ctx, T *source, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_g\n");

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_G);

    /*
     * TODO: Need to handle _g a bit differently for coalescing, since the
     * owner of a coalesced message needs val from all absorbed messages.
     */
    return ctx_internal->g(source, pe);
}

__device__ void
roc_shmem_getmem(roc_shmem_ctx_t ctx, void *dest, const void *source,
                 size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_getmem\n");

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_GET);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, nelems))
        return;
#endif

    ctx_internal->getmem(dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems,
              int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_get\n"));

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_GET);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    size_t size = nelems * sizeof(T);
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, size))
        return;

    nelems = size / sizeof(T);
#endif

    ctx_internal->get(dest, source, nelems, pe);
}

__device__ void
roc_shmem_putmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source,
                     size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_putmem_nbi\n");

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_PUT_NBI);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, nelems))
        return;
#endif

    ctx_internal->putmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_put_nbi\n"));

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_PUT_NBI);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    size_t size = nelems * sizeof(T);
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, size))
        return;

    nelems = size / sizeof(T);
#endif

    ctx_internal->put_nbi(dest, source, nelems, pe);
}

__device__ void
roc_shmem_getmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source,
                     size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_getmem_nbi\n");

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_GET_NBI);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, nelems))
        return;
#endif
    ctx_internal->getmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_get_nbi\n"));

    if (nelems == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_GET_NBI);

#ifdef _WF_COAL_
    // Threads in this WF that successfully coalesce just drop out
    size_t size = nelems * sizeof(T);
    if (!ctx_internal->wavefrontNetCoalescer(pe, source, dest, size))
        return;

    nelems = size / sizeof(T);
#endif

    ctx_internal->get_nbi(dest, source, nelems, pe);
}

__device__ void
roc_shmem_fence(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_fence\n");

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_FENCE);

    ctx_internal->fence();
}

__device__ void
roc_shmem_quiet(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_quiet\n");

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_QUIET);

    ctx_internal->quiet();
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void
roc_shmem_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source,
                 int nreduce, int PE_start, int logPE_stride,
                 int PE_size, T *pWrk, long *pSync)
{
    GPU_DPRINTF("Function: roc_shmem_to_all\n");

    if (nreduce == 0)
        return;

    Context *ctx_internal = get_internal_ctx(ctx);

    if (is_thread_zero_in_block())
        ctx_internal->ctxStats.incStat(NUM_TO_ALL);

    ctx_internal->to_all<T, Op>(dest, source, nreduce, PE_start,
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
roc_shmem_barrier_all(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_barrier_all\n");

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_BARRIER_ALL);

    ctx_internal->barrier_all();
}

__device__ int
roc_shmem_n_pes(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_n_pes\n");

    Context *ctx_internal = get_internal_ctx(ctx);

    return ctx_internal->num_pes;
}

__device__ int
roc_shmem_my_pe(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_my_pe\n");

    Context *ctx_internal = get_internal_ctx(ctx);

    return ctx_internal->my_pe;
}

__device__ uint64_t
roc_shmem_timer()
{
    GPU_DPRINTF("Function: roc_shmem_timer\n");

    return __read_clock();
}

template <typename T>
__device__ T
roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch_add\n"));

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_ATOMIC_FADD);
    T ret = 0;

    ret = ctx_internal->amo_fetch(dest, val, 0, pe, ATOMIC_FADD);

    return ret;
}
template <typename T>
__device__ T
roc_shmem_atomic_fetch_cswap(roc_shmem_ctx_t ctx, T *dest, T cond, T val, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch_cswap\n"));

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_ATOMIC_FCSWAP);
    T ret = 0;

    ret = ctx_internal->amo_fetch(dest, val, cond, pe, ATOMIC_FCAS);

    return ret;
}

template <typename T>
__device__ T
roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch_inc\n"));

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_ATOMIC_FINC);
    T ret = 0;

    ret = ctx_internal->amo_fetch(dest, 1, 0, pe, ATOMIC_FADD);

    return ret;
}
template <typename T>
__device__ T
roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_fetch\n"));

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_ATOMIC_FETCH);
    T ret ;

    ret = ctx_internal->amo_fetch(dest, 0, 0, pe, ATOMIC_FADD);

    return ret;
}

template <typename T>
__device__ void
roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_add\n"));

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_ATOMIC_ADD);

    ctx_internal->amo((void*)dest, val, 0, pe, ATOMIC_FADD);
}

template <typename T>
__device__ void
roc_shmem_atomic_cswap(roc_shmem_ctx_t ctx, T *dest, T cond, T val, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_cswap\n"));

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_ATOMIC_CSWAP);

    ctx_internal->amo(dest, val, cond, pe, ATOMIC_FCAS);
}

template <typename T>
__device__ void
roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF(("Function: roc_shmem_atomic_inc\n"));

    Context *ctx_internal = get_internal_ctx(ctx);
    ctx_internal->ctxStats.incStat(NUM_ATOMIC_INC);

    ctx_internal->amo(dest, 1, 0, pe, ATOMIC_FADD);
}

/**
 * Template generators
 **/
#define TEMPLATE_GEN(T, Op) \
    template __device__ void \
    roc_shmem_to_all<T, Op>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
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
                  size_t nelems, int pe);

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
    roc_shmem_atomic_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe);

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


