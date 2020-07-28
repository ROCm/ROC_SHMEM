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

#include "config.h"

#include "context.hpp"
#include "backend.hpp"

#include "util.hpp"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

__host__
Context::Context(const Backend &handle, bool _shareable)
    : num_pes(handle.getNumPEs()), my_pe(handle.getMyPE()),
      shareable(_shareable)
{ }

__device__
Context::Context(const Backend &handle, bool _shareable)
    : num_pes(handle.getNumPEs()), my_pe(handle.getMyPE()),
      shareable(_shareable)
{
    /*
     * Device-side context constructor is a work-group collective, so make sure
     * all the members have their default values before returning.
     *
     * Each thread is essentially initializing the same thing right over the
     * top of each other for all the default values in context.hh (and the
     * initializer list). It's not incorrect, but it is weird and probably
     * wasteful.
     *
     * TODO: Might consider refactoring so that constructor is always called
     * from a single thread, and the parallel portion of initialization can be
     * a seperate function.  This requires reworking all the derived classes
     * since their constructors actually make use of all the threads to boost
     * performance.
     */
    __syncthreads();
}

__device__ void
Context::lock()
{
    if (!shareable) return;
    /*
     * We need to check this context out to a work-group, and only let threads
     * that are a part of the owning work-group through.  It's a bit like a
     * re-entrant lock, with the added twist that a thread checks out the lock
     * for his entire work-group.
     *
     * TODO: This is a very tough thing to get right for GPU and needs
     * to be tested!  Also it does nothing to ensure that every work-group
     * is getting its fair share of the lock.
     */
    int num_threads_in_wv = wave_SZ();

    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        /*
         * All the metadata associated with this lock needs to be accessed
         * atomically or it will race.
         */
        while (atomicCAS(&ctx_lock, 0, 1) == 1);

        /*
         * If somebody in my work-group already owns the default context, just
         * record how many threads are going to be here and go about our
         * business.
         *
         * If my work-group doesn't own the default context, then
         * we need to wait for it to become available.  Relinquish
         * ctx_lock while waiting or it will never become available.
         *
         */
        int wg_id = get_flat_grid_id();

        while (wg_owner != wg_id) {
            if (wg_owner == -1) {
                wg_owner = wg_id;
                __threadfence();
            } else {
                ctx_lock = 1;
                __threadfence();
                /*
                 * TODO: Might want to consider some back-off here.
                 */
                while (atomicCAS(&ctx_lock, 0, 1) == 1);
            }
        }

        num_threads_in_lock += num_threads_in_wv;
        __threadfence();

        ctx_lock = 0;
        __threadfence();
    }
}

__device__ void
Context::unlock()
{
    if (!shareable) return;

    int num_threads_in_wv = wave_SZ();

    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        while (atomicCAS(&ctx_lock, 0, 1) == 1);

        num_threads_in_lock -= num_threads_in_wv;

        /*
         * Last thread out for this work-group opens the door for other
         * work-groups to take possession.
         */
        if (num_threads_in_lock == 0)
            wg_owner = -1;

        __threadfence();

        ctx_lock = 0;
        __threadfence();
    }
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
            ctxStats.incStat(NUM_MSG_COAL, coalMsgs - 1);
            coal_size += size * (coalMsgs - 1);
        }

        GPU_DPRINTF("Coal: thread [%d - %d] src %p dst %p pe %d coalescing %d "
                    "bytes\n", wv_id, wv_id + coalMsgs - 1, src, dst, pe,
                    coal_size);

        size = coal_size;
    }

    return !coalescable;
}

/*
 * Context dispatch implementations.
 */
__device__ void
Context::threadfence_system()
{
    DISPATCH(threadfence_system());
}

__device__ void
Context::ctx_destroy()
{
    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_FINALIZE);
        gpu_handle->globalStats.accumulateStats(ctxStats);
    }

    DISPATCH(ctx_destroy());
}

__device__ void
Context::putmem(void *dest, const void *source, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctxStats.incStat(NUM_PUT);

#ifdef WF_COAL
    // Threads in this WF that successfully coalesce just drop out
    if (!wavefrontNetCoalescer(pe, source, dest, nelems))
        return;
#endif

    DISPATCH(putmem(dest, source, nelems, pe));
}

__device__ void
Context::getmem(void *dest, const void *source, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctxStats.incStat(NUM_GET);

#ifdef WF_COAL
    // Threads in this WF that successfully coalesce just drop out
    if (!wavefrontNetCoalescer(pe, source, dest, nelems))
        return;
#endif

    DISPATCH(getmem(dest, source, nelems, pe));
}

__device__ void
Context::putmem_nbi(void *dest, const void *source, size_t nelems,
                    int pe)
{
    if (nelems == 0)
        return;

    ctxStats.incStat(NUM_PUT_NBI);

#ifdef WF_COAL
    // Threads in this WF that successfully coalesce just drop out
    if (!wavefrontNetCoalescer(pe, source, dest, nelems))
        return;
#endif

    DISPATCH(putmem_nbi(dest, source, nelems, pe));
}

__device__ void
Context::getmem_nbi(void *dest, const void *source, size_t size, int pe)
{
    if (size == 0)
        return;

    ctxStats.incStat(NUM_GET_NBI);

#ifdef WF_COAL
    // Threads in this WF that successfully coalesce just drop out
    if (!wavefrontNetCoalescer(pe, source, dest, size))
        return;
#endif

    DISPATCH(getmem_nbi(dest, source, size, pe));
}

__device__ void Context::fence()
{
    ctxStats.incStat(NUM_FENCE);

    DISPATCH(fence());
}

__device__ void Context::quiet()
{
    ctxStats.incStat(NUM_QUIET);

    DISPATCH(quiet());
}

__device__ void Context::barrier_all()
{
    ctxStats.incStat(NUM_BARRIER_ALL);

    DISPATCH(barrier_all());
}

__device__ void Context::sync_all()
{
    ctxStats.incStat(NUM_SYNC_ALL);

    DISPATCH(sync_all());
}

__device__ int64_t
Context::amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe)
{
    ctxStats.incStat(NUM_ATOMIC_FADD);

    DISPATCH_RET(amo_fetch_add(dst, value, cond, pe));
}

__device__ void
Context::amo_add(void *dst, int64_t value, int64_t cond, int pe)
{
    ctxStats.incStat(NUM_ATOMIC_ADD);

    DISPATCH(amo_add(dst, value, cond, pe));
}

__device__ int64_t
Context::amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    ctxStats.incStat(NUM_ATOMIC_FCSWAP);

    DISPATCH_RET(amo_fetch_cas(dst, value, cond, pe));
}


__device__ void
Context::amo_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    ctxStats.incStat(NUM_ATOMIC_CSWAP);

    DISPATCH(amo_cas(dst, value, cond, pe));
}
