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

#include "config.h"  // NOLINT(build/include_subdir)

#include "context.hpp"
#include "backend.hpp"

#include "util.hpp"

__host__
Context::Context(const Backend &handle,
                 bool _shareable)
    : num_pes(handle.getNumPEs()),
      my_pe(handle.getMyPE()),
      shareable(_shareable) {
}

__device__
Context::Context(const Backend &handle,
                 bool _shareable)
    : num_pes(handle.getNumPEs()),
      my_pe(handle.getMyPE()),
      shareable(_shareable) {
    /*
     * Device-side context constructor is a work-group collective, so make
     * sure all the members have their default values before returning.
     *
     * Each thread is essentially initializing the same thing right over the
     * top of each other for all the default values in context.hh (and the
     * initializer list). It's not incorrect, but it is weird and probably
     * wasteful.
     *
     * TODO: Might consider refactoring so that constructor is always called
     * from a single thread, and the parallel portion of initialization can be
     * a separate function. This requires reworking all the derived classes
     * since their constructors actually make use of all the threads to boost
     * performance.
     */
    __syncthreads();
}

__device__ void
Context::lock() {
    if (!shareable) {
        return;
    }
    /*
     * We need to check this context out to a work-group, and only let threads
     * that are a part of the owning work-group through. It's a bit like a
     * re-entrant lock, with the added twist that a thread checks out the lock
     * for his entire work-group.
     *
     * TODO: This is a very tough thing to get right for GPU and needs
     * to be tested! Also it does nothing to ensure that every work-group
     * is getting its fair share of the lock.
     */
    int num_threads_in_wv = wave_SZ();

    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        /*
         * All the metadata associated with this lock needs to be accessed
         * atomically or it will race.
         */
        while (atomicCAS(&ctx_lock, 0, 1) == 1) {
        }

        /*
         * If somebody in my work-group already owns the default context, just
         * record how many threads are going to be here and go about our
         * business.
         *
         * If my work-group doesn't own the default context, then
         * we need to wait for it to become available. Relinquish
         * ctx_lock while waiting or it will never become available.
         *
         */
        int wg_id = get_flat_grid_id();

        while (wg_owner != wg_id) {
            if (wg_owner == -1) {
                wg_owner = wg_id;
                __threadfence();
            } else {
                ctx_lock = 0;
                __threadfence();
                /*
                 * TODO: Might want to consider some back-off here.
                 */
                while (atomicCAS(&ctx_lock, 0, 1) == 1) {
                }
            }
        }

        num_threads_in_lock += num_threads_in_wv;
        __threadfence();

        ctx_lock = 0;
        __threadfence();
    }
}

__device__ void
Context::unlock() {
    if (!shareable) {
        return;
    }

    int num_threads_in_wv = wave_SZ();

    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        while (atomicCAS(&ctx_lock, 0, 1) == 1) {
        }

        num_threads_in_lock -= num_threads_in_wv;

        /*
         * Last thread out for this work-group opens the door for other
         * work-groups to take possession.
         */
        if (num_threads_in_lock == 0) {
            wg_owner = -1;
        }

        __threadfence();

        ctx_lock = 0;
        __threadfence();
    }
}

/******************************************************************************
 ********************** CONTEXT DISPATCH IMPLEMENTATIONS **********************
 *****************************************************************************/

/******************************************************************************
 ******************************* DEVICE METHODS *******************************
 *****************************************************************************/
__device__ void
Context::threadfence_system() {
    DISPATCH(threadfence_system());
}

__device__ void
Context::ctx_destroy() {
    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_FINALIZE);
        gpu_handle->globalStats.accumulateStats(ctxStats);
    }

    DISPATCH(ctx_destroy());
}

__device__ void
Context::putmem(void *dest,
                const void *source,
                size_t nelems,
                int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT);

    DISPATCH(putmem(dest, source, nelems, pe));
}

__device__ void
Context::getmem(void *dest,
                const void *source,
                size_t nelems,
                int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET);

    DISPATCH(getmem(dest, source, nelems, pe));
}

__device__ void
Context::putmem_nbi(void *dest,
                    const void *source,
                    size_t nelems,
                    int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_NBI);

    DISPATCH(putmem_nbi(dest, source, nelems, pe));
}

__device__ void
Context::getmem_nbi(void *dest,
                    const void *source,
                    size_t size,
                    int pe) {
    if (size == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_NBI);

    DISPATCH(getmem_nbi(dest, source, size, pe));
}

__device__ void
Context::fence() {
    ctxStats.incStat(NUM_FENCE);

    DISPATCH(fence());
}

__device__ void
Context::quiet() {
    ctxStats.incStat(NUM_QUIET);

    DISPATCH(quiet());
}

__device__ void*
Context::shmem_ptr(const void* dest,
                   int pe) {
    ctxStats.incStat(NUM_SHMEM_PTR);

    DISPATCH_RET_PTR(shmem_ptr(dest, pe));
}

__device__ void
Context::barrier_all() {
    ctxStats.incStat(NUM_BARRIER_ALL);

    DISPATCH(barrier_all());
}

__device__ void
Context::sync_all() {
    ctxStats.incStat(NUM_SYNC_ALL);

    DISPATCH(sync_all());
}

__device__ int64_t
Context::amo_fetch_add(void *dst,
                       int64_t value,
                       int64_t cond,
                       int pe) {
    ctxStats.incStat(NUM_ATOMIC_FADD);

    DISPATCH_RET(amo_fetch_add(dst, value, cond, pe));
}

__device__ void
Context::amo_add(void *dst,
                 int64_t value,
                 int64_t cond,
                 int pe) {
    ctxStats.incStat(NUM_ATOMIC_ADD);

    DISPATCH(amo_add(dst, value, cond, pe));
}

__device__ int64_t
Context::amo_fetch_cas(void *dst,
                       int64_t value,
                       int64_t cond,
                       int pe) {
    ctxStats.incStat(NUM_ATOMIC_FCSWAP);

    DISPATCH_RET(amo_fetch_cas(dst, value, cond, pe));
}

__device__ void
Context::amo_cas(void *dst,
                 int64_t value,
                 int64_t cond,
                 int pe) {
    ctxStats.incStat(NUM_ATOMIC_CSWAP);

    DISPATCH(amo_cas(dst, value, cond, pe));
}

__device__ void
Context::putmem_wg(void *dest,
                   const void *source,
                   size_t nelems,
                   int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_WG);

    DISPATCH(putmem_wg(dest, source, nelems, pe));
}

__device__ void
Context::getmem_wg(void *dest,
                   const void *source,
                   size_t nelems,
                   int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_WG);

    DISPATCH(getmem_wg(dest, source, nelems, pe));
}

__device__ void
Context::putmem_nbi_wg(void *dest,
                       const void *source,
                       size_t nelems,
                       int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_NBI_WG);

    DISPATCH(putmem_nbi_wg(dest, source, nelems, pe));
}

__device__ void
Context::getmem_nbi_wg(void *dest,
                       const void *source,
                       size_t size,
                       int pe) {
    if (size == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_NBI_WG);

    DISPATCH(getmem_nbi_wg(dest, source, size, pe));
}

__device__ void
Context::putmem_wave(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_WAVE);

    DISPATCH(putmem_wave(dest, source, nelems, pe));
}

__device__ void
Context::getmem_wave(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_WAVE);

    DISPATCH(getmem_wave(dest, source, nelems, pe));
}

__device__ void
Context::putmem_nbi_wave(void *dest,
                         const void *source,
                         size_t nelems,
                         int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_NBI_WAVE);

    DISPATCH(putmem_nbi_wave(dest, source, nelems, pe));
}

__device__ void
Context::getmem_nbi_wave(void *dest,
                         const void *source,
                         size_t size,
                         int pe) {
    if (size == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_NBI_WAVE);

    DISPATCH(getmem_nbi_wave(dest, source, size, pe));
}

/******************************************************************************
 ******************************** HOST METHODS ********************************
 *****************************************************************************/

__host__ void
Context::putmem(void *dest,
                const void *source,
                size_t nelems,
                int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_PUT);

    HOST_DISPATCH(putmem(dest, source, nelems, pe));
}

__host__ void
Context::getmem(void *dest,
                const void *source,
                size_t nelems,
                int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_GET);

    HOST_DISPATCH(getmem(dest, source, nelems, pe));
}

__host__ void
Context::putmem_nbi(void *dest,
                    const void *source,
                    size_t nelems,
                    int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_PUT_NBI);

    HOST_DISPATCH(putmem_nbi(dest, source, nelems, pe));
}

__host__ void
Context::getmem_nbi(void *dest,
                    const void *source,
                    size_t nelems,
                    int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_GET_NBI);

    HOST_DISPATCH(getmem_nbi(dest, source, nelems, pe));
}

__host__ int64_t
Context::amo_fetch_add(void *dst,
                       int64_t value,
                       int64_t cond,
                       int pe) {
    ctxHostStats.incStat(NUM_HOST_ATOMIC_FADD);

    HOST_DISPATCH_RET(amo_fetch_add(dst, value, cond, pe));
}

__host__ void
Context::amo_add(void *dst,
                 int64_t value,
                 int64_t cond,
                 int pe) {
    ctxHostStats.incStat(NUM_HOST_ATOMIC_ADD);

    HOST_DISPATCH(amo_add(dst, value, cond, pe));
}

__host__ int64_t
Context::amo_fetch_cas(void *dst,
                       int64_t value,
                       int64_t cond,
                       int pe) {
    ctxHostStats.incStat(NUM_HOST_ATOMIC_FCSWAP);

    HOST_DISPATCH_RET(amo_fetch_cas(dst, value, cond, pe));
}

__host__ void
Context::amo_cas(void *dst,
                 int64_t value,
                 int64_t cond,
                 int pe) {
    ctxHostStats.incStat(NUM_HOST_ATOMIC_CSWAP);

    HOST_DISPATCH(amo_cas(dst, value, cond, pe));
}

__host__ void
Context::fence() {
    ctxHostStats.incStat(NUM_HOST_FENCE);

    HOST_DISPATCH(fence());
}

__host__ void
Context::quiet() {
    ctxHostStats.incStat(NUM_HOST_QUIET);

    HOST_DISPATCH(quiet());
}

__host__ void
Context::sync_all() {
    ctxHostStats.incStat(NUM_HOST_SYNC_ALL);

    HOST_DISPATCH(sync_all());
}

__host__ void
Context::barrier_all() {
    ctxHostStats.incStat(NUM_HOST_BARRIER_ALL);

    HOST_DISPATCH(barrier_all());
}
