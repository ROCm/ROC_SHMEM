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

#include <backend.hpp>

#include "util.hpp"

Backend::Backend()
{
    hipMalloc_assert(&print_lock, sizeof(int));
    *print_lock = 0;

    Backend* this_addr = this;
    hipMemcpyToSymbol(HIP_SYMBOL(gpu_handle), &this_addr,
                      sizeof(this), 0, hipMemcpyHostToDevice);
}

roc_shmem_status_t
Backend::dump_stats()
{
    const auto &stats = globalStats;
    uint64_t total = 0;
    for (int i = 0; i < NUM_STATS; i++)
       total += stats.getStat(i);

    fprintf(stdout, "PE %d: Puts (Blocking/P/Nbi) %llu/%llu/%llu Gets "
            "(Blocking/G/Nbi) %llu/%llu/%llu Fences %llu Quiets %llu ToAll "
            "%llu BarrierAll %llu Wait Until %llu Finalizes %llu  Coalesced "
            "%llu Atomic_FAdd %llu Atomic_FCswap %llu Atomic_FInc %llu "
            "Atomic_Fetch %llu Atomic_Add %llu Atomic_Cswap %llu "
            "Atomic_Inc %llu Tests %llu Total %lu\n",
            my_pe, stats.getStat(NUM_PUT), stats.getStat(NUM_PUT_NBI),
            stats.getStat(NUM_P), stats.getStat(NUM_GET), stats.getStat(NUM_G),
            stats.getStat(NUM_GET_NBI), stats.getStat(NUM_FENCE),
            stats.getStat(NUM_QUIET), stats.getStat(NUM_TO_ALL),
            stats.getStat(NUM_BARRIER_ALL), stats.getStat(NUM_WAIT_UNTIL),
            stats.getStat(NUM_FINALIZE), stats.getStat(NUM_MSG_COAL),
            stats.getStat(NUM_ATOMIC_FADD), stats.getStat(NUM_ATOMIC_FCSWAP),
            stats.getStat(NUM_ATOMIC_FINC), stats.getStat(NUM_ATOMIC_FETCH),
            stats.getStat(NUM_ATOMIC_ADD), stats.getStat(NUM_ATOMIC_CSWAP),
            stats.getStat(NUM_ATOMIC_INC), stats.getStat(NUM_TEST), total);

    return dump_backend_stats();
}

roc_shmem_status_t
Backend::reset_stats()
{
    globalStats.resetStats();

    return reset_backend_stats();
}

Backend::~Backend()
{
}

/*
 * Context dispatch implementations.
 */
__device__ void
Context::threadfence_system() { DISPATCH(threadfence_system()); }

__device__ void Context::ctx_destroy() { DISPATCH(ctx_destroy()); }

__device__ void
Context::putmem(void *dest, const void *source, size_t nelems, int pe)
{ DISPATCH(putmem(dest, source, nelems, pe)); }

__device__ void
Context::getmem(void *dest, const void *source, size_t nelems, int pe)
{ DISPATCH(getmem(dest, source, nelems, pe)); }

__device__ void
Context::putmem_nbi(void *dest, const void *source, size_t nelems,
                          int pe)
{ DISPATCH(putmem_nbi(dest, source, nelems, pe)); }

__device__ void
Context::getmem_nbi(void *dest, const void *source, size_t size, int pe)
{ DISPATCH(getmem_nbi(dest, source, size, pe)); }

__device__ void Context::fence() { DISPATCH(fence()); }

__device__ void Context::quiet() { DISPATCH(quiet()); }

__device__ void Context::barrier_all() { DISPATCH(barrier_all()); }

__device__ int64_t
Context::amo_fetch(void *dst, int64_t value, int64_t cond, int pe,
                         uint8_t atomic_op)
{ DISPATCH_RET(amo_fetch(dst, value, cond, pe, atomic_op)) }

__device__ void
Context::amo(void *dst, int64_t value, int64_t cond, int pe,
                   uint8_t atomic_op)
{ DISPATCH(amo(dst, value, cond, pe, atomic_op)) }
