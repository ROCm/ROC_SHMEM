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

#include "context.hpp"
#include "backend.hpp"
#include "wg_state.hpp"

#include "util.hpp"

Backend::Backend(unsigned num_wgs)
{
    int num_cus;
    if (hipDeviceGetAttribute(&num_cus,
        hipDeviceAttributeMultiprocessorCount, 0)) {
        exit(-ROC_SHMEM_UNKNOWN_ERROR);
    }

    int max_num_wgs = num_cus * 32;

    if (num_wgs > max_num_wgs || num_wgs == 0)
        num_wgs = max_num_wgs;

    /**
     * Hack to support the default context.
     * This works because we allocate enough queues to cover num_wgs contexts,
     * so this ensures we always have enough extra ones for the default
     * context.  This is bad because it also over allocates any buffers that
     * are dependent on num_wgs but not actually related to contexts.
     */
    num_wgs++;

    CHECK_HIP(hipMalloc((void**) &bufferTokens,
                        sizeof(unsigned int) * num_wgs));

    for (int i = 0; i < num_wgs; i++) {
        bufferTokens[i] = 0;
    }

    num_wg = num_wgs;

    CHECK_HIP(hipMalloc(&print_lock, sizeof(*print_lock)));
    *print_lock = 0;
    hipMemcpyToSymbol(HIP_SYMBOL(print_lock), &print_lock,
                      sizeof(print_lock), 0, hipMemcpyHostToDevice);

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
            "Atomic_Inc %llu Tests %llu SyncAll %llu Total %lu\n",
            my_pe, stats.getStat(NUM_PUT), stats.getStat(NUM_P),
            stats.getStat(NUM_PUT_NBI), stats.getStat(NUM_GET),
            stats.getStat(NUM_G),
            stats.getStat(NUM_GET_NBI), stats.getStat(NUM_FENCE),
            stats.getStat(NUM_QUIET), stats.getStat(NUM_TO_ALL),
            stats.getStat(NUM_BARRIER_ALL), stats.getStat(NUM_WAIT_UNTIL),
            stats.getStat(NUM_FINALIZE), stats.getStat(NUM_MSG_COAL),
            stats.getStat(NUM_ATOMIC_FADD), stats.getStat(NUM_ATOMIC_FCSWAP),
            stats.getStat(NUM_ATOMIC_FINC), stats.getStat(NUM_ATOMIC_FETCH),
            stats.getStat(NUM_ATOMIC_ADD), stats.getStat(NUM_ATOMIC_CSWAP),
            stats.getStat(NUM_ATOMIC_INC), stats.getStat(NUM_TEST),
            stats.getStat(NUM_SYNC_ALL), total);

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
    CHECK_HIP(hipFree(bufferTokens));
}

__device__ void
Backend::create_wg_state()
{
    WGState::create();

    __syncthreads();

    /*
     * We preallocate a single wg_private context for this WG and bind it to
     * the WG state.
     */
    Context *ctx = nullptr;

    switch (type) {
        case RO_BACKEND:
            ctx = (Context *) WGState::instance()->
                allocateDynamicShared(sizeof(ROContext));
            new (ctx) ROContext(*this, SHMEM_CTX_WG_PRIVATE);
            break;
        case GPU_IB_BACKEND:
            ctx = (Context *) WGState::instance()->
                allocateDynamicShared(sizeof(GPUIBContext));
            new (ctx) GPUIBContext(*this, SHMEM_CTX_WG_PRIVATE);
            break;
        default:
            assert(false);
    }

    __syncthreads();

    if (is_thread_zero_in_block())
        WGState::instance()->set_private_ctx(ctx);

    __syncthreads();
}

__device__ void
Backend::finalize_wg_state()
{
    WGState::instance()->return_buffers();
}
