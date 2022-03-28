/******************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "backend_bc.hpp"

#include "backend_type.hpp"
#include "context_incl.hpp"
#include "wg_state.hpp"

namespace rocshmem {

Backend::Backend(size_t num_wgs) {
    int num_cus {};
    if (hipDeviceGetAttribute(&num_cus,
        hipDeviceAttributeMultiprocessorCount, 0)) {
        exit(-static_cast<int>(Status::ROC_SHMEM_UNKNOWN_ERROR));
    }
    /*
     * FIXME: Do not use this hard-coded '32'.
     */
    int max_num_wgs {num_cus * 32};

    if (num_wgs > max_num_wgs || num_wgs == 0) {
        num_wgs = max_num_wgs;
    }

    CHECK_HIP(hipGetDevice(&hip_dev_id));

    /*
     * FIXME: kludge to support the default context.
     *
     * The kludge works because we allocate enough queues to cover num_wgs
     * contexts, so this ensures we always have enough extra ones for the
     * default context.
     *
     * This is bad because it also over allocates any buffers that are
     * dependent on num_wgs but not actually related to contexts.
     */
    num_wgs++;

    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&bufferTokens),
                        sizeof(unsigned int) * num_wgs));

    for (int i {0}; i < num_wgs; i++) {
        bufferTokens[i] = 0;
    }

    num_wg = num_wgs;

    /*
     * Initialize 'print_lock' global and copy to the device memory space.
     */
    CHECK_HIP(hipMalloc(&print_lock, sizeof(*print_lock)));
    *print_lock = 0;

    int* print_lock_addr {nullptr};
    CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&print_lock_addr),
                                  HIP_SYMBOL(print_lock)));

    CHECK_HIP(hipMemcpy(print_lock_addr,
                        &print_lock,
                        sizeof(print_lock),
                        hipMemcpyDefault));

    /*
     * Copy this Backend object to 'backend_device_proxy' global in the
     * device memory space to provide a device-side handle to Backend.
     */
    int* device_backend_proxy_addr {nullptr};
    CHECK_HIP(hipGetSymbolAddress(
                  reinterpret_cast<void**>(&device_backend_proxy_addr),
                  HIP_SYMBOL(device_backend_proxy)));

    Backend* this_temp_addr {this};
    CHECK_HIP(hipMemcpy(device_backend_proxy_addr,
                        &this_temp_addr,
                        sizeof(this),
                        hipMemcpyDefault));

    CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&done_init),
                            sizeof(uint8_t)));

    /*
     * Notify other threads that Backend has been initialized.
     */
    *done_init = 0;
}

Backend::~Backend() {
    CHECK_HIP(hipFree(print_lock));
    CHECK_HIP(hipFree(bufferTokens));
}

Status
Backend::dump_stats() {
    printf("PE %d\n", my_pe);

    const auto& device_stats {globalStats};
    printf("DEVICE STATS\n");
    printf("Puts (Blocking/P/Nbi) %llu/%llu/%llu\n",
           device_stats.getStat(NUM_PUT),
           device_stats.getStat(NUM_P),
           device_stats.getStat(NUM_PUT_NBI));
    printf("WG_Puts (Blocking/Nbi) %llu/%llu\n",
           device_stats.getStat(NUM_PUT_WG),
           device_stats.getStat(NUM_PUT_NBI_WG));
    printf("WAVE_Puts (Blocking/Nbi) %llu/%llu\n",
           device_stats.getStat(NUM_PUT_WAVE),
           device_stats.getStat(NUM_PUT_NBI_WAVE));
    printf("Gets (Blocking/G/Nbi) %llu/%llu/%llu\n",
           device_stats.getStat(NUM_GET),
           device_stats.getStat(NUM_G),
           device_stats.getStat(NUM_GET_NBI));
    printf("WG_Gets (Blocking/Nbi) %llu/%llu\n",
           device_stats.getStat(NUM_GET_WG),
           device_stats.getStat(NUM_GET_NBI_WG));
    printf("WAVE_Gets (Blocking/Nbi) %llu/%llu\n",
           device_stats.getStat(NUM_GET_WAVE),
           device_stats.getStat(NUM_GET_NBI_WAVE));
    printf("Fences %llu\n", device_stats.getStat(NUM_FENCE));
    printf("Quiets %llu\n", device_stats.getStat(NUM_QUIET));
    printf("ToAll %llu\n", device_stats.getStat(NUM_TO_ALL));
    printf("BarrierAll %llu\n", device_stats.getStat(NUM_BARRIER_ALL));
    printf("Wait Until %llu\n", device_stats.getStat(NUM_WAIT_UNTIL));
    printf("Finalizes %llu\n", device_stats.getStat(NUM_FINALIZE));
    printf("Coalesced %llu\n", device_stats.getStat(NUM_MSG_COAL));
    printf("Atomic_FAdd %llu\n", device_stats.getStat(NUM_ATOMIC_FADD));
    printf("Atomic_FCswap %llu\n", device_stats.getStat(NUM_ATOMIC_FCSWAP));
    printf("Atomic_FInc %llu\n", device_stats.getStat(NUM_ATOMIC_FINC));
    printf("Atomic_Fetch %llu\n", device_stats.getStat(NUM_ATOMIC_FETCH));
    printf("Atomic_Add %llu\n", device_stats.getStat(NUM_ATOMIC_ADD));
    printf("Atomic_Cswap %llu\n", device_stats.getStat(NUM_ATOMIC_CSWAP));
    printf("Atomic_Inc %llu\n", device_stats.getStat(NUM_ATOMIC_INC));
    printf("Tests %llu\n", device_stats.getStat(NUM_TEST));
    printf("SHMEM_PTR %llu\n", device_stats.getStat(NUM_SHMEM_PTR));
    printf("SyncAll %llu\n", device_stats.getStat(NUM_SYNC_ALL));

    const auto& host_stats {globalHostStats};
    printf("HOST STATS\n");
    printf("Puts (Blocking/P/Nbi) %llu/%llu/%llu\n",
           host_stats.getStat(NUM_HOST_PUT),
           host_stats.getStat(NUM_HOST_P),
           host_stats.getStat(NUM_HOST_PUT_NBI));
    printf("Gets (Blocking/G/Nbi) (%llu/%llu/%llu)\n",
           host_stats.getStat(NUM_HOST_GET),
           host_stats.getStat(NUM_HOST_G),
           host_stats.getStat(NUM_HOST_GET_NBI));
    printf("Fences %llu\n", host_stats.getStat(NUM_HOST_FENCE));
    printf("Quiets %llu\n", host_stats.getStat(NUM_HOST_QUIET));
    printf("ToAll %llu\n", host_stats.getStat(NUM_HOST_TO_ALL));
    printf("BarrierAll %llu\n", host_stats.getStat(NUM_HOST_BARRIER_ALL));
    printf("Wait Until %llu\n", host_stats.getStat(NUM_HOST_WAIT_UNTIL));
    printf("Finalizes %llu\n", host_stats.getStat(NUM_HOST_FINALIZE));
    printf("Atomic_FAdd %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FADD));
    printf("Atomic_FCswap %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FCSWAP));
    printf("Atomic_FInc %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FINC));
    printf("Atomic_Fetch %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FETCH));
    printf("Atomic_Add %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_ADD));
    printf("Atomic_Cswap %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_CSWAP));
    printf("Atomic_Inc %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_INC));
    printf("Tests %llu\n", host_stats.getStat(NUM_HOST_TEST));
    printf("SHMEM_PTR %llu\n", host_stats.getStat(NUM_HOST_SHMEM_PTR));
    printf("SyncAll %llu\n", host_stats.getStat(NUM_HOST_SYNC_ALL));

    return dump_backend_stats();
}

Status
Backend::reset_stats() {
    globalStats.resetStats();
    globalHostStats.resetStats();

    return reset_backend_stats();
}

__device__ void
Backend::wait_wg_init_done() {
    if (is_thread_zero_in_block()) {
        while (*done_init == 0) {
            __roc_inv();
        }
    }
    __syncthreads();
}

__device__ void
Backend::create_wg_state() {
    WGState::create();

    __syncthreads();

    auto* wg_state {WGState::instance()};

    /*
     * Preallocate a single private context for this workgroup (thread-block)
     * and bind it to the WGState instance.
     *
     * The code below carves the allocation out of the dynamic lds
     * partition and then builds the context object in that memory.
     *
     * FIXME: only initialize the WGState with one thread
     */
    Context* ctx {nullptr};

    switch (type) {
        case BackendType::RO_BACKEND:
            ctx = reinterpret_cast<Context*>(
                      wg_state->allocateDynamicShared(sizeof(ROContext)));
            new (ctx) ROContext(*this, ROC_SHMEM_CTX_WG_PRIVATE);
            break;
        case BackendType::GPU_IB_BACKEND:
            ctx = reinterpret_cast<Context*>(
                      wg_state->allocateDynamicShared(sizeof(GPUIBContext)));
            new (ctx) GPUIBContext(*this, ROC_SHMEM_CTX_WG_PRIVATE);
            break;
        //default:
            //assert(false);
    }

    __syncthreads();

    wg_state->team_ctxs_policy.init(team_tracker.get_max_num_teams());

    if (is_thread_zero_in_block()) {
        wg_state->set_private_ctx(ctx);
    }

    __syncthreads();
}

__device__ void
Backend::finalize_wg_state() {
    WGState::instance()->return_buffers();
}

}  // namespace rocshmem
