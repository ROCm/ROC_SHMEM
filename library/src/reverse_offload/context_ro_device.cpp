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

#include "config.h"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <roc_shmem.hpp>

#include "backend_type.hpp"
#include "backend_proxy.hpp"
#include "ro_net_internal.hpp"
#include "ro_net_team.hpp"
#include "hdp_policy.hpp"
#include "context_ro_device.hpp"
#include "reverse_offload/backend_ro.hpp"
#include "wg_state.hpp"

namespace rocshmem {

__host__
ROContext::ROContext(Backend *b, long option)
  : Context(b, true) {
    ROBackend *backend {static_cast<ROBackend*>(b)};
    BackendProxyT &backend_proxy {backend->backend_proxy};
    auto *proxy {backend_proxy.get()};

    CHECK_HIP(hipMalloc(&backend_ctx, sizeof(ro_net_wg_handle)));

    type = BackendType::RO_BACKEND;

    int buffer_id = b->num_wg - 1;
    b->bufferTokens[buffer_id] = 1;

    backend_ctx->write_idx = proxy->queue_descs[buffer_id].write_idx;
    backend_ctx->read_idx = proxy->queue_descs[buffer_id].read_idx;
    backend_ctx->status = proxy->queue_descs[buffer_id].status;
    backend_ctx->host_read_idx = &proxy->queue_descs[buffer_id].read_idx;
    backend_ctx->queue = proxy->queues[buffer_id];
    backend_ctx->queue_size = proxy->queue_size;
    backend_ctx->barrier_ptr = proxy->barrier_ptr;
    backend_ctx->g_ret = proxy->g_ret;
    backend_ctx->atomic_ret.atomic_base_ptr = proxy->atomic_ret->atomic_base_ptr;
    backend_ctx->atomic_ret.atomic_counter = 0;
    ipcImpl_.ipc_bases = b->ipcImpl.ipc_bases;
    backend_ctx->profiler.resetStats();
}

__device__
ROContext::ROContext(Backend *b, long option)
    : Context(b, false) {
    ROBackend *backend {static_cast<ROBackend*>(b)};
    BackendProxyT &backend_proxy {backend->backend_proxy};
    auto *proxy {backend_proxy.get()};

    int thread_id = get_flat_block_id();

    GPU_DPRINTF("Function: ro_ctx_create\n");

    backend_ctx = reinterpret_cast<ro_net_wg_handle *>(
        WGState::instance()->allocateDynamicShared(sizeof(ro_net_wg_handle)));

    ipcImpl_.ipcGpuInit(static_cast<Backend*>(device_backend_proxy),
                        this,
                        thread_id);

    if (is_thread_zero_in_block()) {
        type = BackendType::RO_BACKEND;

        int buffer_id = WGState::instance()->get_global_buffer_id();

        backend_ctx->write_idx = proxy->queue_descs[buffer_id].write_idx;
        backend_ctx->read_idx = proxy->queue_descs[buffer_id].read_idx;
        backend_ctx->status = proxy->queue_descs[buffer_id].status;
        backend_ctx->host_read_idx = &proxy->queue_descs[buffer_id].read_idx;
        backend_ctx->queue = proxy->queues[buffer_id];
        backend_ctx->queue_size = proxy->queue_size;
        backend_ctx->barrier_ptr = proxy->barrier_ptr;
        backend_ctx->g_ret = proxy->g_ret;
        backend_ctx->atomic_ret.atomic_base_ptr = proxy->atomic_ret->atomic_base_ptr;
        backend_ctx->atomic_ret.atomic_counter = proxy->atomic_ret->atomic_counter;
        backend_ctx->profiler.resetStats();
        // TODO: @Brandon Assuming that I am GPU 0, need ID for multi-GPU nodes!
        new (&backend_ctx->hdp_policy) HdpPolicy(*proxy->hdp_policy);
    }
    __syncthreads();
}

__device__ void
ROContext::threadfence_system() {
    int thread_id = get_flat_block_id();
    if (thread_id % WF_SIZE == lowerID()) {
        backend_ctx->hdp_policy.flushCoherency();
    }
    __threadfence_system();
}

__device__ void
ROContext::putmem(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dest) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset,
                         const_cast<void*>(source),
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe, source, dest, nelems);
        if (!must_send_message) {
            return;
        }
        build_queue_element(RO_NET_PUT,
                            dest,
                            (void*)source,
                            nelems,
                            pe,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true);
    }
}

__device__ void
ROContext::getmem(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        const char *src_typed = reinterpret_cast<const char*>(source);
        uint64_t L_offset = const_cast<char*>(src_typed) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy(dest,
                         ipcImpl_.ipc_bases[pe] + L_offset,
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe, source, dest, nelems);
        if (!must_send_message) {
            return;
        }
        build_queue_element(RO_NET_GET,
                            dest,
                            (void*)source,
                            nelems,
                            pe,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true);
    }
}

__device__ void
ROContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dest) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset,
                         const_cast<void*>(source),
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe, source, dest, nelems);
        if (!must_send_message) {
            return;
        }
        build_queue_element(RO_NET_PUT_NBI,
                            dest,
                            (void*)source,
                            nelems,
                            pe,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            false);
    }
}

__device__ void
ROContext::getmem_nbi(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        const char *src_typed = reinterpret_cast<const char*>(source);
        uint64_t L_offset = const_cast<char*>(src_typed) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy(dest,
                         ipcImpl_.ipc_bases[pe] + L_offset,
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe, source, dest, nelems);
        if (!must_send_message) {
            return;
        }
        build_queue_element(RO_NET_GET_NBI,
                            dest,
                            (void*)source,
                            nelems,
                            pe,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            false);
    }
}

__device__ void
ROContext::fence() {
    build_queue_element(RO_NET_FENCE,
                        nullptr,
                        nullptr,
                        0,
                        0,
                        0,
                        0,
                        0,
                        nullptr,
                        nullptr,
                        (MPI_Comm)NULL,
                        ro_net_win_id,
                        backend_ctx,
                        true);
}

__device__ void
ROContext::fence(int pe) {
    // TODO (khamidou): need to check if per pe has any special handling
    build_queue_element(RO_NET_FENCE,
                        nullptr,
                        nullptr,
                        0,
                        0,
                        0,
                        0,
                        0,
                        nullptr,
                        nullptr,
                        (MPI_Comm)NULL,
                        ro_net_win_id,
                        backend_ctx,
                        true);
}

__device__ void
ROContext::quiet() {
    build_queue_element(RO_NET_QUIET,
                        nullptr,
                        nullptr,
                        0,
                        0,
                        0,
                        0,
                        0,
                        nullptr,
                        nullptr,
                        (MPI_Comm)NULL,
                        ro_net_win_id,
                        backend_ctx,
                        true);
}

__device__ void *
ROContext::shmem_ptr(const void *dest, int pe) {
    void *ret = nullptr;
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        void *dst = const_cast<void*>(dest);
        uint64_t L_offset = reinterpret_cast<char*>(dst) - ipcImpl_.ipc_bases[my_pe];
        ret = ipcImpl_.ipc_bases[pe] + L_offset;
    }
    return ret;
}

__device__ void
ROContext::barrier_all() {
    if(is_thread_zero_in_block()) {
        build_queue_element(RO_NET_BARRIER_ALL,
                            nullptr,
                            nullptr,
                            0,
                            0,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true);
    }
    __syncthreads();
}

__device__ void
ROContext::sync_all() {
    if(is_thread_zero_in_block()) {
        build_queue_element(RO_NET_BARRIER_ALL,
                            nullptr,
                            nullptr,
                            0,
                            0,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true);
    }
    __syncthreads();
}

__device__ void
ROContext::sync(roc_shmem_team_t team)
{
    ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
    if(is_thread_zero_in_block()) {
        build_queue_element(RO_NET_SYNC,
                            nullptr,
                            nullptr,
                            0,
                            0,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            team_obj->mpi_comm,
                            ro_net_win_id,
                            backend_ctx,
                            true);
    }
    __syncthreads();
}

__device__ void
ROContext::ctx_create() {
    if (is_thread_zero_in_block()) {
        ROBackend *backend {static_cast<ROBackend*>(device_backend_proxy)};
        BackendProxyT &backend_proxy {backend->backend_proxy};
        auto *proxy {backend_proxy.get()};

        int *pool_alloc_mask = proxy->win_pool_alloc_bitmask;
        int num_ctxs = proxy->max_num_ctxs;

        /*
         * Loop over the mask and find an available index
         * 0: Not allocated (available)
         * 1: Allocated (not available)
         */
        for (int i {0}; i < num_ctxs; i++) {
            if (0 == atomicCAS(&pool_alloc_mask[i], 0, 1)) {
                /* Found an available window */
                ro_net_win_id = i;
                break;
            }
        }
    }
    __syncthreads();
}

__device__ void
ROContext::ctx_destroy() {
    if (is_thread_zero_in_block()) {
        ROBackend *backend {static_cast<ROBackend*>(device_backend_proxy)};
        BackendProxyT &backend_proxy {backend->backend_proxy};
        auto *proxy {backend_proxy.get()};

        build_queue_element(RO_NET_FINALIZE,
                            nullptr,
                            nullptr,
                            0,
                            0,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true);

        /* Mark the allocated window as available */
        proxy->win_pool_alloc_bitmask[ro_net_win_id] = 0;

        int buffer_id = WGState::instance()->get_global_buffer_id();
        proxy->queue_descs[buffer_id].write_idx = backend_ctx->write_idx;

        ROStats &global_handle = proxy->profiler[buffer_id];
        global_handle.accumulateStats(backend_ctx->profiler);
    }

    __syncthreads();
}

__device__ int64_t
ROContext::amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dst) - ipcImpl_.ipc_bases[my_pe];
        auto uncast = ipcImpl_.ipc_bases[pe] + L_offset;
        auto ipc_offset = reinterpret_cast<unsigned long long*>(uncast); // NOLINT
        return ipcImpl_.ipcAMOFetchCas(ipc_offset,
                                       static_cast<unsigned long long>(cond),  // NOLINT
                                       static_cast<unsigned long long>(value));  // NOLINT
    } else {
         uint64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>( /* NOLINT(runtime/int) */
                                  &backend_ctx->atomic_ret.atomic_counter), 1);

        pos = pos % max_nb_atomic;

        int64_t *atomic_base_ptr = reinterpret_cast<int64_t*>(backend_ctx->atomic_ret.atomic_base_ptr);
        int64_t *source = &atomic_base_ptr[pos];

        build_queue_element(RO_NET_AMO_FCAS,
                            dst,
                            (void*)source,
                            value,
                            pe,
                            0,
                            0,
                            0,
                            (void*)cond,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true);

        __threadfence();
        return *source;
    }
}

__device__ void
ROContext::amo_cas(void *dst, int64_t value, int64_t cond, int pe) {
    assert(0);
}

__device__ int64_t
ROContext::amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dst) - ipcImpl_.ipc_bases[my_pe];
        auto uncast = ipcImpl_.ipc_bases[pe] + L_offset;
        auto ipc_offset = reinterpret_cast<unsigned long long*>(uncast);  // NOLINT
        return ipcImpl_.ipcAMOFetchAdd(ipc_offset,
                                       static_cast<unsigned long long>(value));  // NOLINT
    } else {
        uint64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>( /* NOLINT(runtime/int) */
                                 &backend_ctx->atomic_ret.atomic_counter), 1);

        pos = pos % max_nb_atomic;

        int64_t *atomic_base_ptr = reinterpret_cast<int64_t*>(backend_ctx->atomic_ret.atomic_base_ptr);
        int64_t *source = &atomic_base_ptr[pos];

        build_queue_element(RO_NET_AMO_FOP,
                            dst,
                            (void*)source,
                            value,
                            pe,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true,
                            ROC_SHMEM_SUM);

        __threadfence();
        return *source;
    }
}

__device__ void
ROContext::amo_add(void *dst, int64_t value, int64_t cond, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        int64_t ret;
        uint64_t L_offset = reinterpret_cast<char*>(dst) - ipcImpl_.ipc_bases[my_pe];
        auto uncast = ipcImpl_.ipc_bases[pe] + L_offset;
        auto ipc_offset = reinterpret_cast<unsigned long long*>(uncast);  // NOLINT
        ret = ipcImpl_.ipcAMOFetchAdd(ipc_offset,
                                      static_cast<unsigned long long>(value));  // NOLINT
    } else {
        uint64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>( /* NOLINT(runtime/int) */
                                 &backend_ctx->atomic_ret.atomic_counter), 1);

        pos = pos % max_nb_atomic;

        int64_t *atomic_base_ptr = reinterpret_cast<int64_t*>(backend_ctx->atomic_ret.atomic_base_ptr);
        int64_t *source = &atomic_base_ptr[pos];

        build_queue_element(RO_NET_AMO_FOP,
                            dst,
                            (void*)source,
                            value,
                            pe,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            backend_ctx,
                            true,
                            ROC_SHMEM_SUM);
    }
}

__device__ void
ROContext::putmem_wg(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dest) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[pe] + L_offset,
                            const_cast<void*>(source),
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            build_queue_element(RO_NET_PUT,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                true);
        }
    }
    __syncthreads();
}

__device__ void
ROContext::getmem_wg(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        const char *src_typed = reinterpret_cast<const char*>(source);
        uint64_t L_offset = const_cast<char*>(src_typed) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wg(dest,
                            ipcImpl_.ipc_bases[pe] + L_offset,
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            build_queue_element(RO_NET_GET,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                true);
        }
    }
    __syncthreads();
}

__device__ void
ROContext::putmem_nbi_wg(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dest) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[pe] + L_offset,
                            const_cast<void*>(source),
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            build_queue_element(RO_NET_PUT_NBI,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                false);
        }
    }
    __syncthreads();
}

__device__ void
ROContext::getmem_nbi_wg(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        const char *src_typed = reinterpret_cast<const char*>(source);
        uint64_t L_offset = const_cast<char*>(src_typed) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wg(dest,
                            ipcImpl_.ipc_bases[pe] + L_offset,
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            build_queue_element(RO_NET_GET_NBI,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                false);
        }
    }
    __syncthreads();
}

__device__ void
ROContext::putmem_wave(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dest) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[pe] + L_offset,
                              const_cast<void*>(source),
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            build_queue_element(RO_NET_PUT,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                true);
        }
    }
}

__device__ void
ROContext::getmem_wave(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        const char *src_typed = reinterpret_cast<const char*>(source);
        uint64_t L_offset = const_cast<char*>(src_typed) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wave(dest,
                              ipcImpl_.ipc_bases[pe] + L_offset,
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            build_queue_element(RO_NET_GET,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                true);
        }
    }
}

__device__ void
ROContext::putmem_nbi_wave(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        uint64_t L_offset = reinterpret_cast<char*>(dest) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[pe] + L_offset,
                              const_cast<void*>(source),
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            build_queue_element(RO_NET_PUT_NBI,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                false);
        }
    }
}

__device__ void
ROContext::getmem_nbi_wave(void *dest, const void *source, size_t nelems, int pe) {
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        const char *src_typed = reinterpret_cast<const char*>(source);
        uint64_t L_offset = const_cast<char*>(src_typed) - ipcImpl_.ipc_bases[my_pe];
        ipcImpl_.ipcCopy_wave(dest,
                              ipcImpl_.ipc_bases[pe] + L_offset,
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            build_queue_element(RO_NET_GET_NBI,
                                dest,
                                (void*)source,
                                nelems,
                                pe,
                                0,
                                0,
                                0,
                                nullptr,
                                nullptr,
                                (MPI_Comm)NULL,
                                ro_net_win_id,
                                backend_ctx,
                                false);
        }
    }
}

__device__ bool
isFull(uint64_t read_idx,
       uint64_t write_idx,
       uint64_t queue_size) {
    return ((queue_size - (write_idx - read_idx)) == 0);
}

__device__ void
build_queue_element(ro_net_cmds type,
                    void *dst,
                    void *src,
                    size_t size,
                    int pe,
                    int logPE_stride,
                    int PE_size,
                    int PE_root,
                    void *pWrk,
                    long *pSync,
                    MPI_Comm team_comm,
                    int ro_net_win_id,
                    struct ro_net_wg_handle *handle,
                    bool blocking,
                    ROC_SHMEM_OP op,
                    ro_net_types datatype) {
    int threadId = get_flat_block_id();

    uint64_t start = handle->profiler.startTimer();

    unsigned long long old_write_slot = handle->write_idx;
    unsigned long long write_slot;
    do {
        write_slot = old_write_slot;
        // If we think the queue might be full, poll on the in-memory read
        // index.  Otherwise, we are good to go!  In the common case we never
        // need to go to memory.
        while (isFull(handle->read_idx, write_slot, handle->queue_size)) {
            __asm__ volatile ("global_load_dwordx2 %0 %1 off glc slc\n "
                              "s_waitcnt vmcnt(0)" :
                              "=v"(handle->read_idx) :
                              "v"(handle->host_read_idx));

        }
        // Double check that our write_idx is still available and update it.
        // If it isn't then we try again and validate that the new write
        // index is available for the taking.
        old_write_slot = atomicCAS((unsigned long long*)&handle->write_idx,
                                   write_slot,
                                   write_slot + 1);
    } while (write_slot != old_write_slot);

    handle->profiler.endTimer(start, WAITING_ON_SLOT);

    start = handle->profiler.startTimer();
    write_slot = write_slot % handle->queue_size;
    handle->queue[write_slot].type = type;
    handle->queue[write_slot].PE = pe;
    handle->queue[write_slot].size = size;
    handle->queue[write_slot].dst = dst;

    // Inline commands will pack the data value in the src field.
    if (type == RO_NET_P) {
       memcpy(&handle->queue[write_slot].src, src, size);
    } else {
       handle->queue[write_slot].src = src;
    }

    handle->queue[write_slot].threadId = threadId;

    if (type == RO_NET_AMO_FOP) {
        handle->queue[write_slot].op = op;
    }
    if (type == RO_NET_AMO_FCAS) {
        handle->queue[write_slot].pWrk = pWrk;
    }
    if (type == RO_NET_TO_ALL) {
        handle->queue[write_slot].logPE_stride = logPE_stride;
        handle->queue[write_slot].PE_size = PE_size;
        handle->queue[write_slot].pWrk = pWrk;
        handle->queue[write_slot].pSync = pSync;
        handle->queue[write_slot].op = op;
        handle->queue[write_slot].datatype = datatype;
    }
    if (type == RO_NET_TEAM_TO_ALL) {
        handle->queue[write_slot].op = op;
        handle->queue[write_slot].datatype = datatype;
        handle->queue[write_slot].team_comm = team_comm;
    }
    if (type == RO_NET_BROADCAST) {
        handle->queue[write_slot].logPE_stride = logPE_stride;
        handle->queue[write_slot].PE_size = PE_size;
        handle->queue[write_slot].pSync = pSync;
        handle->queue[write_slot].PE_root = PE_root;
        handle->queue[write_slot].datatype = datatype;
    }
    if (type == RO_NET_TEAM_BROADCAST) {
        handle->queue[write_slot].PE_root = PE_root;
        handle->queue[write_slot].datatype = datatype;
        handle->queue[write_slot].team_comm = team_comm;
    }
    if (type == RO_NET_ALLTOALL) {
        handle->queue[write_slot].datatype = datatype;
        handle->queue[write_slot].team_comm = team_comm;
        handle->queue[write_slot].pWrk = pWrk;
    }
    if (type == RO_NET_FCOLLECT) {
        handle->queue[write_slot].datatype = datatype;
        handle->queue[write_slot].team_comm = team_comm;
        handle->queue[write_slot].pWrk = pWrk;
    }
    if(type == RO_NET_SYNC) {
        handle->queue[write_slot].team_comm = team_comm;
    }

    handle->profiler.endTimer(start, PACK_QUEUE);

    // Make sure queue element data is visible to CPU
    start = handle->profiler.startTimer();
    __threadfence();
    handle->profiler.endTimer(start, THREAD_FENCE_1);

    // Make data as ready and make visible to CPU
    start = handle->profiler.startTimer();
    handle->queue[write_slot].valid = 1;
    __threadfence();
    handle->profiler.endTimer(start, THREAD_FENCE_2);

    // Blocking requires the CPU to complete the operation.
    start = handle->profiler.startTimer();
    if (blocking) {
        int net_status = 0;
        do {
            // TODO: Vega supports 7 bits, Fiji only 4
            __asm__ volatile ("s_sleep 32\n"
                              "global_load_sbyte %0 %1 off glc slc\n "
                                "s_waitcnt vmcnt(0)" :
                                "=v"(net_status) :
                                "v"(&handle->status[threadId]));
        } while (net_status == 0);

        handle->status[threadId] = 0;
        __threadfence();
    }
    handle->profiler.endTimer(start, WAITING_ON_HOST);
}

}  // namespace rocshmem
