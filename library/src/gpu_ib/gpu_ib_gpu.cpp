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

#include <hip/hip_runtime.h>
#include <roc_shmem.hpp>

#include "config.h"  // NOLINT(build/include_subdir)
#include "context_incl.hpp"
#include "backend_ib.hpp"
#include "backend_type.hpp"
#include "wg_state.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

/*
 * TODO(bpotter): these will go in a policy class based on DC/RC.
 * I am not completely sure at this point what else is needed in said class,
 * so just leave them up here for now.
 */
__device__ __host__ QueuePair*
GPUIBContext::getQueuePair(int pe) {
    return networkImpl.getQueuePair(device_qp_proxy, pe);
}

__device__ __host__ int
GPUIBContext::getNumQueuePairs() {
    return networkImpl.getNumQueuePairs();
}

__device__ __host__ int
GPUIBContext::getNumDest() {
    return networkImpl.getNumDest();
}

__host__
GPUIBContext::GPUIBContext(const Backend &backend,
                           int64_t options)
    : Context(backend, true) {
    type = BackendType::GPU_IB_BACKEND;

    GPUIBBackend* b = static_cast<GPUIBBackend*>(const_cast<Backend*>(&backend));
    int buffer_id = b->num_wg - 1;
    b->bufferTokens[buffer_id] = 1;
    networkImpl = b->networkImpl;
    base_heap = b->heap.get_heap_bases().data();
    networkImpl.networkHostInit(this, buffer_id);

    barrier_sync = b->barrier_sync;
    ipcImpl_.ipc_bases = b->ipcImpl.ipc_bases;
}

__device__
GPUIBContext::GPUIBContext(const Backend &b,
                           int64_t option)
    : Context(b, false) {
    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();

    __syncthreads();

    GPUIBBackend *roc_shmem_handle =
        static_cast<GPUIBBackend*>(device_backend_proxy);

    type = BackendType::GPU_IB_BACKEND;

    networkImpl = roc_shmem_handle->networkImpl;

    auto *wg_state = WGState::instance();
    size_t dyn_heap_bytes = num_pes * sizeof(char*);
    auto *uncast_heap_ptr = wg_state->allocateDynamicShared(dyn_heap_bytes);
    char **heap_bases = reinterpret_cast<char**>(uncast_heap_ptr);

    const auto& handle_heap_bases = roc_shmem_handle->heap.get_heap_bases();
    for (int i = thread_id; i < num_pes; i = i + block_size) {
        heap_bases[i] = handle_heap_bases[i];
    }
    ipcImpl_.ipcGpuInit(roc_shmem_handle, this, thread_id);

    int remote_conn = getNumQueuePairs();
    size_t dyn_conn_bytes = remote_conn * sizeof(QueuePair);
    auto *uncast_conn_bytes = wg_state->allocateDynamicShared(dyn_conn_bytes);
    device_qp_proxy = reinterpret_cast<QueuePair*>(uncast_conn_bytes);

    /*
     * Reserve free QPs to form my context.
     */
    if (thread_id == 0) {
        int buffer_id = wg_state->get_global_buffer_id();
        /*
         * Copy construct reserved QPs from backend into this context.
         */
        base_heap = heap_bases;
        networkImpl.networkGpuInit(this, buffer_id);

        barrier_sync = roc_shmem_handle->barrier_sync;
    }
    __syncthreads();
}

__device__ void
GPUIBContext::ctx_destroy() {
    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();

    __syncthreads();

    for (int i = thread_id; i < getNumQueuePairs(); i += block_size) {
        auto *qp = getQueuePair(i);
        qp->~QueuePair();
    }

    __syncthreads();
}

__device__
GPUIBContext::~GPUIBContext() {
    ctx_destroy();
}

__device__ void
GPUIBContext::fence() {
    fence_.flush();
}

__device__ void
GPUIBContext::putmem_nbi(void *dest,
                         const void *source,
                         size_t nelems,
                         int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];

    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset,
                         const_cast<void*>(source),
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe,
                                                   source,
                                                   dest,
                                                   nelems);
        if (!must_send_message) {
            return;
        }

        auto *qp = getQueuePair(pe);
        qp->put_nbi<THREAD>(base_heap[pe] + L_offset,
                            source,
                            nelems,
                            pe,
                            true);
    }
}

__device__ void
GPUIBContext::getmem_nbi(void *dest,
                         const void *source,
                         size_t nelems,
                         int pe) {
    const char *src_typed = reinterpret_cast<const char*>(source);
    uint64_t L_offset = const_cast<char*>(src_typed) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy(dest,
                         ipcImpl_.ipc_bases[pe] + L_offset,
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe,
                                                   source,
                                                   dest,
                                                   nelems);
        if (!must_send_message) {
            return;
        }

        auto *qp = getQueuePair(pe);
        qp->get_nbi<THREAD>(base_heap[pe] + L_offset,
                            dest,
                            nelems,
                            pe,
                            true);
    }
}

__device__ void
GPUIBContext::quiet() {
    for (int k = 0; k < getNumDest(); k++) {
        getQueuePair(k)->quiet_single_heavy<THREAD>(k);
    }
    fence_.flush();
}

__device__ void*
GPUIBContext::shmem_ptr(const void* dest,
                        int pe) {
    void *ret = nullptr;
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        void *dst = const_cast<void*>(dest);
        uint64_t L_offset = reinterpret_cast<char*>(dst) - base_heap[my_pe];
        ret = ipcImpl_.ipc_bases[pe] + L_offset;
    }
    return ret;
}

__device__ void
GPUIBContext::threadfence_system() {
    int thread_id = get_flat_block_id();

    if (thread_id % WF_SIZE == lowerID()) {
        getQueuePair(my_pe)->hdp_policy.flushCoherency();
    }
    __threadfence_system();
}

__device__ void
GPUIBContext::getmem(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe) {
    const char *src_typed = reinterpret_cast<const char*>(source);
    uint64_t L_offset = const_cast<char*>(src_typed) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy(dest,
                         ipcImpl_.ipc_bases[pe] + L_offset,
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe,
                                                   source,
                                                   dest,
                                                   nelems);
        if (!must_send_message) {
            return;
        }
        auto *qp = getQueuePair(pe);
        qp->get_nbi_cqe<THREAD>(base_heap[pe] + L_offset,
                                dest,
                                nelems,
                                pe,
                                true);
        qp->quiet_single<THREAD>();
    }
    fence_.flush();
}


__device__ void
GPUIBContext::putmem(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset,
                         const_cast<void*>(source),
                         nelems);
    } else {
        bool must_send_message = wf_coal_.coalesce(pe,
                                                   source,
                                                   dest,
                                                   nelems);
        if (!must_send_message) {
            return;
        }
        auto *qp = getQueuePair(pe);
        qp->put_nbi_cqe<THREAD>(base_heap[pe] + L_offset,
                                source,
                                nelems,
                                pe,
                                true);
        qp->quiet_single<THREAD>();
    }
    fence_.flush();
}

__device__ int64_t
GPUIBContext::amo_fetch_add(void *dst,
                            int64_t value,
                            int64_t cond,
                            int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dst) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        auto uncast = ipcImpl_.ipc_bases[pe] + L_offset;
        auto ipc_offset =
            reinterpret_cast<unsigned long long*>(uncast);  // NOLINT
        return ipcImpl_.ipcAMOFetchAdd(ipc_offset,
            static_cast<unsigned long long>(value));  // NOLINT
    } else {
        auto *qp = getQueuePair(pe);
        return qp->atomic_fetch(base_heap[pe] + L_offset,
                                value,
                                cond,
                                pe,
                                true,
                                MLX5_OPCODE_ATOMIC_FA);
    }
}

__device__ int64_t
GPUIBContext::amo_fetch_cas(void *dst,
                            int64_t value,
                            int64_t cond,
                            int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dst) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        auto uncast = ipcImpl_.ipc_bases[pe] + L_offset;
        auto ipc_offset =
            reinterpret_cast<unsigned long long*>(uncast); // NOLINT
        return ipcImpl_.ipcAMOFetchCas(ipc_offset,
            static_cast<unsigned long long>(cond),  // NOLINT
            static_cast<unsigned long long>(value));  // NOLINT
    } else {
        auto *qp = getQueuePair(pe);
        return qp->atomic_fetch(base_heap[pe] + L_offset,
                                value,
                                cond,
                                pe,
                                true,
                                MLX5_OPCODE_ATOMIC_CS);
    }
}

__device__ void
GPUIBContext::amo_add(void *dst,
                      int64_t value,
                      int64_t cond,
                      int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dst) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        auto ipc_offset = ipcImpl_.ipc_bases[pe] + L_offset;
        auto *ipc_off_ull =
            reinterpret_cast<unsigned long long*>(ipc_offset);  // NOLINT
        ipcImpl_.ipcAMOAdd(ipc_off_ull,
                           static_cast<unsigned long long >(value));  // NOLINT
    } else {
        auto *qp = getQueuePair(pe);
        qp->atomic_nofetch(base_heap[pe] + L_offset,
                           value,
                           cond,
                           pe,
                           true,
                           MLX5_OPCODE_ATOMIC_FA);
    }
}

__device__ void
GPUIBContext::amo_cas(void *dst,
                      int64_t value,
                      int64_t cond,
                      int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dst) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        auto ipc_offset = ipcImpl_.ipc_bases[pe] + L_offset;
        auto *ipc_off_ull =
            reinterpret_cast<unsigned long long*>(ipc_offset);  // NOLINT
        ipcImpl_.ipcAMOCas(ipc_off_ull,
                           static_cast<unsigned long long>(cond),  // NOLINT
                           static_cast<unsigned long long>(value));  // NOLINT
    } else {
        auto *qp = getQueuePair(pe);
        qp->atomic_nofetch(base_heap[pe] + L_offset,
                           value,
                           cond,
                           pe,
                           true,
                           MLX5_OPCODE_ATOMIC_CS);
    }
}

/******************************************************************************
 ************************ WORKGROUP/WAVE-LEVEL RMA API ************************
 *****************************************************************************/
__device__ void
GPUIBContext::putmem_nbi_wg(void *dest,
                            const void *source,
                            size_t nelems,
                            int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[pe] + L_offset,
                            const_cast<void*>(source),
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            auto *qp = getQueuePair(pe);
            qp->put_nbi<WG>(base_heap[pe] + L_offset,
                            source,
                            nelems,
                            pe,
                            true);
        }
    }
    __syncthreads();
}

__device__ void
GPUIBContext::putmem_nbi_wave(void *dest,
                              const void *source,
                              size_t nelems,
                              int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[pe] + L_offset,
                              const_cast<void*>(source),
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            auto *qp = getQueuePair(pe);
            qp->put_nbi<WAVE>(base_heap[pe] + L_offset,
                              source,
                              nelems,
                              pe,
                              true);
        }
    }
}

__device__ void
GPUIBContext::putmem_wg(void *dest,
                        const void *source,
                        size_t nelems,
                        int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
    auto *qp = getQueuePair(pe);
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[pe] + L_offset,
                            const_cast<void*>(source),
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            qp->put_nbi_cqe<WG>(base_heap[pe] + L_offset,
                                source,
                                nelems,
                                pe,
                                true);
        }
    qp->quiet_single<WG>();
    }
    __syncthreads();
    fence_.flush();
}

__device__ void
GPUIBContext::putmem_wave(void *dest,
                          const void *source,
                          size_t nelems,
                          int pe) {
    uint64_t L_offset = reinterpret_cast<char*>(dest) - base_heap[my_pe];
    auto *qp = getQueuePair(pe);
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[pe] + L_offset,
                              const_cast<void*>(source),
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            qp->put_nbi_cqe<WAVE>(base_heap[pe] + L_offset,
                                  source,
                                  nelems,
                                  pe,
                                  true);
        }
    qp->quiet_single<WAVE>();
    }
    fence_.flush();
}

__device__ void
GPUIBContext::getmem_wg(void *dest,
                        const void *source,
                        size_t nelems,
                        int pe) {
    const char *src_typed = reinterpret_cast<const char*>(source);
    uint64_t L_offset = const_cast<char*>(src_typed) - base_heap[my_pe];
    auto *qp = getQueuePair(pe);
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wg(dest,
                            ipcImpl_.ipc_bases[pe] + L_offset,
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            qp->get_nbi_cqe<WG>(base_heap[pe] + L_offset,
                                dest,
                                nelems,
                                pe,
                                true);
        }
    qp->quiet_single<WG>();
    }
    __syncthreads();
    fence_.flush();
}

__device__ void
GPUIBContext::getmem_wave(void *dest,
                          const void *source,
                          size_t nelems,
                          int pe) {
    const char *src_typed = reinterpret_cast<const char*>(source);
    uint64_t L_offset = const_cast<char*>(src_typed) - base_heap[my_pe];
    auto *qp = getQueuePair(pe);
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wave(dest,
                              ipcImpl_.ipc_bases[pe] + L_offset,
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            qp->get_nbi_cqe<WAVE>(base_heap[pe] + L_offset,
                                  dest,
                                  nelems,
                                  pe,
                                  true);
        }
    qp->quiet_single<WAVE>();
    }
    fence_.flush();
}

__device__ void
GPUIBContext::getmem_nbi_wg(void *dest,
                            const void *source,
                            size_t nelems,
                            int pe) {
    const char *src_typed = reinterpret_cast<const char*>(source);
    uint64_t L_offset = const_cast<char*>(src_typed) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wg(dest,
                            ipcImpl_.ipc_bases[pe] + L_offset,
                            nelems);
    } else {
        if (is_thread_zero_in_block()) {
            auto *qp = getQueuePair(pe);
            qp->get_nbi<WG>(base_heap[pe] + L_offset,
                            dest,
                            nelems,
                            pe,
                            true);
        }
    }
    __syncthreads();
}

__device__ void
GPUIBContext::getmem_nbi_wave(void *dest,
                              const void *source,
                              size_t nelems,
                              int pe) {
    const char *src_typed = reinterpret_cast<const char*>(source);
    uint64_t L_offset = const_cast<char*>(src_typed) - base_heap[my_pe];
    if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        ipcImpl_.ipcCopy_wave(dest,
                              ipcImpl_.ipc_bases[pe] + L_offset,
                              nelems);
    } else {
        if (is_thread_zero_in_wave()) {
            auto *qp = getQueuePair(pe);
            qp->get_nbi<WAVE>(base_heap[pe] + L_offset,
                              dest,
                              nelems,
                              pe,
                              true);
        }
    }
}

}  // namespace rocshmem
