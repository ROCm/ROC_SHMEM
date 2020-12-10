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

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include <hip/hip_runtime.h>
#include <roc_shmem.hpp>

#include "context.hpp"
#include "backend.hpp"
#include "wg_state.hpp"
#include "queue_pair.hpp"
#include "util.hpp"

/*
 * TODO: These will go in a policy class based on DC/RC. I'm not completely
 * sure at this point what else is needed in said class, so just leave them
 * up here for now.
 */
__device__ __host__ QueuePair*
GPUIBContext::getQueuePair(int pe)
{
#ifdef USE_DC
    return rtn_gpu_handle;
#else
    return &rtn_gpu_handle[pe];
#endif
}

__device__ __host__ int
GPUIBContext::getNumQueuePairs()
{
#ifdef USE_DC
    return 1;
#else
    return num_pes;
#endif
}

__host__
GPUIBContext::GPUIBContext(const Backend &backend, long options)
    : Context(backend, true)
{
    type = BackendType::GPU_IB_BACKEND;

    int remote_conn = getNumQueuePairs();

    CHECK_HIP(hipMalloc(&rtn_gpu_handle, sizeof(QueuePair) * remote_conn));

    const GPUIBBackend* b = static_cast<const GPUIBBackend *>(&backend);

    int buffer_id = b->num_wg - 1;
    b->bufferTokens[buffer_id] = 1;

    for (int i = 0; i < getNumQueuePairs(); i++) {
        /*
         * RC gpu_qp is actually [NUM_PE][NUM_WG] qps but is flattend. Each
         * num_pe entry contains num_wg QPs connected to that PE. For RC need
         * to iterate gpu_qp[i][buffer_id] to collect a single QP for each
         * connected PE in order to build context. For DC NUM_PE = 1 so can
         * just use buffer_id directly.
         */
        int offset = b->num_wg * i + buffer_id;
        new (getQueuePair(i)) QueuePair(b->gpu_qps[offset]);
        getQueuePair(i)->global_qp = &b->gpu_qps[offset];
        getQueuePair(i)->num_cqs = getNumQueuePairs();
        getQueuePair(i)->atomic_ret.atomic_base_ptr =
            &b->atomic_ret->atomic_base_ptr[max_nb_atomic * buffer_id];
    }

    base_heap = b->heap_bases;
    barrier_sync = b->barrier_sync;
    current_heap_offset = b->current_heap_offset;
    g_ret = b->g_ret;
    ipcImpl.ipc_bases = b->ipcImpl.ipc_bases;
}

__device__
GPUIBContext::GPUIBContext(const Backend &b, long option)
    : Context(b, false)
{
    GPU_DPRINTF("Function: gpu_ib_ctx_create\n");

    int i = 0;
    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();

    __syncthreads();

    GPUIBBackend* roc_shmem_handle = static_cast<GPUIBBackend *>(gpu_handle);

    type = BackendType::GPU_IB_BACKEND;

    int remote_conn = getNumQueuePairs();

    char ** heap_bases = (char **) WGState::instance()->allocateDynamicShared(
        sizeof(*heap_bases) * num_pes);

    for (i = thread_id; i < num_pes; i = i + block_size) {
        heap_bases[i] = roc_shmem_handle->heap_bases[i];
    }
    ipcImpl.ipcGpuInit(roc_shmem_handle, this, thread_id);
    rtn_gpu_handle = reinterpret_cast<QueuePair*>(
        WGState::instance()->allocateDynamicShared(
        sizeof(QueuePair) * remote_conn));

    /*
     * Reserve free QPs to form my context.
     */
    if (thread_id == 0) {
        int buffer_id = WGState::instance()->get_global_buffer_id();
        /*
         * Copy construct reserved QPs from backend into this context.
         */
        for (int i = 0; i < getNumQueuePairs(); i++) {
            int offset = roc_shmem_handle->num_wg * i + buffer_id;
            new (getQueuePair(i))
                QueuePair(roc_shmem_handle->gpu_qps[offset]);

            getQueuePair(i)->global_qp = &roc_shmem_handle->gpu_qps[offset];
            getQueuePair(i)->num_cqs = getNumQueuePairs();
            getQueuePair(i)->atomic_ret.atomic_base_ptr =
                &roc_shmem_handle->atomic_ret->atomic_base_ptr
                [max_nb_atomic * buffer_id];
        }

        base_heap = heap_bases;
        barrier_sync = roc_shmem_handle->barrier_sync;
        current_heap_offset =
            roc_shmem_handle->current_heap_offset;
        g_ret = roc_shmem_handle->g_ret;
    }
    __syncthreads();
}

__device__ void
GPUIBContext::ctx_destroy()
{
    GPU_DPRINTF("Function: gpu_ib_ctx_destroy\n");

    GPUIBBackend* roc_shmem_handle = static_cast<GPUIBBackend *>(gpu_handle);

    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();

    for (int i = thread_id; i < getNumQueuePairs(); i += block_size)
        getQueuePair(i)->~QueuePair();

    __syncthreads();
}

__device__ void
GPUIBContext::fence()
{
    //ipcImpl.ipcFence();
    for (int i = 0; i < getNumQueuePairs(); i++)
        rtn_gpu_handle[i].fence(i);
    flushStores();
}

__device__ void
GPUIBContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_putmem_nbi\n");

    uint64_t L_offset = (char *) dest - base_heap[my_pe];

    if (ipcImpl.isIpcAvailable(my_pe, pe)) {
        ipcImpl.ipcCopy(ipcImpl.ipc_bases[pe] + L_offset, (void*) source, nelems);
    } else {
        bool must_send_message = wf_coal.coalesce(pe, source, dest, nelems);
        if (!must_send_message) {
            return;
        }

        getQueuePair(pe)->put_nbi(base_heap[pe] + L_offset,
                                  source, nelems, pe, true);
    }
}

__device__ void
GPUIBContext::getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_getmem_nbi\n");

    uint64_t L_offset = (char *) source - base_heap[my_pe];
    if (ipcImpl.isIpcAvailable(my_pe, pe)) {
        ipcImpl.ipcCopy(dest, ipcImpl.ipc_bases[pe] + L_offset, nelems);
    } else {
        bool must_send_message = wf_coal.coalesce(pe, source, dest, nelems);
        if (!must_send_message) {
            return;
        }

        getQueuePair(pe)->get_nbi(base_heap[pe] + L_offset, dest,
                              nelems, pe, true);
    }
}

__device__ void
GPUIBContext::quiet()
{
    //ipcImpl.ipcFence();
    for (int i = 0; i < getNumQueuePairs(); i++)
        rtn_gpu_handle[i].quiet_single();
    flushStores();
}

__device__ void*
GPUIBContext::shmem_ptr(const void* dest, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_shmem_ptr\n");

    void * ret = NULL;
    if(ipcImpl.isIpcAvailable(my_pe, pe)){
        uint64_t L_offset = (char *) dest - base_heap[my_pe];
        ret = ipcImpl.ipc_bases[pe] + L_offset;
    }
    return ret;
}

__device__ void
GPUIBContext::threadfence_system() { }

__device__ void
GPUIBContext::getmem(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_getmem\n");
    getmem_nbi(dest, source, nelems, pe);
    //ipcImpl.ipcFence();

    getQueuePair(pe)->quiet_single();
    flushStores();
}

__device__ void
GPUIBContext::putmem(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_putmem\n");
    putmem_nbi(dest, source, nelems, pe);
    //ipcImpl.ipcFence();

    getQueuePair(pe)->quiet_single();
    flushStores();
}

__device__ int64_t
GPUIBContext::amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe)
{
    uint64_t L_offset = (char *) dst - base_heap[my_pe];
    if(ipcImpl.isIpcAvailable(my_pe, pe)){
        return ipcImpl.ipcAMOFetchAdd(reinterpret_cast<unsigned long long*>
                                      (ipcImpl.ipc_bases[pe]+ L_offset),
                                      static_cast<unsigned long long >(value));
    }else{
        return getQueuePair(pe)->atomic_fetch(base_heap[pe] + L_offset, value,
                                              cond, pe, true,
                                              MLX5_OPCODE_ATOMIC_FA);
    }
}

__device__ int64_t
GPUIBContext::amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    uint64_t L_offset = (char *) dst - base_heap[my_pe];
    if(ipcImpl.isIpcAvailable(my_pe, pe)){
        return ipcImpl.ipcAMOFetchCas(reinterpret_cast<unsigned long long*>
                                      (ipcImpl.ipc_bases[pe]+ L_offset),
                                      static_cast<unsigned long long >(cond),
                                      static_cast<unsigned long long >(value));
    }else{
        return getQueuePair(pe)->atomic_fetch(base_heap[pe] + L_offset, value,
                                              cond, pe, true,
                                              MLX5_OPCODE_ATOMIC_CS);
    }
}

__device__ void
GPUIBContext::amo_add(void *dst, int64_t value, int64_t cond, int pe)
{
    uint64_t L_offset = (char *) dst - base_heap[my_pe];
    if(ipcImpl.isIpcAvailable(my_pe, pe)){
        ipcImpl.ipcAMOAdd(reinterpret_cast<unsigned long long*>
                          (ipcImpl.ipc_bases[pe]+ L_offset),
                          static_cast<unsigned long long >(value));
    }else{
        getQueuePair(pe)->atomic_nofetch(base_heap[pe] + L_offset, value,
                                         cond, pe, true,
                                         MLX5_OPCODE_ATOMIC_FA);
    }
}

__device__ void
GPUIBContext::amo_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    uint64_t L_offset = (char *) dst - base_heap[my_pe];
    if(ipcImpl.isIpcAvailable(my_pe, pe)){
        ipcImpl.ipcAMOCas(reinterpret_cast<unsigned long long*>
                          (ipcImpl.ipc_bases[pe]+ L_offset),
                          static_cast<unsigned long long >(cond),
                          static_cast<unsigned long long >(value));
    }else{
        getQueuePair(pe)->atomic_nofetch(base_heap[pe] + L_offset, value,
                                         cond, pe, true,
                                         MLX5_OPCODE_ATOMIC_CS);
    }
}
