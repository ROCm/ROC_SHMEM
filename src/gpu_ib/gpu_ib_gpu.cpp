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

#include "hip/hip_runtime.h"

#include <roc_shmem.hpp>

#include "gpu_ib_internal.hpp"
#include "backend.hpp"
#include "rtn_internal.hpp"


/*
 * TODO: These will go in a policy class based on DC/RC. I'm not completely
 * sure at this point what else is needed in said class, so just leave them
 * up here for now.
 */
__device__ QueuePair*
GPUIBContext::getQueuePair(int pe)
{
#ifdef _USE_DC_
    return rtn_gpu_handle;
#else
    return &rtn_gpu_handle[pe];
#endif
}

__device__ int
GPUIBContext::getNumQueuePairs()
{
#ifdef _USE_DC_
    return 1;
#else
    return num_pes;
#endif
}

__device__ void
GPUIBContext::ctx_create(long option)
{
    GPU_DPRINTF("Function: gpu_ib_ctx_create\n");
    int i = 0;
    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();

    struct roc_shmem* roc_shmem_handle = (struct roc_shmem *)
        gpu_handle->backend_handle;

    type = GPU_IB_BACKEND;

    int remote_conn = getNumQueuePairs();

    char ** heap_bases = (char **)
        allocateDynamicShared(sizeof(*heap_bases) * num_pes);

    for (i = thread_id; i < num_pes; i = i + block_size) {
        heap_bases[i] = roc_shmem_handle->heap_bases[i];
    }
#ifdef _USE_IPC_
    __shared__ uintptr_t ipc_bases[MAX_NUM_GPUs];
    shm_size  = roc_shmem_handle->shm_size;
    for (i = thread_id; i < shm_size; i++)
        ipc_bases[i]=roc_shmem_handle->ipc_bases[i];

     ipc_bases = ipc_bases;
#endif // _USE_IPC_

    rtn_gpu_handle = (QueuePair *)
        allocateDynamicShared(sizeof(QueuePair) * remote_conn);

    int block_id = get_flat_grid_id();
    int l_queue_id = block_id;

    if (thread_id == 0) {
        auto rtn_handle = roc_shmem_handle->rtn_handle;

#ifdef _RECYCLE_QUEUE_
        int cu_idx = get_hw_cu_index();
        if (block_id < roc_shmem_handle->rtn_handle->num_qps) {
            // first WG batch => set up the hardware to queues infommation
            rtn_handle->softohw[block_id] = cu_idx;
            int pos = atomicAdd(rtn_handle->hwtoqueue[cu_idx], 1);
            rtn_handle->hwtoqueue[cu_idx][pos] = l_queue_id;
            //reserve local queue => reserve all remote queues as well for RC
            rtn_handle->queueTocken[l_queue_id] = 1;
        } else {
            int pos = 0;
            do {
                pos++;
                // FIXME: seems broken not to modulo pos
                l_queue_id = rtn_handle->hwtoqueue[cu_idx][pos];
            } while (atomicCAS(&rtn_handle->queueTocken[l_queue_id], 0, 1));
        }
#endif

        queue_id = l_queue_id;

        for (int i = 0; i < getNumQueuePairs(); i++) {
            int offset = rtn_handle->num_qps * i + l_queue_id;
            new (getQueuePair(i))
                QueuePair(roc_shmem_handle->rtn_handle, offset);
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

    struct roc_shmem* roc_shmem_handle =
        (struct roc_shmem *) gpu_handle->backend_handle;

    int thread_id = get_flat_block_id();
    int block_size = get_flat_block_size();

    for (int i = thread_id; i < getNumQueuePairs(); i += block_size)
        getQueuePair(i)->~QueuePair();

    __syncthreads();

#ifdef _RECYCLE_QUEUE_
    if (thread_id == 0)
        roc_shmem_handle->rtn_handle->queueTocken[queue_id] = 0;
#endif

    __syncthreads();
}

__device__ void
GPUIBContext::fence()
{
    for (int i = 0; i < getNumQueuePairs(); i++)
        rtn_gpu_handle[i].fence(i);
}

__device__ void
GPUIBContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_putmem_nbi\n");

    uint64_t L_offset = (char *) dest - base_heap[my_pe];
#ifdef _USE_IPC_
    if ((pe / shm_size) == (my_pe / shm_size))
    {
        memcpy(ipc_bases[pe] + L_offset, source, nelems);
    } else
#endif
    {
        getQueuePair(pe)->put_nbi(base_heap[pe] + L_offset,
                                  source, nelems, pe, true);
    }
}

__device__ void
GPUIBContext::getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_getmem_nbi\n");

    uint64_t L_offset = (char *) source - base_heap[my_pe];
    getQueuePair(pe)->get_nbi(base_heap[pe] + L_offset, dest,
                              nelems, pe, true);
}

__device__ void
GPUIBContext::quiet()
{
    for (int i = 0; i < getNumQueuePairs(); i++)
        rtn_gpu_handle[i].quiet_single();
}

__device__ void
GPUIBContext::threadfence_system() { }

__device__ void
GPUIBContext::getmem(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_getmem\n");
    getmem_nbi(dest, source, nelems, pe);

    getQueuePair(pe)->quiet_single();
}

__device__ void
GPUIBContext::putmem(void *dest, const void *source, size_t nelems, int pe)
{
    GPU_DPRINTF("Function: gpu_ib_putmem\n");
    putmem_nbi(dest, source, nelems, pe);

    getQueuePair(pe)->quiet_single();
}
__device__ int64_t
GPUIBContext::amo_fetch(void *dst, int64_t value, int64_t cond, int pe,
                        uint8_t atomic_op)
{
    uint64_t L_offset = (char *) dst - base_heap[my_pe];
    return getQueuePair(pe)->
        atomic_fetch(base_heap[pe] + L_offset, value,
                     cond, pe, true, atomic_op);
}
__device__ void
GPUIBContext::amo(void *dst, int64_t value, int64_t cond, int pe,
                  uint8_t atomic_op)
{
    uint64_t L_offset = (char *) dst - base_heap[my_pe];
    getQueuePair(pe)->atomic_nofetch(base_heap[pe] + L_offset,
                                     value, cond, pe, true, atomic_op);
}
