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

#include <stdio.h>
#include <stdlib.h>
#include <ro_net.hpp>
#include <ro_net_internal.hpp>
#include <hdp_helper.hpp>
#include <unistd.h>

#define HW_ID_WV_ID_OFFSET 0
#define HW_ID_SD_ID_OFFSET 4
#define HW_ID_CU_ID_OFFSET 8
#define HW_ID_SE_ID_OFFSET 13

#define HW_ID_WV_ID_SIZE 4
#define HW_ID_SD_ID_SIZE 2
#define HW_ID_CU_ID_SIZE 4
#define HW_ID_SE_ID_SIZE 2

#define WVS_PER_SD 10
#define WVS_PER_CU 40
#define WVS_PER_SE 640

__device__ int
get_hw_wv_index() {
    unsigned wv_id, sd_id, cu_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 0, 4)" : "=s"(wv_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 4, 2)" : "=s"(sd_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 2)" : "=s"(se_id));
    
    // Note that we can't use the SIZES above because some of them are over
    // provisioned (i.e. 4 bits for wave but we have only 10) and we have an
    // exact number of queues.
    //return (se_id << (HW_ID_CU_ID_SIZE + HW_ID_SD_ID_SIZE + HW_ID_WV_ID_SIZE)) +
    //        (cu_id << (HW_ID_SD_ID_SIZE + HW_ID_WV_ID_SIZE)) +
    //        (sd_id << (HW_ID_WV_ID_SIZE)) + wv_id;

    return wv_id + sd_id * 10 + cu_id * 40 + se_id * 640;
}

/***
 *
 * External Device-side API functions
 *
 ***/
__device__ void
ro_net_init(ro_net_handle_t* handle_E,
              ro_net_wg_handle_t **wg_handle_E)
{
    struct ro_net_handle * handle = (struct ro_net_handle *) handle_E;
    __shared__ ro_net_wg_handle wg_handle;
    #ifdef RECYCLE_QUEUES
    // Try to reserve a queue for submitted network commands.  We currently
    // require each work-group to have a dedicated queue.  In
    // RECYCLE_QUEUES mode, each WG fights for ownership of a queue with
    // all other WGs and returns the queue to the free pool of queues when
    // the WG terminates.
    //
    // The first queue we try to get is always based on our WV slot ID.
    // We essentially try to "bind" queues to hardware slots so that when
    // a WG finishes, the WG that is scheduled to replace it always gets
    // the same queue, so that there is no contention when the total number
    // of queues is >= the maximum number of WGs that can be scheduled on
    // the hardware.  We couldn't do this based on logical grid IDs since
    // there is no correspondence between WG IDs that finish and WG IDs
    // that are scheduled to replace them.
    int hw_wv_slot = get_hw_wv_index();
    int queue_index = (hw_wv_slot * 64) /
        (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z);
    queue_index %= handle->num_queues;

    // If the number of queues are <= the maximum number of WGs that can
    // be scheduled, then we are going to end up fighting with other WGs
    // for them.  Iterate over all available queue tokens and find an
    // avilable queue.
    while (atomicCAS(&handle->queueTokens[queue_index], 1, 0) == 0)
        queue_index = (queue_index + 1) % handle->num_queues;

    #else
    // Assume we have a queue for each work-group on this grid.  We do
    // not reuse queues or take advantage of the fact that only so many
    // WGs can be scheduled on the GPU at once.
    // TODO: assert size??
    int queue_index =  hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x
        + hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
    #endif

    // Device side memcpy is very slow, so do elementwise copy.
    wg_handle.queueTokenIndex = queue_index;
    wg_handle.write_idx = handle->queue_descs[queue_index].write_idx;
    wg_handle.read_idx = handle->queue_descs[queue_index].read_idx;
    wg_handle.status = handle->queue_descs[queue_index].status;
    wg_handle.host_read_idx = &handle->queue_descs[queue_index].read_idx;
    wg_handle.queue =  handle->queues[queue_index];
    wg_handle.num_pes =  handle->num_pes;
    wg_handle.my_pe =  handle->my_pe;
    wg_handle.queue_size =  handle->queue_size;
    wg_handle.num_queues =  handle->num_queues;
    wg_handle.queueTokens = handle->queueTokens;
    wg_handle.barrier_ptr = handle->barrier_ptr;
    wg_handle.profiler.waitingOnSlot  = 0;
    wg_handle.profiler.threadFence1   = 0;
    wg_handle.profiler.threadFence2   = 0;
    wg_handle.profiler.packQueue      = 0;
    wg_handle.profiler.waitingOnHost  = 0;
    wg_handle.profiler.shmem_wait     = 0;
    wg_handle.profiler.enabled        = 1;
    // TODO: Assuming that I am GPU 0, need ID for multi-GPU nodes!
    wg_handle.hdp_flush = handle->hdp_regs[0].HDP_MEM_FLUSH_CNTL;

    *wg_handle_E = ((ro_net_wg_handle_t*) &wg_handle);

}

__device__ void
ro_net_ctx_create(long option, ro_net_ctx_t *ctx,
                    ro_net_wg_handle_t *wg_handle)
{
    *ctx = (ro_net_ctx_t)wg_handle;
}

__device__ void
ro_net_ctx_destroy(ro_net_ctx_t ctx)
{
}

__device__ void
ro_net_threadfence_system(ro_net_wg_handle_t *handle_E)
{
    struct ro_net_wg_handle * handle =
        (struct ro_net_wg_handle *) handle_E;

    *(handle->hdp_flush) = 0x1;
}

__device__ void
ro_net_putmem(ro_net_ctx_t ctx, void *dst, void *src, int size, int pe)
{
    build_queue_element(RO_NET_PUT, dst, src, size, pe, 0, 0, nullptr,
                        nullptr, (struct ro_net_wg_handle *) ctx, true);
}

__device__ void
ro_net_getmem(ro_net_ctx_t ctx, void *dst, void *src, int size, int pe)
{
    build_queue_element(RO_NET_GET, dst, src, size, pe, 0, 0, nullptr,
                        nullptr, (struct ro_net_wg_handle *) ctx, true);
}

__device__ void
ro_net_putmem_nbi(ro_net_ctx_t ctx, void *dst, void *src, int size, int pe)
{
    build_queue_element(RO_NET_PUT_NBI, dst, src, size, pe, 0, 0, nullptr,
                        nullptr, (struct ro_net_wg_handle *) ctx, false);
}

__device__ void
ro_net_getmem_nbi(ro_net_ctx_t ctx, void *dst, void *src, int size, int pe)
{
    build_queue_element(RO_NET_GET_NBI, dst, src, size, pe, 0, 0, nullptr,
                        nullptr, (struct ro_net_wg_handle *) ctx, false);
}

__device__ void
ro_net_fence(ro_net_ctx_t ctx)
{
    build_queue_element(RO_NET_FENCE, nullptr, nullptr, 0, 0, 0, 0, nullptr,
                        nullptr, (struct ro_net_wg_handle *) ctx, true);
}

__device__ void
ro_net_quiet(ro_net_ctx_t ctx)
{
    build_queue_element(RO_NET_QUIET, nullptr, nullptr, 0, 0, 0, 0, nullptr,
                        nullptr, (struct ro_net_wg_handle *) ctx, true);
}

__device__ void
ro_net_float_sum_to_all(float *dest, float *source, int nreduce,
                          int PE_start, int logPE_stride, int PE_size,
                          float *pWrk, long *pSync,
                          ro_net_wg_handle_t* handle)
{
    build_queue_element(RO_NET_FLOAT_SUM_TO_ALL,
                        dest, source, nreduce, PE_start, logPE_stride, PE_size,
                        pWrk, pSync, (struct ro_net_wg_handle *) handle,
                        true);
}

__device__ void
ro_net_wait_until(ro_net_ctx_t ctx, void *ptr, ro_net_cmps cmp, int val)
{
    struct ro_net_wg_handle * handle = (struct ro_net_wg_handle *) ctx;
    uint64_t start = 0;

    PVAR_START();

    volatile int * int_ptr = (int*) ptr;
    if (cmp == RO_NET_CMP_EQ) {
        while (*int_ptr != val) {
            __ro_inv();
        }
    }
    PVAR_END(handle->profiler.shmem_wait);
}

__device__ void
ro_net_barrier_all(ro_net_wg_handle_t* handle_E)
{
    struct ro_net_wg_handle * handle =
        (struct ro_net_wg_handle *) handle_E;

    build_queue_element(RO_NET_BARRIER_ALL, NULL, NULL, 0, 0, 0, 0, NULL,
                        NULL, handle, true);
}
/*
__device__ void
ro_net_wg_barrier(ro_net_wg_handle_t* handle_E)
{
    struct ro_net_wg_handle * handle =
        (struct ro_net_wg_handle *) handle_E;
    int wgs = handle->my_pe % 64;
    gws_barrier_init(wgs,  hipGridDim_x * hipGridDim_y,  handle->barrier_ptr);
    gws_barrier_wait(wgs,  hipGridDim_x * hipGridDim_y);
    *handle->barrier_ptr = 0;

}
*/
__device__ int
ro_net_n_pes(ro_net_wg_handle_t* handle_E)
{
    struct ro_net_wg_handle * handle =
        (struct ro_net_wg_handle *) handle_E;
    return handle->num_pes;
}

__device__ int
ro_net_my_pe(ro_net_wg_handle_t* handle_E)
{
    struct ro_net_wg_handle * handle =
        (struct ro_net_wg_handle *) handle_E;
    return handle->my_pe;
}

__device__ void
ro_net_finalize(ro_net_handle_t* handle_E,
                  ro_net_wg_handle_t *wg_handle_E)
{
    struct ro_net_handle * handle = (struct ro_net_handle *) handle_E;
    struct ro_net_wg_handle * wg_handle =
        (struct ro_net_wg_handle *) wg_handle_E;

    build_queue_element(RO_NET_FINALIZE, nullptr, nullptr, 0, 0, 0, 0,
                        nullptr, nullptr, wg_handle, true);
    handle->queue_descs[wg_handle->queueTokenIndex].write_idx =
        wg_handle->write_idx;

#ifdef PROFILE
    handle->profiler[wg_handle->queueTokenIndex].waitingOnSlot
        += wg_handle->profiler.waitingOnSlot;
    handle->profiler[wg_handle->queueTokenIndex].threadFence1
        += wg_handle->profiler.threadFence1;
    handle->profiler[wg_handle->queueTokenIndex].threadFence2
        += wg_handle->profiler.threadFence2;
    handle->profiler[wg_handle->queueTokenIndex].packQueue
        += wg_handle->profiler.packQueue;
    handle->profiler[wg_handle->queueTokenIndex].waitingOnHost
        += wg_handle->profiler.waitingOnHost;
    handle->profiler[wg_handle->queueTokenIndex].shmem_wait
        += wg_handle->profiler.shmem_wait;
#endif

    // Make sure queue has updated write_idx before releasing it to another
    // work-group
    __threadfence();

    handle->queueTokens[wg_handle->queueTokenIndex] =  1;

    __threadfence();
}

__device__ uint64_t
ro_net_timer(ro_net_wg_handle_t * wg_handle_E)
{
    struct ro_net_wg_handle * wg_handle =
        (struct ro_net_wg_handle *) wg_handle_E;
    if (wg_handle->profiler.enabled)
        return  __read_clock();
    return 0;
}

/***
 *
 * Internal Device-side API functions
 *
 ***/
__device__ bool isFull(uint64_t read_idx, uint64_t write_idx,
                       uint64_t queue_size) {
    return ((queue_size - (write_idx - read_idx)) == 0);
}

__device__ void build_queue_element(ro_net_cmds type, void* dst, void * src,
                                    size_t size, int pe, int logPE_stride,
                                    int PE_size, void* pWrk,
                                    long *pSync,
                                    struct ro_net_wg_handle *handle,
                                    bool blocking)
{
    uint64_t start = 0;
    int threadId = (hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y)) +
        (hipThreadIdx_y * hipBlockDim_x) + hipThreadIdx_x;

    PVAR_START();
    unsigned long long old_write_slot = handle->write_idx;
    unsigned long long write_slot;
    do {
        write_slot = old_write_slot;
        // If we think the queue might be full, poll on the in-memory read
        // index.  Otherwise, we are good to go!  In the common case we never
        // need to go to memory.
        while (isFull(handle->read_idx, write_slot, handle->queue_size))
        {
//            __ro_inv();
//            handle->read_idx = *handle->host_read_idx;

            __asm__ volatile ("global_load_dwordx2 %0 %1 off glc slc\n "
                              "s_waitcnt vmcnt(0)" :
                              "=v"(handle->read_idx) :
                              "v"(handle->host_read_idx));

        }
        // Double check that our write_idx is still available and update it.
        // If it isn't then we try again and validate that the new write
        // index is available for the taking.
        old_write_slot = atomicCAS((unsigned long long*) &handle->write_idx,
            write_slot, write_slot + 1);
    } while (write_slot != old_write_slot); 

    PVAR_END(handle->profiler.waitingOnSlot);

    PVAR_START();
    write_slot = write_slot % handle->queue_size;
    handle->queue[write_slot].type = type;
    handle->queue[write_slot].PE = pe;
    handle->queue[write_slot].size = size;
    handle->queue[write_slot].dst = dst;
    handle->queue[write_slot].src = src;
    handle->queue[write_slot].threadId = threadId;

    /*
        * TODO: Might be more efficient to inline the stores.
    int1  val1 = make_int1((int)type);
    int2  val2 = make_int2(pe, size);
    long1 val3 = make_long1((long)src);
    long1 val4 = make_long1((long)dst);
    __store_dword(val1,
        (volatile int1*)&(handle->queue[write_slot].type));
    __store_dwordx2(val2,
        (volatile int2*)&(handle->queue[write_slot].PE));
    __store_long_dwordx2(val3,
        (volatile long1*)&(handle->queue[write_slot].src));
    __store_long_dwordx2(val4,
        (volatile long1*)&(handle->queue[write_slot].dst));
    */

    if (type == RO_NET_FLOAT_SUM_TO_ALL) {
        handle->queue[write_slot].logPE_stride = logPE_stride;
        handle->queue[write_slot].PE_size = PE_size;
        handle->queue[write_slot].pWrk = pWrk;
        handle->queue[write_slot].pSync = pSync;
    }
    PVAR_END(handle->profiler.packQueue);

    // Make sure queue element data is visible to CPU
    PVAR_START();
    __threadfence();
    PVAR_END(handle->profiler.threadFence1);

    // Make data as ready and make visible to CPU
    PVAR_START();
    handle->queue[write_slot].valid = 1;
    __threadfence();
    PVAR_END(handle->profiler.threadFence2);

    // Blocking requires the CPU to complete the operation.
    PVAR_START();
    if (blocking) {
        int net_status;
        do {
            // At will take at least 1-2us to satisfy any request, best case.
            // TODO: Vega supports 7 bits, Fiji only 4
            __asm__ volatile ("s_sleep 32\n"
                              "global_load_sbyte %0 %1 off glc slc\n "
                                "s_waitcnt vmcnt(0)" :
                                "=v"(net_status) :
                                "v"(&handle->status[threadId]));
        } while (net_status == 0);
//        while (!handle->status[threadId]) {
//            __ro_inv();
//        }
        handle->status[threadId] = 0;
        __threadfence();
    }
    PVAR_END(handle->profiler.waitingOnHost);
}

/*
#define BARRIER_INIT(num, val) { \
    unsigned int m0_backup, new_m0;\
    __asm__ __volatile__(\
            "s_mov_b32 %0 m0\n"\
            "v_readfirstlane_b32 %1 %2\n"\
            "s_nop 0\n" \
            "s_mov_b32 m0 %1\n" \
            "s_nop 0\n" \
            "ds_gws_init %3 offset:0 gds\n" \
            "s_waitcnt lgkmcnt(0) expcnt(0)\n" \
            "s_mov_b32 m0 %0\n" \
            "s_nop 0" \
            : "=s"(m0_backup), "=s"(new_m0)\
            : "v"(num<<0x10), "{v0}"(val-1)\
            : "memory"); \
}


__device__ void
gws_barrier_init(unsigned int bar_num, unsigned int bar_val,
                 unsigned int *bar_inited)
{

    int wgid = hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
               hipBlockIdx_z * hipGridDim_x * hipGridDim_z;

    if (wgid == 0)
    {
        if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0)
            BARRIER_INIT(bar_num, bar_val);

        __threadfence();

        if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0)
            atomicAdd(bar_inited, 1);
    }

    __threadfence();
    if (hipThreadIdx_x == 0 && hipThreadIdx_y == 0)
    {
        // Wait for WG0 to initialize the barriers
        while (atomicOr(bar_inited, 0) == 0);
    }
    __threadfence();
}

__device__ void
gws_barrier_wait(unsigned int bar_num, unsigned int reset_val)
{
    unsigned int m0_backup;
    unsigned int new_m0;
    // Save off M0 and prepare a new value for it.
    // First part saves off values, second sets M0, waits at the barrier,
    // and then resets M0.
    //
    // Note that we set M0[21:16] instead of M[5:0] to
    // give us the barrier number. The hardware apparently
    // pulls from M[21:16] despite what the documentation says.
    // Similarly, we source the barrier-reset value from VGPR0
    // because, no matter what register this instruction is given,
    // it pulls the value from VGPR0.
    //
    // We are required to force the ds_gws_barrier's reset value into VGPR0
    // (this is the {v0} constraint) due to some unknown hardware issue
    // that we have observed on at least Vega 10, where this instruction
    // will pull whatever value is in VGPR0 to do its reset.
    __asm__ __volatile__(
            "s_mov_b32 %0 m0\n"
            "v_readfirstlane_b32 %1 %2\n"
            "s_nop 0\n"
            "s_mov_b32 m0 %1\n"
            "s_nop 0\n"
            "ds_gws_barrier %3 offset:0 gds\n"
            "s_waitcnt lgkmcnt(0) expcnt(0)\n"
            "s_mov_b32 m0 %0\n"
            "s_nop 0"
            : "=s"(m0_backup), "=s"(new_m0)
            : "v"(bar_num << 0x10), "{v0}"(reset_val-1)
            : "memory");
}
*/
