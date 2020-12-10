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

#include "wg_state.hpp"
#include "context.hpp"
#include "backend.hpp"

__device__ void
WGState::create()
{
    if (is_thread_zero_in_block())
        new (instance()) WGState();

    __syncthreads();
}

__device__
WGState::WGState()
{
    HIP_DYNAMIC_SHARED(char, baseDynamicPtr);

    // Set the heap base to account for space used by this context
    dynamicPtr = baseDynamicPtr + sizeof(*this);
    buffer_id = reserve_wg_buffers(gpu_handle->num_wg);
}

__device__ void
WGState::return_buffers()
{
    wg_ctx->ctx_destroy();

    if (is_thread_zero_in_block())
        gpu_handle->bufferTokens[buffer_id] = 0;

    __syncthreads();
}

__device__ int
WGState::get_global_buffer_id() const
{
    return buffer_id;
}

__device__ char *
WGState::allocateDynamicShared(size_t size)
{
    if (is_thread_zero_in_block()) {
        dynamicPtr += size;
         // ROCm (as of 3.0) device-side printf doesn't handle %p format for
         // the char * data type correctly, so we need to cast to some other
         // type (e.g. void *) to make this work.
        GPU_DPRINTF("Allocating %u bytes dynamic LDS.  Heap ptr at %p.\n",
                    size, (void *) dynamicPtr);
    }

    // dynamicPtr is updated for all threads after this call and can be
    // returned per-thread.
    __syncthreads();

    return dynamicPtr - size;
}

__device__ unsigned int
WGState::reserve_wg_buffers(int num_buffers)
{
    // Try to reserve a wg global buffers for this context.
    // In RECYCLE_QUEUES mode, each WG fights for ownership of wg buffers
    // with all other WGs and returns the wg buffers to the free pool of
    // buffers when the WG terminates.
    //
    // The first buffer we try to get is always based on our WV slot ID.
    // We essentially try to "bind" buffers to hardware slots so that when
    // a WG finishes, the WG that is scheduled to replace it always gets
    // the same buffer, so that there is no contention when the total number
    // of buffers is >= the maximum number of WGs that can be scheduled on
    // the hardware.  We couldn't do this based on logical grid IDs since
    // there is no correspondence between WG IDs that finish and WG IDs
    // that are scheduled to replace them.
    int hw_wv_slot = get_hw_wv_index();
    int buffer_index = hw_wv_slot % num_buffers;

    // If the number of buffers are <= the maximum number of WGs that can
    // be scheduled, then we are going to end up fighting with other WGs
    // for them.  Iterate over all available buffer tokens and find an
    // avilable buffer.
    while (atomicCAS(&gpu_handle->bufferTokens[buffer_index], 0, 1) == 1)
        buffer_index = (buffer_index + 1) % num_buffers;

    return buffer_index;
}

__device__ WGState *
WGState::instance()
{
   /*
    * WGState is allocated at the start of the dynamic shared segment.  Each
    * work-group that calls this function will receive its own private
    * WGState.
    */
   HIP_DYNAMIC_SHARED(WGState, wg_state);
   return wg_state;
}
