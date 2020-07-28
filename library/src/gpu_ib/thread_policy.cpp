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

#include "queue_pair.hpp"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

__device__ void
SingleThreadImpl::quiet(QueuePair* handle)
{ handle->quiet_internal(); }

__device__ void
MultiThreadImpl::quiet(QueuePair* handle)
{
    int thread_id = get_flat_block_id();
    /*
     * Each WF selects one thread to perform the quiet.  Only one thread
     * per WG is allowed to do a quiet at once to avoid races with the CQ.
     */
    if (thread_id % WF_SIZE == lowerID()) {
        while (atomicCAS(&cq_lock, 0, 1) == 1);
        handle->quiet_internal();
        __threadfence();
        cq_lock = 0;
    }
}
__device__ void
SingleThreadImpl::decQuietCounter(uint32_t *quiet_counter, int num)
{
    *quiet_counter -= num;
}

__device__ void
MultiThreadImpl::decQuietCounter(uint32_t *quiet_counter, int num)
{
    atomicSub(quiet_counter, num);
}

void
SingleThreadImpl::setDBval(uint64_t val)
{
    db_val = val;
}

void
MultiThreadImpl::setDBval(uint64_t val)
{ }

__device__ void
SingleThreadImpl::finishPost(QueuePair* handle, bool ring_db,
                             int num_wqes, int pe, uint16_t le_sq_counter,
                             uint8_t opcode)
{
    if (ring_db) {
        uint64_t db_val = this->db_val;
        handle->compute_db_val_opcode(&db_val, le_sq_counter, opcode);
        handle->ring_doorbell(db_val);
    }

    handle->quiet_counter++;
}

__device__ void
MultiThreadImpl::finishPost(QueuePair* handle, bool ring_db,
                            int num_wqes, int pe, uint16_t le_sq_counter,
                            uint8_t opcode)
{
    /*
     * For RC, we can't allow a wave to have different PEs in it, else the
     * doorbell ringing logic will not work.  This little for loop forces
     * control flow divergence based on the PE.  It works well for small
     * numbers of PEs, but we might want a different solution for large
     * numbers.
     */
    if (handle->connection_policy.forcePostDivergence()) {
        for (int i = 0; i < handle->num_cqs; i++) {
            if (i != pe)
                continue;

            finishPost_internal(handle, ring_db, num_wqes, pe, le_sq_counter,
                                opcode);
        }
    } else {
            finishPost_internal(handle, ring_db, num_wqes, pe, le_sq_counter,
                                opcode);
    }
    // Unlock the SQ for other wavefronts.
    if (get_flat_block_id() % WF_SIZE == lowerID())
        sq_lock = 0;
}

__device__ void
MultiThreadImpl::finishPost_internal(QueuePair* handle, bool ring_db,
                                     int num_wqes, int pe,
                                     uint16_t le_sq_counter, uint8_t opcode)
{
   /*
    * Assuming here that postLock locks out all wavefronts in this WG but
    * one, and that this will select a single thread in the wavefront.
    */
    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        int waveSZ = wave_SZ();
        if (ring_db) {
            uint64_t db_val =
                handle->current_sq[8 * ((handle->sq_counter - num_wqes)
                % handle->max_nwqe)];
            handle->ring_doorbell(db_val);
        }

        handle->quiet_counter += waveSZ;
    }
}

__device__ void
SingleThreadImpl::postLock(QueuePair* handle)
{
    handle->hdp_policy.hdp_flush();
    handle->waitCQSpace(1);
}

__device__ void
MultiThreadImpl::postLock(QueuePair* handle)
{
    int thread_id = get_flat_block_id();
    int active_threads = wave_SZ();

    if (thread_id % WF_SIZE == lowerID()) {
        handle->hdp_policy.hdp_flush();
        /*
         * Don't let more than one wave in this WG go any further or a horrible
         * variety of impossible to debug race conditions can occur.
         */
        while(atomicCAS(&sq_lock, 0, 1) == 1);

        /*
         * This is a tiny bit over-aggressive as it assumes that all of the
         * active_threads are going to the same PE when calculating whether
         * we are full.
         */
        handle->waitCQSpace(active_threads);
    }

    /*
     * Double check we've got the same exec mask (assuming divergence after
     * the previous if.
     */
    if (active_threads != wave_SZ())
        __builtin_trap();
}
