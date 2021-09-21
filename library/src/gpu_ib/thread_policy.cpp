/******************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#include "config.h"  // NOLINT(build/include_subdir)

#include "queue_pair.hpp"
#include "thread_policy.hpp"

__device__ void
SingleThreadImpl::quiet(QueuePair *handle) {
    handle->quiet_internal<THREAD>();
}

__device__ void
SingleThreadImpl::quiet_heavy(QueuePair *handle, int pe) {
    handle->zero_b_rd<THREAD>(pe);
    handle->quiet_internal<THREAD>();
}

__device__ void
MultiThreadImpl::quiet(QueuePair *handle) {
    int thread_id = get_flat_block_id();
    /*
     * Each WF selects one thread to perform the quiet.  Only one thread
     * per WG is allowed to do a quiet at once to avoid races with the CQ.
     */
    if (thread_id % WF_SIZE == lowerID()) {
        while (atomicCAS(&(handle->threadImpl.cq_lock), 0, 1) == 1) {
        }
        handle->quiet_internal<THREAD>();
        __threadfence();
        handle->threadImpl.cq_lock = 0;
    }
}

__device__ void
MultiThreadImpl::quiet_heavy(QueuePair *handle, int pe) {
    int thread_id = get_flat_block_id();
    /*
     * Each WF selects one thread to perform the quiet.  Only one thread
     * per WG is allowed to do a quiet at once to avoid races with the CQ.
     */
    if (thread_id % WF_SIZE == lowerID()) {
        // zero_byte read
        handle->zero_b_rd<THREAD>(pe);

        while (atomicCAS(&(handle->threadImpl.cq_lock), 0, 1) == 1) {
        }
        handle->quiet_internal<THREAD>();
        __threadfence();
        handle->threadImpl.cq_lock = 0;
    }
}


__device__ void
WG::quiet(QueuePair *handle) {
    handle->quiet_internal<WG>();
}

__device__ void
WG::quiet_heavy(QueuePair *handle, int pe) {
    handle->zero_b_rd<THREAD>(pe);
    handle->quiet_internal<WG>();
}


__device__ void
WAVE::quiet(QueuePair *handle) {
    int thread_id = get_flat_block_id();
    /*
     * Each WF selects one thread to perform the quiet.  Only one thread
     * per WG is allowed to do a quiet at once to avoid races with the CQ.
     */

    if (thread_id % WF_SIZE == 0) {
        while (atomicCAS(&(handle->threadImpl.cq_lock), 0, 1) == 1) {
        }
        handle->quiet_internal<WAVE>();
        __threadfence();
        handle->threadImpl.cq_lock = 0;
    }
}

__device__ void
WAVE::quiet_heavy(QueuePair *handle, int pe) {
    int thread_id = get_flat_block_id();
    /*
     * Each WF selects one thread to perform the quiet.  Only one thread
     * per WG is allowed to do a quiet at once to avoid races with the CQ.
     */
    if (thread_id % WF_SIZE == 0) {
        // post a zero-byte read
        handle->zero_b_rd<THREAD>(pe);
        while (atomicCAS(&(handle->threadImpl.cq_lock), 0, 1) == 1) {
        }
        handle->quiet_internal<WAVE>();
        __threadfence();
        handle->threadImpl.cq_lock = 0;
    }
}

__device__ void
SingleThreadImpl::decQuietCounter(uint32_t *quiet_counter, int num) {
    *quiet_counter -= num;
}

__device__ void
MultiThreadImpl::decQuietCounter(uint32_t *quiet_counter, int num) {
    atomicSub(quiet_counter, num);
}

__device__ void
WG::decQuietCounter(uint32_t *quiet_counter, int num) {
    atomicSub(quiet_counter, num);
}

__device__ void
WAVE::decQuietCounter(uint32_t *quiet_counter, int num) {
    *quiet_counter -= num;
}

template<bool cqe>
__device__ void
SingleThreadImpl::finishPost(QueuePair *handle,
                             bool ring_db,
                             int num_wqes,
                             int pe,
                             uint16_t le_sq_counter,
                             uint8_t opcode) {
    if (ring_db) {
        uint64_t db_val = handle->db_val;
        handle->compute_db_val_opcode(&db_val, le_sq_counter, opcode);
        handle->update_wqe_ce<cqe>(num_wqes);
        handle->ring_doorbell(db_val);
    }
}

template<bool cqe>
__device__ void
MultiThreadImpl::finishPost(QueuePair *handle,
                            bool ring_db,
                            int num_wqes,
                            int pe,
                            uint16_t le_sq_counter,
                            uint8_t opcode) {
    /*
     * For RC, we can't allow a wave to have different PEs in it, else the
     * doorbell ringing logic will not work.  This little for loop forces
     * control flow divergence based on the PE.  It works well for small
     * numbers of PEs, but we might want a different solution for large
     * numbers.
     */
    if (handle->connection_policy.forcePostDivergence()) {
        for (int i = 0; i < handle->num_cqs; i++) {
            if (i != pe) {
                continue;
            }

            finishPost_internal<cqe>(handle,
                                     ring_db,
                                     num_wqes,
                                     pe,
                                     le_sq_counter,
                                     opcode);
        }
    } else {
            finishPost_internal<cqe>(handle,
                                     ring_db,
                                     num_wqes,
                                     pe,
                                     le_sq_counter,
                                     opcode);
    }
}

template<bool cqe>
__device__ void
MultiThreadImpl::finishPost_internal(QueuePair *handle,
                                     bool ring_db,
                                     int num_wqes,
                                     int pe,
                                     uint16_t le_sq_counter,
                                     uint8_t opcode) {
    /*
     * Assuming here that postLock locks out all wavefronts in this WG but
     * one, and that this will select a single thread in the wavefront.
     */
    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        if (ring_db) {
            uint64_t db_val =
                handle->current_sq[8 * ((handle->sq_counter - num_wqes)
                % handle->max_nwqe)];
            handle->update_wqe_ce<cqe>(num_wqes);
            handle->ring_doorbell(db_val);
        }

        handle->threadImpl.sq_lock = 0;
    }
}

template <bool cqe>
__device__ void
WG::finishPost(QueuePair *handle,
               bool ring_db,
               int num_wqes,
               int pe,
               uint16_t le_sq_counter,
               uint8_t opcode) {
    if (ring_db) {
        uint64_t db_val =
            handle->current_sq[8 * ((handle->sq_counter - num_wqes)
            % handle->max_nwqe)];
            handle->update_wqe_ce<cqe>(num_wqes);
        handle->ring_doorbell(db_val);
    }
}

template <bool cqe>
__device__ void
WAVE::finishPost(QueuePair *handle,
                 bool ring_db,
                 int num_wqes,
                 int pe,
                 uint16_t le_sq_counter,
                 uint8_t opcode) {
    if (ring_db) {
        uint64_t db_val =
            handle->current_sq[8 * ((handle->sq_counter - num_wqes)
            % handle->max_nwqe)];
            handle->update_wqe_ce<cqe>(num_wqes);
        handle->ring_doorbell(db_val);
    }
    handle->threadImpl.sq_lock = 0;
}

__device__ void
SingleThreadImpl::postLock(QueuePair *handle, int pe) {
    handle->hdp_policy.hdp_flush();
    handle->waitCQSpace(1);
}

__device__ void
MultiThreadImpl::postLock_internal(QueuePair *handle) {
    int thread_id = get_flat_block_id();
    int active_threads = wave_SZ();

    if (thread_id % WF_SIZE == lowerID()) {
        handle->hdp_policy.hdp_flush();
        /*
         * Don't let more than one wave in this WG go any further or a
         * horrible variety of impossible to debug race conditions can occur.
         */
        while (atomicCAS(&(handle->threadImpl.sq_lock), 0, 1) == 1) {
        }

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
    if (active_threads != wave_SZ()) {
        __builtin_trap();
    }
}

__device__ void
MultiThreadImpl::postLock(QueuePair *handle, int pe) {
    /*
     * For RC, we can't allow a wave to have different PEs in it, else the
     * doorbell ringing logic will not work.  This little for loop forces
     * control flow divergence based on the PE.  It works well for small
     * numbers of PEs, but we might want a different solution for large
     * numbers.
     */
    if (handle->connection_policy.forcePostDivergence()) {
        for (int i = 0; i < handle->num_cqs; i++) {
            if (i != pe) {
                continue;
            }

            postLock_internal(handle);
        }
    } else {
            postLock_internal(handle);
    }
}

__device__ void
WG::postLock(QueuePair *handle, int pe) {
    handle->hdp_policy.hdp_flush();
    handle->waitCQSpace(1);
}

__device__ void
WAVE::postLock(QueuePair *handle, int pe) {
    handle->hdp_policy.hdp_flush();

    /*
    * Don't let more than one wave in this WG go any further or a horrible
    * variety of impossible to debug race conditions can occur.
    */
    while (atomicCAS(&(handle->threadImpl.sq_lock), 0, 1) == 1) {
    }

    /*
     * This is a tiny bit over-aggressive as it assumes that all of the
     * active_threads are going to the same PE when calculating whether
     * we are full.
     */
    handle->waitCQSpace(1);
}

template <typename T>
__device__ T
SingleThreadImpl::threadAtomicAdd(T *val, T value) {
    T old_val = *val;
    *val += value;
    return old_val;
}

template <typename T>
__device__ T
MultiThreadImpl::threadAtomicAdd(T *val, T value) {
    return atomicAdd(val, value);
}

template <typename T>
__device__ T
WG::threadAtomicAdd(T *val, T value) {
    T old_val = *val;
    *val += value;
    return old_val;
}

template <typename T>
__device__ T
WAVE::threadAtomicAdd(T *val, T value) {
    return atomicAdd(val, value);
}

#define TYPE_GEN(T) \
    template \
    __device__ T \
    SingleThreadImpl::threadAtomicAdd<T>(T *val, T value); \
    template \
    __device__ T \
    MultiThreadImpl::threadAtomicAdd<T>(T *val, T value); \
    template \
    __device__ T \
    WG::threadAtomicAdd<T>(T *val, T value); \
    template \
    __device__ T \
    WAVE::threadAtomicAdd<T>(T *val, T value);

TYPE_GEN(float)
TYPE_GEN(double)
TYPE_GEN(int)
TYPE_GEN(unsigned int)
TYPE_GEN(unsigned long long)  // NOLINT(runtime/int)

#define TYPE_BOOL(T) \
    template \
    __device__ void \
    SingleThreadImpl::finishPost<T>(QueuePair *handle, \
                                    bool ring_db, \
                                    int num_wqes, \
                                    int pe, \
                                    uint16_t le_sq_counter, \
                                    uint8_t opcode); \
    template \
    __device__ void \
    MultiThreadImpl::finishPost<T>(QueuePair *handle, \
                                   bool ring_db, \
                                   int num_wqes, \
                                   int pe, \
                                   uint16_t le_sq_counter, \
                                   uint8_t opcode); \
    template \
    __device__ void \
    WG::finishPost<T>(QueuePair *handle, \
                      bool ring_db, \
                      int num_wqes, \
                      int pe, \
                      uint16_t le_sq_counter, \
                      uint8_t opcode) ;\
    template \
    __device__ void \
    WAVE::finishPost<T>(QueuePair *handle, \
                        bool ring_db, \
                        int num_wqes, \
                        int pe, \
                        uint16_t le_sq_counter, \
                        uint8_t opcode); \
    template \
    __device__ void \
    MultiThreadImpl::finishPost_internal<T>(QueuePair *handle, \
                                            bool ring_db, \
                                            int num_wqes, \
                                            int pe, \
                                            uint16_t le_sq_counter, \
                                            uint8_t opcode);

TYPE_BOOL(true)
TYPE_BOOL(false)
