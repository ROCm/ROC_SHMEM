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
#ifndef THREADPOLICY_H
#define THREADPOLICY_H

#include "config.h"

class QueuePair;

/*
 * GPU single-thread policy class. Only a single work-item per work-group
 * is allowed to call into a ROC_SHMEM function (unless it is specifically
 * called out as a collective API. This thread policy is the fastest but
 * is not as flexible.
 */
class SingleThreadImpl
{
    /*
     * Cached value for the first 8 bytes of the SQ used to ring the doorbell
     * without a global memory access.
     */
    uint64_t db_val = 0;

  public:
    __device__ void quiet(QueuePair* handle);
    __device__ void decQuietCounter(uint32_t *quiet_counter, int num);
    __device__ void finishPost(QueuePair* handle, bool ring_db,
                               int num_wqes, int pe,
                               uint16_t le_sq_counter, uint8_t opcode);
    __device__ void postLock(QueuePair* handle);
    __device__ void setDBval(uint64_t val);

    template <typename T> __device__ T threadAtomicAdd(T *val, T value = 1)
    {
        T old_val = *val;
        *val += value;
        return old_val;
    }
};

/*
 * GPU multi-thread policy class. Multiple work-items per work-group are
 * allowed to call into a ROC_SHMEM function.  A bit slower than its
 * single-thread counterpart but it enables a much more flexible user-facing
 * API.
 */
class MultiThreadImpl
{
    /*
     * Per-wg locks for the CQ and the SQ, repectively.
     */
    uint32_t cq_lock = 0;
    uint32_t sq_lock = 0;

  public:
    __device__ void quiet(QueuePair* handle);
    __device__ void decQuietCounter(uint32_t *quiet_counter, int num);
    __device__ void finishPost(QueuePair* handle, bool ring_db,
                                   int num_wqes, int pe,
                                   uint16_t le_sq_counter, uint8_t opcode);
    __device__ void postLock(QueuePair* handle);
    __device__ void setDBval(uint64_t val);

    template <typename T> __device__ T
    threadAtomicAdd(T *val, T value = 1) { return atomicAdd(val, value); }
};

/*
 * Select which one of our thread policies to use at compile time.
 */
#ifdef _USE_THREADS_
typedef MultiThreadImpl ThreadImpl;
#else
typedef SingleThreadImpl ThreadImpl;
#endif

#endif //THREADPOLICY_H
