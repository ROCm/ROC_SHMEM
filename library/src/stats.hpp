/******************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef STATS_H
#define STATS_H

#include "roc_shmem.hpp"

enum roc_shmem_stats {
    NUM_PUT = 0,
    NUM_PUT_NBI,
    NUM_P,
    NUM_GET,
    NUM_G,
    NUM_GET_NBI,
    NUM_FENCE,
    NUM_QUIET,
    NUM_TO_ALL,
    NUM_BARRIER_ALL,
    NUM_WAIT_UNTIL,
    NUM_FINALIZE,
    NUM_MSG_COAL,
    NUM_ATOMIC_FADD,
    NUM_ATOMIC_FCSWAP,
    NUM_ATOMIC_FINC,
    NUM_ATOMIC_FETCH,
    NUM_ATOMIC_ADD,
    NUM_ATOMIC_CSWAP,
    NUM_ATOMIC_INC,
    NUM_TEST,
    NUM_SYNC_ALL,
    NUM_STATS
};

typedef unsigned long long StatType;

template <int I>
class Stats
{
    StatType stats[I] = {0};

  public:

    __device__
    uint64_t startTimer() const { return roc_shmem_timer(); }

    __device__
    void endTimer(uint64_t start, int index)
    {
        incStat(index, roc_shmem_timer() - start);
    }

    __device__
    void incStat(int index, int value = 1) { atomicAdd(&stats[index], value); }

    __device__
    void accumulateStats(const Stats<I> &otherStats)
    {
        for (int i = 0; i < I; i++)
            incStat(i, otherStats.getStat(i));
    }

    __host__ __device__
    void resetStats() { memset(&stats, 0, sizeof(StatType) * I); }

    __host__ __device__
    StatType getStat(int index) const { return stats[index]; }
};

template <int I>
class NullStats
{
  public:

    __device__ uint64_t startTimer() const { return 0; }
    __device__ void endTimer(uint64_t start, int index) { }
    __device__ void incStat(int index, int value = 1) { }
    __device__ void accumulateStats(const NullStats<I> &otherStats) { }
    __host__ __device__ void resetStats() { }
    __host__ __device__ StatType getStat(int index) const { return 0; }
};

#ifdef PROFILE
typedef Stats<NUM_STATS> ROCStats;
#else
typedef NullStats<NUM_STATS> ROCStats;
#endif

#endif
