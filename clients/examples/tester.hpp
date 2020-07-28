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

#ifndef _TESTER_HPP_
#define _TESTER_HPP_

#include <roc_shmem.hpp>

#include "tester_arguments.hpp"

/******************************************************************************
 * TESTER CLASS TYPES
 *****************************************************************************/
enum TestType
{
    GetTestType             = 0,
    GetNBITestType          = 1,
    PutTestType             = 2,
    PutNBITestType          = 3,
    GetSwarmTestType        = 4,
    ReductionTestType       = 5,
    AMO_FAddTestType        = 6,
    AMO_FIncTestType        = 7,
    AMO_FetchTestType       = 8,
    AMO_FCswapTestType      = 9,
    AMO_AddTestType         = 10,
    AMO_IncTestType         = 11,
    AMO_CswapTestType       = 12,
    InitTestType            = 13,
    PingPongTestType        = 14,
    BarrierTestType         = 15,
    RandomAccessTestType    = 16
};

enum OpType
{
    PutType =0,
    GetType =1
};

/******************************************************************************
 * TESTER INTERFACE
 *****************************************************************************/
class Tester
{
  public:
    explicit Tester(TesterArguments args);
    virtual ~Tester();

    void
    execute();

    static Tester*
    create(TesterArguments args);

  protected:
    virtual void
    resetBuffers() = 0;

    virtual void
    launchKernel(dim3 gridSize,
                 dim3 blockSize,
                 int loop,
                 uint64_t size) = 0;

    virtual void
    verifyResults(uint64_t size) = 0;

    int num_msgs = 0;
    int num_timed_msgs = 0;

    TesterArguments args;

    TestType _type;

    hipStream_t stream;

    uint64_t *timer = nullptr;

  private:
    void
    print(uint64_t size);

    void
    barrier();

    uint64_t
    gpuCyclesToMicroseconds(uint64_t cycles);

    uint64_t
    timerAvgInMicroseconds();

    bool
    peLaunchesKernel();

    hipEvent_t start_event;
    hipEvent_t stop_event;
};

#endif /* _TESTER_HPP */
