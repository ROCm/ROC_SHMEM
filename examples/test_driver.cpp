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

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "hip/hip_runtime.h"

#include "util.h"
#include "roc_shmem.hpp"
#include "tests.hpp"

int main(int argc, char * argv[])
{
    int num_wgs, num_threads, algorithm, numprocs, myid;
    uint64_t max_msg_size;
    int testnum = 0;
    uint64_t size =1;

    setup(argc, argv, &num_wgs, &num_threads, &max_msg_size, &numprocs,
          &myid, &algorithm);

    Tester *test = Tester::Create( (TestType) algorithm, myid);

    if(((TestType) algorithm) == AMO_FAddTestType  ||
        ((TestType) algorithm) == AMO_AddTestType  ||
        ((TestType) algorithm) == AMO_FCswapTestType  ||
        ((TestType) algorithm) == AMO_CswapTestType  ||
        ((TestType) algorithm) == AMO_FIncTestType ||
        ((TestType) algorithm) == AMO_IncTestType  ||
        ((TestType) algorithm) == AMO_FetchTestType||
        ((TestType) algorithm) == PingPongTestType ){
        size = 8;
        max_msg_size = 8;
    }

    test->initBuffers(max_msg_size, wg_size);

    if (((TestType) algorithm) == InitTestType) {
        test->freeBuffers();
        roc_shmem_finalize();
        delete test;
        return 0;
    }

    int num_loops = loop;

    hipStream_t stream;
    hipStreamCreate(&stream);

    hipEvent_t start_event, stop_event;
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);

    uint64_t *timer;
    hipMalloc((void**)&timer, sizeof(uint64_t) * num_wgs);

    for (; size <= max_msg_size; size <<= 1) {

        test->resetBuffers();

        if (size > large_message_size)
            num_loops = loop_large;

        Barrier();

        uint64_t timer_avg;
        float total_kern_time;

        if (myid == 0 || (((TestType) algorithm) == ReductionTestType) ||
             (((TestType) algorithm) == PingPongTestType) )  {

            memset(timer, 0, sizeof(uint64_t) * num_wgs);

            const dim3 blockSize(wg_size, 1, 1);
            const dim3 gridSize(num_wgs, 1, 1);

            hipEventRecord(start_event, stream);

            test->launchKernel(gridSize, blockSize, stream, num_loops, skip,
                               timer, size);

            hipEventRecord(stop_event, stream);

            hipError_t err = hipStreamSynchronize(stream);
            if (err != hipSuccess) printf("error = %d \n", err);

            // Get the average accross each WG
            timer_avg = calcAvg(timer, num_wgs);
            hipEventElapsedTime(&total_kern_time, start_event, stop_event);
            total_kern_time = total_kern_time / 1000;
        }

        Barrier();

        // data validation
        test->verifyResults(myid, size);

        Barrier();

        if (myid == 0) {
            double latency_avg = (1.0 * (timer_avg)) / (test->numMsgs());
            double bandwidth_avg = ((test->numMsgs() * size) /
                (total_kern_time)) / pow(2, 30);

            fprintf(stdout, "\n##### Message Size %lu #####\n", size);

            fprintf(stdout, "%*s%*s\n",
                    FIELD_WIDTH + 1, "Latency AVG (us)",
                    FIELD_WIDTH + 1, "Bandwidth (GB/s)");

            fprintf(stdout, "%*.*f %*.*f\n",
                    FIELD_WIDTH, FLOAT_PRECISION, latency_avg,
                    FIELD_WIDTH, FLOAT_PRECISION, bandwidth_avg);

            roc_shmem_dump_stats();
            roc_shmem_reset_stats();

            fflush(stdout);
        }
    }

    Barrier();
    hipFree(timer);

    test->freeBuffers();
    delete test;

    Barrier();

    roc_shmem_finalize();

    return 0;
}
