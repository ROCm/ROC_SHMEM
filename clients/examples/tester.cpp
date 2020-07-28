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

#include "tester.hpp"

#include <hip/hip_runtime.h>
#include <mpi.h>
#include <roc_shmem.hpp>

#include "primitive_tester.hpp"
#include "primitive_amo_tester.hpp"
#include "ping_pong_tester.hpp"
#include "swarm_tester.hpp"
#include "collective_tester.hpp"
#include "random_access_tester.hpp"

Tester::Tester(TesterArguments args)
    : args(args)
{
    _type = (TestType) args.algorithm;
    hipStreamCreate(&stream);
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    hipMalloc((void**)&timer, sizeof(uint64_t) * args.num_wgs);
}

Tester::~Tester()
{
    hipFree(timer);
}

Tester*
Tester::create(TesterArguments args)
{
    int rank = args.myid;

    if (rank == 0)
        std::cout << "*** Creating Test: ";

    TestType type = (TestType) args.algorithm;

    switch (type) {
        case InitTestType:
            if (rank == 0)
                std::cout << "Init ***" << std::endl;
            return new PrimitiveTester(args);
        case GetTestType:
            if (rank == 0)
                std::cout << "Blocking Gets***" << std::endl;
            return new PrimitiveTester(args);
        case GetNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking Gets***" << std::endl;
            return new PrimitiveTester(args);
        case PutTestType:
            if (rank == 0)
                std::cout << "Blocking Puts***" << std::endl;
            return new PrimitiveTester(args);
        case PutNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking Puts***" << std::endl;
            return new PrimitiveTester(args);
        case GetSwarmTestType:
            if (rank == 0)
                std::cout << "Get Swarm***" << std::endl;
            return new GetSwarmTester(args);
        case ReductionTestType:
            if (rank == 0)
                std::cout << "All-to-All Reduction***" << std::endl;
            return new CollectiveTester(args);
        case AMO_FAddTestType:
            if (rank == 0)
                std::cout << "AMO Fetch_Add***" << std::endl;
            return new PrimitiveAMOTester(args);
        case AMO_FIncTestType:
            if (rank == 0)
                std::cout << "AMO Fetch_Inc***" << std::endl;
            return new PrimitiveAMOTester(args);
        case AMO_FetchTestType:
            if (rank == 0)
                std::cout << "AMO Fetch***" << std::endl;
            return new PrimitiveAMOTester(args);
        case AMO_FCswapTestType:
            if (rank == 0)
                std::cout << "AMO Fetch_CSWAP***" << std::endl;
            return new PrimitiveAMOTester(args);
        case AMO_AddTestType:
            if (rank == 0)
                std::cout << "AMO Add***" << std::endl;
            return new PrimitiveAMOTester(args);
        case AMO_IncTestType:
            if (rank == 0)
                std::cout << "AMO Inc***" << std::endl;
            return new PrimitiveAMOTester(args);
        case AMO_CswapTestType:
            if (rank == 0)
                std::cout << "AMO Cswap***" << std::endl;
            return new PrimitiveAMOTester(args);
        case PingPongTestType:
            if (rank == 0)
                std::cout << "PingPong***" << std::endl;
            return new PingPongTester(args);
         case BarrierTestType:
            if (rank == 0)
                std::cout << "Barrier_All***" << std::endl;
            return new CollectiveTester(args);
         case RandomAccessTestType:
            if (rank == 0)
                std::cout << "Random_Access***" << std::endl;
            return new RandomAccessTester(args);
        default:
            if (rank == 0)
                std::cout << "Unknown***" << std::endl;
            return new PrimitiveTester(args);
    }
}

void
Tester::execute()
{
    if (_type == InitTestType)
        return;

    int num_loops = args.loop;

    /**
     * Some tests loop through data sizes in powers of 2 and report the
     * results for those ranges.
     */
    for (uint64_t size = args.min_msg_size;
         size <= args.max_msg_size;
         size <<= 1) {

        resetBuffers();

        /**
         * Restricts the number of iterations of really large messages.
         */
        if (size > args.large_message_size)
            num_loops = args.loop_large;

        barrier();

        /**
         * This conditional launches the HIP kernel.
         *
         * Some tests may only launch a single kernel. These kernels will
         * be kicked off by the initiator (denoted by the args.myid check).
         *
         * Other tests will initiate of both sides and launch from both
         * rocshmem pes.
         */
        if (peLaunchesKernel()) {
            /**
             * TODO:
             * Verify that this timer type is actually uint64_t on the
             * device side.
             */
            memset(timer, 0, sizeof(uint64_t) * args.num_wgs);

            const dim3 blockSize(args.wg_size, 1, 1);
            const dim3 gridSize(args.num_wgs, 1, 1);

            hipEventRecord(start_event, stream);

            launchKernel(gridSize, blockSize, num_loops, size);

            hipEventRecord(stop_event, stream);

            hipError_t err = hipStreamSynchronize(stream);
            if (err != hipSuccess) {
                printf("error = %d \n", err);
            }

        }

        barrier();

        // data validation
        verifyResults(size);

        barrier();

        print(size);
    }
}

bool
Tester::peLaunchesKernel()
{
    bool is_launcher;

    /**
     * The PE assigned 0 is always active in these tests.
     */
    is_launcher = args.myid == 0;

    /**
     * Some test types are active on both sides.
     */
    is_launcher = is_launcher ||
                  (_type == ReductionTestType) ||
                  (_type == PingPongTestType)  ||
                  (_type == BarrierTestType)   ||
                  (_type == RandomAccessTestType);

    return is_launcher;
}

void
Tester::print(uint64_t size)
{
    if (args.myid != 0) {
        return;
    }

    uint64_t timer_avg = timerAvgInMicroseconds();
    double latency_avg = static_cast<double>(timer_avg) / num_timed_msgs;

    float total_kern_time_ms;
    hipEventElapsedTime(&total_kern_time_ms, start_event, stop_event);
    float total_kern_time_us = total_kern_time_ms / 1000;
    double bandwidth_avg_gbs = num_msgs * size / total_kern_time_us / pow(2, 30);

    int field_width = 20;
    int float_precision = 2;

    printf("\n##### Message Size %lu #####\n", size);

    printf("%*s%*s\n",
           field_width + 1, "Latency AVG (us)",
           field_width + 1, "Bandwidth (GB/s)");

    printf("%*.*f %*.*f\n",
           field_width, float_precision, latency_avg,
           field_width, float_precision, bandwidth_avg_gbs);

    roc_shmem_dump_stats();
    roc_shmem_reset_stats();

    fflush(stdout);
}

void
Tester::barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
}

uint64_t
Tester::gpuCyclesToMicroseconds(uint64_t cycles)
{
    /**
     * The dGPU asm core timer runs at 27MHz. This is different from the
     * core clock returned by HIP. For an APU, this is different and might
     * need adjusting.
     */
    uint64_t gpu_frequency_MHz = 27;

    /**
     * hipDeviceGetAttribute(&gpu_frequency_khz,
     *                       hipDeviceAttributeClockRate,
     *                       0);
     */

    return cycles / gpu_frequency_MHz;
}

uint64_t
Tester::timerAvgInMicroseconds()
{
    uint64_t sum = 0;

    for (int i = 0; i < args.num_wgs; i++) {
       sum += gpuCyclesToMicroseconds(timer[i]);
    }

    return sum / args.num_wgs;
}
