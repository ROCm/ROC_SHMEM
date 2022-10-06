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

#include <functional>
#include <vector>
#include <iostream>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <roc_shmem.hpp>

#include "barrier_all_tester.hpp"
#include "broadcast_tester.hpp"
#include "empty_tester.hpp"
#include "team_broadcast_tester.hpp"
#include "primitive_tester.hpp"
#include "primitive_mr_tester.hpp"
#include "team_ctx_primitive_tester.hpp"
#include "team_ctx_infra_tester.hpp"
#include "primitive_amo_tester.hpp"
#include "ping_pong_tester.hpp"
#include "swarm_tester.hpp"
#include "reduction_tester.hpp"
#include "team_reduction_tester.hpp"
#include "random_access_tester.hpp"
#include "shmem_ptr_tester.hpp"
#include "extended_primitives.hpp"
#include "alltoall_tester.hpp"
#include "fcollect_tester.hpp"
#include "sync_tester.hpp"

Tester::Tester(TesterArguments args)
    : args(args)
{
    _type = (TestType) args.algorithm;
    _shmem_context = args.shmem_context;
    hipStreamCreate(&stream);
    hipEventCreate(&start_event);
    hipEventCreate(&stop_event);
    hipMalloc((void**)&timer, sizeof(uint64_t) * args.num_wgs);
}

Tester::~Tester()
{
    hipFree(timer);
    hipEventDestroy(stop_event);
    hipEventDestroy(start_event);
    hipStreamDestroy(stream);
}

std::vector<Tester*>
Tester::create(TesterArguments args)
{
    int rank = args.myid;
    std::vector<Tester*> testers;

    if (rank == 0)
        std::cout << "*** Creating Test: ";

    TestType type = (TestType) args.algorithm;

    switch (type) {
        case InitTestType:
            if (rank == 0)
                std::cout << "Init ***" << std::endl;
            testers.push_back(new EmptyTester(args));
            return testers;
        case GetTestType:
            if (rank == 0)
                std::cout << "Blocking Gets***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        case GetNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking Gets***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        case PutTestType:
            if (rank == 0)
                std::cout << "Blocking Puts***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        case PutNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking Puts***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        case TeamCtxInfraTestType:
            if (rank == 0)
                std::cout << "Team Ctx Infra test***" << std::endl;
            testers.push_back(new TeamCtxInfraTester(args));
            return testers;
        case TeamCtxGetTestType:
            if (rank == 0)
                std::cout << "Blocking Team Ctx Gets***" << std::endl;
            testers.push_back(new TeamCtxPrimitiveTester(args));
            return testers;
        case TeamCtxGetNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking Team Ctx Gets***" << std::endl;
            testers.push_back(new TeamCtxPrimitiveTester(args));
            return testers;
        case TeamCtxPutTestType:
            if (rank == 0)
                std::cout << "Blocking Team Ctx Puts***" << std::endl;
            testers.push_back(new TeamCtxPrimitiveTester(args));
            return testers;
        case TeamCtxPutNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking Team Ctx Puts***" << std::endl;
            testers.push_back(new TeamCtxPrimitiveTester(args));
            return testers;
        case PTestType:
            if (rank == 0)
                std::cout << "P Test***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        case GTestType:
            if (rank == 0)
                std::cout << "G Test***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
        case GetSwarmTestType:
            if (rank == 0)
                std::cout << "Get Swarm***" << std::endl;
            testers.push_back(new GetSwarmTester(args));
            return testers;
        case TeamReductionTestType:
            if (rank == 0)
                std::cout << "All-to-All Team-based Reduction***" << std::endl;
            testers.push_back(
                new TeamReductionTester<float, ROC_SHMEM_SUM>(
                    args,
                    [](float& f1, float& f2)
                    {
                        f1 = 1;
                        f2 = 1;
                    },
                    [](float v, float n_pes)
                    {
                        return (v == n_pes) ?
                            std::make_pair(true, "") :
                            std::make_pair(false,
                                "Got " + std::to_string(v) + ", Expect " + std::to_string(n_pes));
                    }
                )
            );
            return testers;
        case ReductionTestType:
            if (rank == 0)
                std::cout << "All-to-All Reduction***" << std::endl;

            testers.push_back(
                new ReductionTester<float, ROC_SHMEM_SUM>(
                    args,
                    [](float& f1, float& f2)
                    {
                        f1 = 1;
                        f2 = 1;
                    },
                    [](float v, float n_pes)
                    {
                        return (v == n_pes) ?
                            std::make_pair(true, "") :
                            std::make_pair(false,
                                "Got " + std::to_string(v) + ", Expect " + std::to_string(n_pes));
                    }
                )
            );

#if 0
            testers.push_back(
                new ReductionTester<double, ROC_SHMEM_SUM>(
                    args,
                    [](double& f1, double& f2)
                    {
                        f1=1;
                        f2=1;
                    },
                    [](double v, double n_pes)
                    {
                        return (v == n_pes) ?
                            std::make_pair(true, "") :
                            std::make_pair(false,
                                "Got "+ std::to_string(v) + ", Expect " + std::to_string(n_pes));
                    }
                )
            );

            testers.push_back( new ReductionTester<long double, ROC_SHMEM_SUM>(args,
                            [](long double& f1,long  double& f2){f1=1; f2=1;},
                            [](long double v){ return (v==2.0) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 2.0. [r3]."); }));
            testers.push_back( new ReductionTester<short, ROC_SHMEM_SUM>(args,
                            [](short& f1, short& f2){f1=1; f2=2;},
                            [](short v){ return (v==3) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 3. [r4]."); }));
            testers.push_back( new ReductionTester<int, ROC_SHMEM_SUM>(args,
                            [](int& f1, int& f2){f1=1; f2=2;},
                            [](int v){ return (v==3) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 3. [r5]."); }));
            testers.push_back( new ReductionTester<long long, ROC_SHMEM_SUM>(args,
                            [](long long& f1, long long& f2){f1=1; f2=2;},
                            [](long long v){ return (v==3) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 3. [r6]."); }));
            testers.push_back( new ReductionTester<long long, ROC_SHMEM_SUM>(args,
                            [](long long& f1, long long& f2){f1=1; f2=2;},
                            [](long long v){ return (v==3) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 3. [r7]."); }));
            // seems like deadlock or soemthing, this test hang forever
            testers.push_back( new ReductionTester<short, ROC_SHMEM_MIN>(args,
                            [](short& f1, short& f2){f1=1; f2=2;},
                            [](short v){ return (v==1) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 1. [r8]."); }));
            testers.push_back( new ReductionTester<int, ROC_SHMEM_MIN>(args,
                            [](int& f1, int& f2){f1=1; f2=2;},
                            [](int v){ return (v==1) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 1. [r9]."); }));
            testers.push_back( new ReductionTester<long long, ROC_SHMEM_MIN>(args,
                            [](long long& f1, long long& f2){f1=1; f2=2;},
                            [](long long v){ return (v==1) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 1. [r10]."); }));
            testers.push_back( new ReductionTester<long long, ROC_SHMEM_MIN>(args,
                            [](long long& f1, long long& f2){f1=1; f2=2;},
                            [](long long v){ return (v==1) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 1. [r11]."); }));
            testers.push_back( new ReductionTester<int, ROC_SHMEM_MAX>(args,
                            [](int& f1, int& f2){f1=1; f2=2;},
                            [](int v){ return (v==2) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 2. [r12]."); }));
            testers.push_back( new ReductionTester<long long, ROC_SHMEM_MAX>(args,
                            [](long long& f1, long long& f2){f1=1; f2=2;},
                            [](long long v){ return (v==2) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 2. [r13]."); }));
            testers.push_back( new ReductionTester<long long, ROC_SHMEM_MAX>(args,
                            [](long long& f1, long long& f2){f1=1; f2=2;},
                            [](long long v){ return (v==2) ? std::make_pair(true, "") :
                                   std::make_pair(false, "Got "+ std::to_string(v) +", Expect 2. [r14]."); }));
#endif
            return testers;
        case BroadcastTestType:
            if (rank == 0) {
                std::cout << "Broadcast Test***" << std::endl;
            }
            testers.push_back(
                new BroadcastTester<long>(
                    args,
                    [](long& f1, long& f2)
                    {
                        f1 = 1;
                        f2 = 2;
                    },
                    [rank](long v)
                    {
                        long expected_val;
                        /**
                         * The verification routine here requires that the
                         * PE_root value is 0 which denotes that the
                         * sending processing element is rank 0.
                         *
                         * The difference in expected values arises from
                         * the specification for broadcast where the
                         * PE_root processing element does not copy the
                         * contents from its own source to dest during
                         * the broadcast.
                         */
                        if (rank == 0) {
                            expected_val = 2;
                        } else {
                            expected_val = 1;
                        }

                        return (v == expected_val) ?
                            std::make_pair(true, ""):
                            std::make_pair(false,
                                           "Rank " + std::to_string(rank) +
                                           ", Got " + std::to_string(v) +
                                           ", Expect " +
                                           std::to_string(expected_val));
                    }
                )
            );
            return testers;
        case TeamBroadcastTestType:
            if (rank == 0) {
                std::cout << "Team Broadcast Test***" << std::endl;
            }
            testers.push_back(
                new TeamBroadcastTester<long>(
                    args,
                    [](long& f1, long& f2)
                    {
                        f1 = 1;
                        f2 = 2;
                    },
                    [rank](long v)
                    {
                        long expected_val;
                        /**
                         * The verification routine here requires that the
                         * PE_root value is 0 which denotes that the
                         * sending processing element is rank 0.
                         *
                         * The difference in expected values arises from
                         * the specification for broadcast where the
                         * PE_root processing element does not copy the
                         * contents from its own source to dest during
                         * the broadcast.
                         */
                        if (rank == 0) {
                            expected_val = 2;
                        } else {
                            expected_val = 1;
                        }

                        return (v == expected_val) ?
                            std::make_pair(true, ""):
                            std::make_pair(false,
                                           "Rank " + std::to_string(rank) +
                                           ", Got " + std::to_string(v) +
                                           ", Expect " +
                                           std::to_string(expected_val));
                    }
                )
            );
            return testers;
        case AllToAllTestType:
            if (rank == 0) {
                std::cout << "Alltoall Test***" << std::endl;
            }
            testers.push_back(
                new AlltoallTester<int64_t>(
                    args,
                    [rank](int64_t& f1, int64_t& f2, int64_t dest_pe)
                    {
                        const long SRC_SHIFT = 16;
                        // Make value for each src, dst pair unique
                        // by shifting src by SRC_SHIFT bits
                        f1 = (rank << SRC_SHIFT) + dest_pe;
                        f2 = -1;
                    },
                    [rank](int64_t v, int64_t src_pe)
                    {
                        const long SRC_SHIFT = 16;
                        // See if we obtained unique value
                        long expected_val = (src_pe << SRC_SHIFT) + rank;

                        return (v == expected_val) ?
                            std::make_pair(true, ""):
                            std::make_pair(false,
                                           "Rank " + std::to_string(rank) +
                                           ", Got " + std::to_string(v) +
                                           ", Expect " +
                                           std::to_string(expected_val));
                    }
                )
            );
            return testers;
        case FCollectTestType:
            if (rank == 0) {
                std::cout << "Fcollect Test***" << std::endl;
            }
            testers.push_back(
                new FcollectTester<int64_t>(
                    args,
                    [rank](int64_t& f1, int64_t& f2)
                    {
                        f1 = rank;
                        f2 = -1;
                    },
                    [rank](int64_t v, int64_t src_pe)
                    {
                        int64_t expected_val = src_pe;

                        return (v == expected_val) ?
                            std::make_pair(true, ""):
                            std::make_pair(false,
                                           "Rank " + std::to_string(rank) +
                                           ", Got " + std::to_string(v) +
                                           ", Expect " +
                                           std::to_string(expected_val));
                    }
                )
            );
            return testers;
        case AMO_FAddTestType:
            if (rank == 0)
                std::cout << "AMO Fetch_Add***" << std::endl;
            testers.push_back(new PrimitiveAMOTester(args));
            return testers;
        case AMO_FIncTestType:
            if (rank == 0)
                std::cout << "AMO Fetch_Inc***" << std::endl;
            testers.push_back(new PrimitiveAMOTester(args));
            return testers;
        case AMO_FetchTestType:
            if (rank == 0)
                std::cout << "AMO Fetch***" << std::endl;
            testers.push_back(new PrimitiveAMOTester(args));
            return testers;
        case AMO_FCswapTestType:
            if (rank == 0)
                std::cout << "AMO Fetch_CSWAP***" << std::endl;
            testers.push_back(new PrimitiveAMOTester(args));
            return testers;
        case AMO_AddTestType:
            if (rank == 0)
                std::cout << "AMO Add***" << std::endl;
            testers.push_back(new PrimitiveAMOTester(args));
            return testers;
        case AMO_IncTestType:
            if (rank == 0)
                std::cout << "AMO Inc***" << std::endl;
            testers.push_back(new PrimitiveAMOTester(args));
            return testers;
        case PingPongTestType:
            if (rank == 0)
                std::cout << "PingPong***" << std::endl;
            testers.push_back(new PingPongTester(args));
            return testers;
         case BarrierAllTestType:
            if (rank == 0)
                std::cout << "Barrier_All***" << std::endl;
            testers.push_back(new BarrierAllTester(args));
            return testers;
         case SyncAllTestType:
            if (rank == 0)
                std::cout << "SyncAll***" << std::endl;
            testers.push_back(new SyncTester(args));
            return testers;
         case SyncTestType:
            if (rank == 0)
                std::cout << "Sync***" << std::endl;
            testers.push_back(new SyncTester(args));
            return testers;
         case RandomAccessTestType:
            if (rank == 0)
                std::cout << "Random_Access***" << std::endl;
            testers.push_back(new RandomAccessTester(args));
            return testers;
        case ShmemPtrTestType:
            if (rank == 0)
                std::cout << "Shmem_Ptr***" << std::endl;
            testers.push_back(new ShmemPtrTester(args));
            return testers;
        case WGGetTestType:
            if (rank == 0)
                std::cout << "Blocking WG level Gets***" << std::endl;
            testers.push_back(new ExtendedPrimitiveTester(args));
            return testers;
        case WGGetNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking WG level Gets***" << std::endl;
            testers.push_back(new ExtendedPrimitiveTester(args));
            return testers;
        case WGPutTestType:
            if (rank == 0)
                std::cout << "Blocking WG level Puts***" << std::endl;
            testers.push_back(new ExtendedPrimitiveTester(args));
            return testers;
        case WGPutNBITestType:
            if (rank == 0)
                std::cout << "Non-Blocking WG level Puts***" << std::endl;
            testers.push_back(new ExtendedPrimitiveTester(args));
            return testers;
        case PutNBIMRTestType:
            if (rank == 0)
                std::cout << "Non-Blocking Put message rate***" << std::endl;
            testers.push_back(new PrimitiveMRTester(args));
            return testers;
        default:
            if (rank == 0)
                std::cout << "Unknown***" << std::endl;
            testers.push_back(new PrimitiveTester(args));
            return testers;
    }
    return testers;
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

        resetBuffers(size);

        /**
         * Restricts the number of iterations of really large messages.
         */
        if (size > args.large_message_size)
            num_loops = args.loop_large;

        barrier();

        preLaunchKernel();

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

//            roc_shmem_dump_stats();
            roc_shmem_reset_stats();
        }

        barrier();

        postLaunchKernel();

        // data validation
        verifyResults(size);

        barrier();

        if (_type != TeamCtxInfraTestType) {
            print(size);
        }
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
                  (_type == TeamReductionTestType) ||
                  (_type == BroadcastTestType) ||
                  (_type == TeamBroadcastTestType) ||
                  (_type == AllToAllTestType)  ||
                  (_type == FCollectTestType)  ||
                  (_type == PingPongTestType)  ||
                  (_type == BarrierAllTestType)   ||
                  (_type == SyncTestType)   ||
                  (_type == SyncAllTestType)   ||
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
    double avg_msg_rate = num_timed_msgs / (timer_avg / 1e6);

    float total_kern_time_ms;
    hipEventElapsedTime(&total_kern_time_ms, start_event, stop_event);
    float total_kern_time_s = total_kern_time_ms / 1000;
    double bandwidth_avg_gbs = num_msgs * size * bw_factor / total_kern_time_s / pow(2, 30);

    int field_width = 20;
    int float_precision = 2;

    printf("\n##### Message Size %lu #####\n", size);

    printf("%*s%*s%*s\n",
           field_width + 1, "Latency AVG (us)",
           field_width + 1, "Bandwidth (GB/s)",
           field_width + 1, "Avg Message rate (Messages/s)");

    printf("%*.*f %*.*f %*.*f\n",
           field_width, float_precision, latency_avg,
           field_width, float_precision, bandwidth_avg_gbs,
           field_width, float_precision, avg_msg_rate);

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
