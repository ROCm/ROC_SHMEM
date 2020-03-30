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
#include <sys/time.h>
#include "hip/hip_runtime.h"
#include "roc_shmem.hpp"
#include "util.h"

#include "config.h"

#include <mpi.h>

void
Barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

int64_t
gpuCyclesToMicroseconds(int64_t cycles)
{
    // dGPU asm core timer runs at 27MHz.  This is different from the core
    // clock returned by HIP.  For APU this is different and might need
    // adjusting.
    int gpu_frequency_khz = 27000;
//    hipDeviceGetAttribute(&gpu_frequency_khz,
//                          hipDeviceAttributeClockRate,
//                          0);

    return cycles / (gpu_frequency_khz / 1000);
}

uint64_t
calcAvg(uint64_t * timer, int num_wgs) {
    uint64_t sum = 0;
    for (int i = 0; i < num_wgs; i++) {
       sum += gpuCyclesToMicroseconds(timer[i]);
    }
    return sum / num_wgs;
}

void
show_usage(std::string name)
{
    std::cerr << "Usage: " << name << std::endl
             << "\t-t <Number of roc_shmem service threads>" << std::endl
             << "\t-w <Number of Work-groups>" << std::endl
             << "\t-a <Algorithm number to test>" << std::endl
             << "\t-s <Maximum message size (in Bytes)>" << std::endl;
}

uint64_t
grab_number(int argc, char* argv[], int arg_id)
{
    unsigned long long value;
    if (arg_id < argc) {
        value = atoll(argv[arg_id]);
    } else {
        std::cerr << "Value required after option" << std::endl;
        show_usage(argv[0]);
        exit(-1);
    }
    return value;
}

void
setup(int argc, char* argv[], int *num_wgs, int* num_threads,
      uint64_t *max_msg_size, int *numprocs, int *myid, int *algorithm)
{

    // defaults
    *num_threads = 1;
    *num_wgs = 1;
    *max_msg_size = (1 << 20);
    *algorithm = 0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            exit(-1);
        } else if (arg == "-t") {
            i++;
            *num_threads = grab_number(argc, argv, i);
        } else if (arg == "-w") {
            i++;
            *num_wgs = grab_number(argc, argv, i);
        } else if (arg == "-s") {
            i++;
            *max_msg_size = grab_number(argc, argv, i);
        } else if (arg == "-a") {
            i++;
            *algorithm = grab_number(argc, argv, i);
        } else {
            show_usage(argv[0]);
            exit(-1);
        }
    }

    assert(roc_shmem_pre_init() == ROC_SHMEM_SUCCESS);

    assert(roc_shmem_init(*num_wgs) == ROC_SHMEM_SUCCESS);

    *numprocs = roc_shmem_n_pes();
    *myid = roc_shmem_my_pe();

    if (*numprocs != 2) {
        if (*myid == 0) {
            std::cerr << "This test requires exactly two processes"
            << std::endl;
        }
        exit(-1);
    }

}
