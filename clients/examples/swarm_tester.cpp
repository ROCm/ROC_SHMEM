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

#include "swarm_tester.hpp"

#include <roc_shmem.hpp>

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void
GetSwarmTest(int loop,
             int skip,
             uint64_t *timer,
             char *s_buf,
             char *r_buf,
             int size)
{
    __shared__ roc_shmem_ctx_t ctx;

    int provided;
    roc_shmem_wg_init_thread(SHMEM_THREAD_MULTIPLE, &provided);
    assert(provided == SHMEM_THREAD_MULTIPLE);

    roc_shmem_wg_ctx_create(SHMEM_CTX_WG_PRIVATE, &ctx);

    __syncthreads();

    int index = hipThreadIdx_x * size;
    uint64_t start = 0;

    for (int i = 0; i < loop + skip; i++) {

        if (i == skip)
            start = roc_shmem_timer();

        roc_shmem_getmem(ctx, &r_buf[index], &s_buf[index], size, 1);

        __syncthreads();

    }

    atomicAdd((unsigned long long *) &timer[hipBlockIdx_x],
                roc_shmem_timer() - start);

    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
GetSwarmTester::GetSwarmTester(TesterArguments args)
        : PrimitiveTester(args)
{
}

GetSwarmTester::~GetSwarmTester()
{
}


void
GetSwarmTester::launchKernel(dim3 gridSize,
                             dim3 blockSize,
                             int loop,
                             uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(GetSwarmTest,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       s_buf,
                       r_buf,
                       size);

    num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
    num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void
GetSwarmTester::verifyResults(uint64_t size)
{
    if (args.myid == 0) {
        for (int i = 0; i < size * args.wg_size; i++) {
            if (r_buf[i] != '0') {
                fprintf(stderr, "Data validation error at idx %d\n", i);
                fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
                exit(-1);
            }
        }
    }
}
