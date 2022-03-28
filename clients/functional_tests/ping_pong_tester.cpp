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

#include "ping_pong_tester.hpp"

#include <roc_shmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void
PingPongTest(int loop,
             int skip,
             uint64_t *timer,
             int *r_buf,
             ShmemContextType ctx_type)
{
    __shared__ roc_shmem_ctx_t ctx;

    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);

    int pe = roc_shmem_ctx_my_pe(ctx);

    if (hipThreadIdx_x == 0) {
        uint64_t start;

        for (int i = 0; i < loop + skip; i++) {
            if (i == skip) {
                start = roc_shmem_timer();
            }

            if (pe == 0) {
                roc_shmem_ctx_int_p(ctx, r_buf, i + 1, 1);
                roc_shmem_int_wait_until(r_buf, ROC_SHMEM_CMP_EQ, i + 1);
            } else {
                roc_shmem_int_wait_until(r_buf, ROC_SHMEM_CMP_EQ, i + 1);
                roc_shmem_ctx_int_p(ctx, r_buf, i + 1, 0);
            }
        }
        timer[hipBlockIdx_x] =  roc_shmem_timer() - start;
    }
    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PingPongTester::PingPongTester(TesterArguments args)
    : Tester(args)
{
    r_buf = (int*)roc_shmem_malloc(sizeof(int));
}

PingPongTester::~PingPongTester()
{
    roc_shmem_free(r_buf);
}

void
PingPongTester::resetBuffers()
{
    memset(r_buf, 0, sizeof(int));
}

void
PingPongTester::launchKernel(dim3 gridSize,
                             dim3 blockSize,
                             int loop,
                             uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(PingPongTest,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       r_buf,
                       _shmem_context);

    num_msgs = (loop + args.skip) * gridSize.x;
    num_timed_msgs = loop * gridSize.x;
}

void
PingPongTester::verifyResults(uint64_t size)
{
}
