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

#include "collective_tester.hpp"

#include <roc_shmem.hpp>

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void
CollectiveTest(int loop,
               int skip,
               uint64_t *timer,
               float *s_buf,
               float *r_buf,
               float *pWrk,
               long *pSync,
               int size,
               TestType type)
{
    __shared__ roc_shmem_ctx_t ctx;

    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(SHMEM_CTX_WG_PRIVATE, &ctx);

    uint64_t start;
    if (hipThreadIdx_x == 0) {
        start = roc_shmem_timer();

    }

    __syncthreads();

    for (int i = 0; i < loop; i++) {
        switch(type) {
            case ReductionTestType:
                roc_shmem_wg_to_all<float, ROC_SHMEM_SUM>(
                        ctx, r_buf, s_buf, size, 0, 0, 2, pWrk, pSync);
                break;

            case BarrierTestType:
                roc_shmem_wg_barrier_all(ctx);
                break;

            default:
                break;
        }
    }

    __syncthreads();

    if (hipThreadIdx_x == 0) {
        timer[hipBlockIdx_x] = roc_shmem_timer() - start;
    }

    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
CollectiveTester::CollectiveTester(TesterArguments args)
    : Tester(args)
{
    s_buf = (float *)roc_shmem_malloc(args.max_msg_size * sizeof(float));
    r_buf = (float *)roc_shmem_malloc(args.max_msg_size * sizeof(float));

    size_t p_wrk_size =
        std::max(args.max_msg_size / 2 + 1, SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    pWrk = (float *)roc_shmem_malloc(p_wrk_size * sizeof(float));

    size_t p_sync_size = SHMEM_REDUCE_SYNC_SIZE;
    pSync = (long *)roc_shmem_malloc(p_sync_size * sizeof(long));

    for (int i = 0; i < p_sync_size; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }
}

CollectiveTester::~CollectiveTester()
{
    roc_shmem_free(s_buf);
    roc_shmem_free(r_buf);
    roc_shmem_free(pWrk);
    roc_shmem_free(pSync);
}

void
CollectiveTester::launchKernel(dim3 gridSize,
                               dim3 blockSize,
                               int loop,
                               uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(CollectiveTest,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       s_buf,
                       r_buf,
                       pWrk,
                       pSync,
                       size,
                       _type);
}

void
CollectiveTester::resetBuffers()
{
    for (int i = 0; i < args.max_msg_size; i++) {
        s_buf[i] = 1.0;
        r_buf[i] = 1.0;
    }
}

void
CollectiveTester::verifyResults(uint64_t size)
{
    if(_type == ReductionTestType) {
        for (int i = 0; i < size; i++) {
            if (r_buf[i] != 2.0) {
                fprintf(stderr, "Data validation error at idx %d\n", i);
                fprintf(stderr, "Got %f, Expected %f\n", r_buf[i], 2.0);
                exit(-1);
            }
        }
    }
}
