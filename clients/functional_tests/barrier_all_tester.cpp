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

#include "barrier_all_tester.hpp"

#include <roc_shmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void BarrierAllTest(int loop, int skip, uint64_t *timer) {
  __shared__ roc_shmem_ctx_t ctx;

  roc_shmem_wg_init();
  roc_shmem_wg_ctx_create(ROC_SHMEM_CTX_WG_PRIVATE, &ctx);

  uint64_t start;
  for (int i = 0; i < loop + skip; i++) {
    if (hipThreadIdx_x == 0 && i == skip) {
      start = roc_shmem_timer();
    }

    __syncthreads();

    roc_shmem_ctx_wg_barrier_all(ctx);
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
BarrierAllTester::BarrierAllTester(TesterArguments args) : Tester(args) {}

BarrierAllTester::~BarrierAllTester() {}

void BarrierAllTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                    uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(BarrierAllTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, timer);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void BarrierAllTester::resetBuffers(uint64_t size) {}

void BarrierAllTester::verifyResults(uint64_t size) {}
