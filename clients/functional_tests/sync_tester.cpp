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

#include "sync_tester.hpp"

#include <roc_shmem.hpp>

using namespace rocshmem;
roc_shmem_team_t team_sync_world_dup;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void SyncTest(int loop, int skip, uint64_t *timer, TestType type,
                         ShmemContextType ctx_type, roc_shmem_team_t team) {
  __shared__ roc_shmem_ctx_t ctx;
  roc_shmem_wg_init();
  roc_shmem_wg_ctx_create(ctx_type, &ctx);

  uint64_t start;
  for (int i = 0; i < loop + skip; i++) {
    if (hipThreadIdx_x == 0 && i == skip) {
      start = roc_shmem_timer();
    }

    __syncthreads();
    switch (type) {
      case SyncAllTestType:
        roc_shmem_ctx_wg_sync_all(ctx);
        break;
      case SyncTestType:
        roc_shmem_ctx_wg_team_sync(ctx, team);
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
SyncTester::SyncTester(TesterArguments args) : Tester(args) {}

SyncTester::~SyncTester() {}

void SyncTester::resetBuffers(uint64_t size) {}

void SyncTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                              uint64_t size) {
  size_t shared_bytes = 0;

  int n_pes = roc_shmem_team_n_pes(ROC_SHMEM_TEAM_WORLD);

  team_sync_world_dup = ROC_SHMEM_TEAM_INVALID;
  roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                               &team_sync_world_dup);

  hipLaunchKernelGGL(SyncTest, gridSize, blockSize, shared_bytes, stream, loop,
                     args.skip, timer, _type, _shmem_context,
                     team_sync_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop;
}

void SyncTester::verifyResults(uint64_t size) {}
