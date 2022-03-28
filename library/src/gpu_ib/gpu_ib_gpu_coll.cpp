/******************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <roc_shmem.hpp>

#include "context_incl.hpp"
#include "gpu_ib_gpu_templates.hpp"
#include "util.hpp"

namespace rocshmem {

__device__ void
GPUIBContext::internal_direct_barrier(int pe,
                                      int n_pes,
                                      int64_t *pSync) {
    int64_t flag_val = 1;
    if (pe == 0) {
        for (size_t i = 1; i < n_pes; i++) {
            wait_until(&pSync[i], ROC_SHMEM_CMP_EQ, flag_val);
            pSync[i] = ROC_SHMEM_SYNC_VALUE;
        }
        for (size_t i = 1; i < n_pes; i++) {
            put_nbi(&pSync[0], &flag_val, 1, i);
        }

    } else {
        put_nbi(&pSync[pe], &flag_val, 1, 0);
        wait_until(&pSync[0], ROC_SHMEM_CMP_EQ, flag_val);
        pSync[0] = ROC_SHMEM_SYNC_VALUE;
    }
}

__device__ void
GPUIBContext::internal_atomic_barrier(int pe,
                                      int n_pes,
                                      int64_t *pSync) {
    int64_t flag_val = 1;
    if (pe == 0) {
        wait_until(&pSync[0], ROC_SHMEM_CMP_EQ, (int64_t)(n_pes - 1));
        pSync[0] = ROC_SHMEM_SYNC_VALUE;

        for (size_t i = 1; i < n_pes; i++) {
            put_nbi(&pSync[0], &flag_val, 1, i);
        }
    } else {
        amo_add(&pSync[0], flag_val, 0, 0);
        wait_until(&pSync[0], ROC_SHMEM_CMP_EQ, flag_val);
        pSync[0] = ROC_SHMEM_SYNC_VALUE;
    }
}

__device__ void
GPUIBContext::sync_all() {
    __syncthreads();
    if (is_thread_zero_in_block()) {
        int n_pes = num_pes;
        int pe = my_pe;

        if (n_pes < 64) {
            internal_direct_barrier(pe, n_pes, barrier_sync);
        } else {
            internal_atomic_barrier(pe, n_pes, barrier_sync);
        }
    }
    __threadfence();
    __syncthreads();
}

__device__ void
GPUIBContext::barrier_all() {
    sync_all();
    if (is_thread_zero_in_block()) {
        quiet();
    }
    __syncthreads();
}

}  // namespace rocshmem
