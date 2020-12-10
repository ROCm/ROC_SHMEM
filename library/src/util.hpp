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

#ifndef UTIL_H
#define UTIL_H

#include <hip/hip_runtime.h>

#include "config.h"

#define CHECK_HIP(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

enum class BackendType {
    RO_BACKEND,
    GPU_IB_BACKEND
};

#define SFENCE()   asm volatile("sfence" ::: "memory")

#ifdef DEBUG
# define DPRINTF(x) if (ROC_SHMEM_DEBUG) printf x
#else
# define DPRINTF(x) do {} while (0)
#endif

#ifdef DEBUG
#define GPU_DPRINTF(...) gpu_dprintf(__VA_ARGS__);
#else
#define GPU_DPRINTF(...) do {} while (0)
#endif

// TODO: Cannot currently be extracted correctly from ROCm, so hardcoded
const int gpu_clock_freq_mhz = 27;

bool ROC_SHMEM_DEBUG = false;

const int WF_SIZE = 64;

/* Device-side internal functions */
__device__ void __roc_inv();
__device__ uint64_t __read_clock();

#define HW_ID_WV_ID_OFFSET 0
#define HW_ID_SD_ID_OFFSET 4
#define HW_ID_CU_ID_OFFSET 8
#define HW_ID_SE_ID_OFFSET 13

#define HW_ID_WV_ID_SIZE 4
#define HW_ID_SD_ID_SIZE 2
#define HW_ID_CU_ID_SIZE 4
#define HW_ID_SE_ID_SIZE 2

#define WVS_PER_SD 10
#define WVS_PER_CU 40
#define WVS_PER_SE 640

__device__ int get_hw_wv_index();
__device__ int get_hw_cu_index();

/*
 * Returns true if the caller's thread index is (0, 0, 0) in its block.
 */
__device__ bool is_thread_zero_in_block();

/*
 * Returns true if the caller's block index is (0, 0, 0) in its grid.  All
 * threads in the same block will return the same answer.
 */
__device__ bool is_block_zero_in_grid();

/*
 * Returns the number of threads in the caller's flattened thread block.
 */
__device__ int get_flat_block_size();

/*
 * Returns the flattened thread index of the calling thread within its
 * thread block.
 */
__device__ int get_flat_block_id();

/*
 * Returns the flattened block index that the calling thread is a member of in
 * in the grid. Callers from the same block will have the same index.
 */
__device__ int get_flat_grid_id();

__device__ uint32_t lowerID();
__device__ int wave_SZ();

extern __constant__ int *print_lock;

template <typename ...Args>
__device__ void
gpu_dprintf(const char *fmt, const Args &...args)
{
    for (int i = 0; i < WF_SIZE; i ++) {
        if ((get_flat_block_id() % WF_SIZE) == i) {
            /*
             * GPU-wide global lock that ensures that both prints are executed
             * by a single thread atomically.  We deliberately break control
             * flow so that only a single thread in a WF accesses the lock at a
             * time.  If multiple threads in the same WF attempt to gain the
             * lock at the same time, you have a classic GPU control flow
             * deadlock caused by threads in the same WF waiting on each other.
             */
            while (atomicCAS(print_lock, 0, 1) == 1);

            printf("WG (%d, %d, %d) TH (%d, %d, %d) ", hipBlockIdx_x,
                    hipBlockIdx_y, hipBlockIdx_z, hipThreadIdx_x,
                    hipThreadIdx_y, hipThreadIdx_z);
            printf(fmt, args...);

            *print_lock = 0;
        }
    }
}

__device__ void memcpy(void* dst, void* src, size_t size);

int rocm_init();

void rocm_memory_lock_to_fine_grain(void *ptr, size_t size, void **gpu_ptr,
                                    int gpu_id);

hsa_amd_hdp_flush_t * rocm_hdp(void);
void free_rocm_hdp(hsa_amd_hdp_flush_t* hpd);

#endif
