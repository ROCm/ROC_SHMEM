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

#ifndef ROCSHMEM_LIBRARY_SRC_UTIL_HPP
#define ROCSHMEM_LIBRARY_SRC_UTIL_HPP

#include <hip/hip_runtime.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "config.h"
#include "constants.hpp"

namespace rocshmem {

__device__  inline int
uncached_load_ubyte(uint8_t* src) {
    int ret;
    __asm__  volatile("global_load_ubyte %0 %1 off glc slc \n"
                       "s_waitcnt vmcnt(0)"
                : "=v" (ret)
                : "v" (src));
    return ret;
}

template <typename T>
__device__ inline T
uncached_load(T* src) {
    T ret;
    switch(sizeof(T)) {
      case 4:
        __asm__  volatile("global_load_dword %0 %1 off glc slc \n"
                          "s_waitcnt vmcnt(0)"
                    : "=v" (ret)
                    : "v" (src));
        break;
      case 8:
        __asm__  volatile("global_load_dwordx2 %0 %1 off glc slc \n"
                          "s_waitcnt vmcnt(0)"
                    : "=v" (ret)
                    : "v" (src));
            break;
      default:
        break;
    }
    return ret;
}

#define LOAD(VAR) __atomic_load_n((VAR), __ATOMIC_SEQ_CST)
#define STORE(DST, SRC) __atomic_store_n((DST), (SRC), __ATOMIC_SEQ_CST)

#define CHECK_HIP(cmd) {\
    hipError_t error = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

#define SFENCE() asm volatile("sfence" ::: "memory")

#ifdef DEBUG
# define DPRINTF(...) do {\
    printf(__VA_ARGS__);\
} while (0);
#else
# define DPRINTF(...) do {\
} while (0);
#endif

#ifdef DEBUG
#define GPU_DPRINTF(...) do {\
        gpu_dprintf(__VA_ARGS__);\
} while (0);
#else
#define GPU_DPRINTF(...) do {\
} while (0);
#endif

const extern int gpu_clock_freq_mhz;

/* Device-side internal functions */
__device__ void __roc_inv();
__device__ void __roc_flush();
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

__device__ int
get_hw_wv_index();

__device__ int
get_hw_cu_index();

/*
 * Returns true if the caller's thread index is (0, 0, 0) in its block.
 */
__device__ bool
is_thread_zero_in_block();

/*
 * Returns true if the caller's block index is (0, 0, 0) in its grid.  All
 * threads in the same block will return the same answer.
 */
__device__ bool
is_block_zero_in_grid();

/*
 * Returns the number of threads in the caller's flattened thread block.
 */
__device__ int
get_flat_block_size();

/*
 * Returns the flattened thread index of the calling thread within its
 * thread block.
 */
__device__ int
get_flat_block_id();

/*
 * Returns the flattened block index that the calling thread is a member of in
 * in the grid. Callers from the same block will have the same index.
 */
__device__ int
get_flat_grid_id();

/*
 * Returns true if the caller's thread flad_id is 0 in its wave.
 */
__device__ bool
is_thread_zero_in_wave();


__device__ uint32_t
lowerID();

__device__ int
wave_SZ();

extern __constant__ int *print_lock;

template <typename ...Args>
__device__ void
gpu_dprintf(const char* fmt,
            const Args &...args) {
    for (int i {0}; i < WF_SIZE; i ++) {
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

            printf("WG (%lu, %lu, %lu) TH (%lu, %lu, %lu) ",
                    hipBlockIdx_x,
                    hipBlockIdx_y,
                    hipBlockIdx_z,
                    hipThreadIdx_x,
                    hipThreadIdx_y,
                    hipThreadIdx_z);
            printf(fmt, args...);

            *print_lock = 0;
        }
    }
}

__device__ void
memcpy(void* dst, void* src, size_t size);

__device__ void
memcpy_wg(void* dst, void* src, size_t size);

__device__ void
memcpy_wave(void* dst, void* src, size_t size);

int
rocm_init();

void
rocm_memory_lock_to_fine_grain(void* ptr,
                               size_t size,
                               void** gpu_ptr,
                               int gpu_id);

// Returns clock frequency used by s_memrealtime() in Mhz
int
wallClk_freq_mhz();

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_UTIL_HPP
