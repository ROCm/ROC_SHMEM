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

#include "config.h"
#include "util.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace rocshmem {

const int gpu_clock_freq_mhz {wallClk_freq_mhz()};

__constant__ int *print_lock;

typedef struct device_agent {
    hsa_agent_t agent;
    hsa_amd_memory_pool_t pool;
} device_agent_t;

std::vector<device_agent_t> gpu_agents;
std::vector<device_agent_t> cpu_agents;

__device__ void inline
store_asm(uint8_t* val,
          uint8_t* dst,
          int size) {
    switch(size) {
      case 2: {
        int16_t val16 {*(reinterpret_cast<int16_t*>(val))};
        asm  volatile("flat_store_short %0 %1 glc slc"
                          : : "v"(dst), "v"(val16));
        break;
      }
      case 4: {
        int32_t val32 {*(reinterpret_cast<int32_t*>(val))};
        asm  volatile("flat_store_dword %0 %1 glc slc"
                          : : "v"(dst), "v"(val32));
        break;
      }
      case 8: {
        int64_t val64 {*(reinterpret_cast<int64_t*>(val))};
        asm  volatile("flat_store_dwordx2 %0 %1 glc slc"
                          : : "v"(dst), "v"(val64));
        break;
      }
      default:
        break;
    }
}

__device__ void
memcpy(void* dst,
       void* src,
       size_t size) {
    uint8_t *dst_bytes {static_cast<uint8_t*>(dst)};
    uint8_t *src_bytes {static_cast<uint8_t*>(src)};

    for (int i = 8; i > 1; i >>= 1) {
        while (size >= i) {
            store_asm(src_bytes,
                      dst_bytes,
                      i);
            src_bytes += i;
            dst_bytes += i;
            size -= i;
        }
    }

    if (size == 1) {
        *dst_bytes = *src_bytes;
    }
}

__device__ void
memcpy_wg(void* dst,
          void* src,
          size_t size) {
    int thread_id {get_flat_block_id()};
    int block_size {get_flat_block_size()};

    int cpy_size {};
    uint8_t* dst_bytes {nullptr};
    uint8_t* dst_def {nullptr};
    uint8_t* src_bytes {nullptr};
    uint8_t* src_def {nullptr};

    dst_def = reinterpret_cast<uint8_t*>(dst);
    src_def = reinterpret_cast<uint8_t*>(src);
    dst_bytes = dst_def;
    src_bytes = src_def;

    for (int j {8}; j > 1; j >>= 1) {
        cpy_size = size / j;
        for(int i {thread_id}; i < cpy_size; i += block_size) {
            dst_bytes = dst_def;
            src_bytes = src_def;

            src_bytes += i * j;
            dst_bytes += i * j;

            store_asm(src_bytes,
                      dst_bytes,
                      j);
        }
        size -= cpy_size * j;
        dst_def += cpy_size * j;
        src_def += cpy_size * j;
    }

    if (size == 1) {
        if (is_thread_zero_in_block()) {
            *dst_bytes = *src_bytes;
        }
    }
}

__device__ void
memcpy_wave(void* dst,
            void* src,
            size_t size) {
    uint8_t* dst_bytes {static_cast<uint8_t*>(dst)};
    uint8_t* src_bytes {static_cast<uint8_t*>(src)};

    int cpy_size {};
    int thread_id {get_flat_block_id()};
    for(int j {8}; j > 1; j >>= 1) {
        cpy_size = size / j;
        for (int i {thread_id}; i < cpy_size; i += WF_SIZE) {
            store_asm(src_bytes,
                      dst_bytes,
                      j);
            src_bytes += i * j;
            dst_bytes += i * j;
            size -= cpy_size * j;
        }
    }

    if (size == 1) {
        if (is_thread_zero_in_wave()) {
            *dst_bytes = *src_bytes;
        }
    }
}

__device__ uint32_t
lowerID () {
    return __ffsll(__ballot(1)) - 1;
}

__device__ int
wave_SZ() {
    return __popcll(__ballot(1));
}

/* Device-side internal functions */
__device__ void
__roc_inv() {
    asm volatile ("buffer_wbinvl1;");
}

__device__ void
__roc_flush() {
#ifdef USE_CACHED
#if __gfx90a__
    asm volatile("s_dcache_wb;");
    asm volatile("buffer_wbl2;");
#endif
#endif
}

__device__ uint64_t
__read_clock() {
    uint64_t clock {};
    asm volatile("s_memrealtime %0\n"
                 "s_waitcnt lgkmcnt(0)\n"
                     : "=s" (clock));
    return clock;
}

__device__ int
get_hw_wv_index() {
    unsigned wv_id {};
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 0, 4)"
                     : "=s"(wv_id));
    unsigned sd_id {};
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 4, 2)"
                     : "=s"(sd_id));
    unsigned cu_id {};
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)"
                     : "=s"(cu_id));
    unsigned se_id {};
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 2)"
                     : "=s"(se_id));
    /*
     * Note that we can't use the SIZES above because some of them are over
     * provisioned (i.e. 4 bits for wave but we have only 10) and we have an
     * exact number of queues.
     */
    /*
    return (se_id << (HW_ID_CU_ID_SIZE + HW_ID_SD_ID_SIZE + HW_ID_WV_ID_SIZE))
           + (cu_id << (HW_ID_SD_ID_SIZE + HW_ID_WV_ID_SIZE))
           + (sd_id << (HW_ID_WV_ID_SIZE)) + wv_id;
    */
    return wv_id +
           sd_id * 10 +
           cu_id * 40 +
           se_id * 640;
}

__device__ int
get_hw_cu_index() {
    unsigned cu_id {};
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)"
                      : "=s"(cu_id));
    unsigned se_id {};
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 2)"
                      : "=s"(se_id));
    return cu_id +
           se_id * 16;
}

__device__ bool
is_thread_zero_in_block() {
    return hipThreadIdx_x == 0 &&
           hipThreadIdx_y == 0 &&
           hipThreadIdx_z == 0;
}

__device__ bool
is_block_zero_in_grid() {
    return hipBlockIdx_x == 0 &&
           hipBlockIdx_y == 0 &&
           hipBlockIdx_z == 0;
}

__device__ int
get_flat_block_size() {
    return hipBlockDim_x *
           hipBlockDim_y *
           hipBlockDim_z;
}

__device__ int
get_flat_block_id() {
    return hipThreadIdx_x +
           hipThreadIdx_y * hipBlockDim_x +
           hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
}

__device__ int
get_flat_grid_id() {
    return hipBlockIdx_x +
           hipBlockIdx_y * hipGridDim_x +
           hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
}

__device__ bool
is_thread_zero_in_wave() {
    return get_flat_block_id() % WF_SIZE;
}

hsa_status_t
rocm_hsa_amd_memory_pool_callback(hsa_amd_memory_pool_t memory_pool,
                                  void* data) {
    hsa_amd_memory_pool_global_flag_t pool_flag {};

    hsa_status_t status {
        hsa_amd_memory_pool_get_info(memory_pool,
                                     HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                     &pool_flag)};

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failure to get pool info: 0x%x", status);
        return status;
    }

    if (pool_flag == (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
        *static_cast<hsa_amd_memory_pool_t*>(data) = memory_pool;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
rocm_hsa_agent_callback(hsa_agent_t agent,
                        void* data) {
    hsa_device_type_t device_type {};

    hsa_status_t status {hsa_agent_get_info(agent,
                                            HSA_AGENT_INFO_DEVICE,
                                            &device_type)};

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failure to get device type: 0x%x", status);
        return status;
    }

    if (device_type == HSA_DEVICE_TYPE_GPU) {
        gpu_agents.emplace_back();
        gpu_agents.back().agent = agent;
        status = hsa_amd_agent_iterate_memory_pools(
                                        agent,
                                        rocm_hsa_amd_memory_pool_callback,
                                        &(gpu_agents.back().pool));
    }

    if (device_type == HSA_DEVICE_TYPE_CPU) {
        cpu_agents.emplace_back();
        cpu_agents.back().agent = agent;
        status = hsa_amd_agent_iterate_memory_pools(
                                        agent,
                                        rocm_hsa_amd_memory_pool_callback,
                                        &(cpu_agents.back().pool));
    }

    return status;
}

int
rocm_init() {
    hsa_status_t status {hsa_init()};

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failure to open HSA connection: 0x%x", status);
        return 1;
    }

    status = hsa_iterate_agents(rocm_hsa_agent_callback,
                                nullptr);

    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        printf("Failure to iterate HSA agents: 0x%x", status);
        return 1;
    }

    return 0;
}

void
rocm_memory_lock_to_fine_grain(void* ptr,
                               size_t size,
                               void** gpu_ptr,
                               int gpu_id) {
    hsa_status_t status {
        hsa_amd_memory_lock_to_pool(ptr,
                                    size,
                                    &(gpu_agents[gpu_id].agent),
                                    1,
                                    cpu_agents[0].pool,
                                    0,
                                    gpu_ptr)
    };

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failed to lock memory pool (%p): 0x%x\n", ptr, status);
        exit(-1);
    }
}

// TODO(kpunniya): use runtime value instead of hard-coded value
int
wallClk_freq_mhz() {
    hipDeviceProp_t deviceProp {};
    CHECK_HIP(hipGetDeviceProperties(&deviceProp, 0));
    switch(deviceProp.gcnArch) {
        case 900:  // MI25
            return 27;
        case 906:
            return 25; // MI50,MI60
        case 908:
            return 25; // MI100
        case 910:
            return 25; // MI200
        default:
            assert(false && "clock data unavailable");
            return 0;
    }
}

}  // namespace rocshmem
