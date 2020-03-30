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

#include "config.h"

#include "util.hpp"

/* Device-side internal functions */
__device__ void __roc_inv() { asm volatile ("buffer_wbinvl1_vol;"); }
__device__ uint64_t __read_clock() {
    uint64_t clock;
    asm volatile ("s_memrealtime %0\n\t"
                  "s_waitcnt lgkmcnt(0)\n\t"
                    : "=s" (clock));
    return clock;
}

__device__ int
get_hw_wv_index() {
    unsigned wv_id, sd_id, cu_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 0, 4)" : "=s"(wv_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 4, 2)" : "=s"(sd_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 2)" : "=s"(se_id));
/*
    // Note that we can't use the SIZES above because some of them are over
    // provisioned (i.e. 4 bits for wave but we have only 10) and we have an
    // exact number of queues.
    return (se_id << (HW_ID_CU_ID_SIZE + HW_ID_SD_ID_SIZE + HW_ID_WV_ID_SIZE))
           + (cu_id << (HW_ID_SD_ID_SIZE + HW_ID_WV_ID_SIZE))
           + (sd_id << (HW_ID_WV_ID_SIZE)) + wv_id;
*/
    return wv_id + sd_id * 10 + cu_id * 40 + se_id * 640;
}

__device__ int
get_hw_cu_index() {
    unsigned cu_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 2)" : "=s"(se_id));
    return cu_id + se_id * 16;
}

__device__ bool
is_thread_zero_in_block()
{
    return hipThreadIdx_x == 0 && hipThreadIdx_y == 0 && hipThreadIdx_z == 0;
}

__device__ bool
is_block_zero_in_grid()
{
    return hipBlockIdx_x == 0 && hipBlockIdx_y == 0 && hipBlockIdx_z == 0;
}

__device__ int
get_flat_block_size()
{
    return hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
}

__device__ int
get_flat_block_id()
{
    return hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_z *
        hipBlockDim_x * hipBlockDim_y;
}

__device__ int
get_flat_grid_id()
{
    return hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x + hipBlockIdx_z *
        hipGridDim_x * hipGridDim_y;
}
