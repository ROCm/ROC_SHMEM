/******************************************************************************
 * Copyright (c) 2017-2018 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HDP_HELPER_HPP
#define HDP_HELPER_HPP

#include "config.h"

#include "util.hpp"

#include <sys/mman.h>

#include <sched.h>
#include <sys/syscall.h>
#include <sys/sysinfo.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#define FIJI_HDP_READ 0x2f4c
#define FIJI_HDP_FLUSH 0x5480

#define VEGA10_HDP_READ 0x3fc4
//#define VEGA10_HDP_READ 0x3fb8
//#define VEGA10_HDP_READ 0x4600

#define VEGA10_HDP_FLUSH 0x385c
//#define VEGA10_HDP_FLUSH 0x3fb8

#ifdef __HIP_ARCH_GFX900__
#define HDP_READ VEGA10_HDP_READ
#define HDP_FLUSH VEGA10_HDP_FLUSH
#define HDP_READ_INV_VAL 0x01
//#define HDP_READ_INV_VAL 0xFFFF
#elif __HIP_ARCH_GFX803__
#define HDP_READ FIJI_HDP_READ
#define HDP_FLUSH FIJI_HDP_FLUSH
#define HDP_READ_INV_VAL 0x001A1FE0
#else
#error "Unknown GPU. RTN requires Fiji or Vega GPUs"
#endif

#define HDP_FLUSH_VAL 0x01

#define HDP_READ_OFF (HDP_READ % getpagesize())
#define HDP_READ_PAOFF (HDP_READ - HDP_READ_OFF)

#define HDP_FLUSH_OFF (HDP_FLUSH % getpagesize())
#define HDP_FLUSH_PAOFF (HDP_FLUSH - HDP_FLUSH_OFF)

#define CHECK_HIP(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", \
              hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }\
}

typedef volatile struct  hdp_reg {
    unsigned int  *cpu_hdp_flush;
    unsigned int  *gpu_hdp_flush;
    uint32_t      rkey;
#ifdef _USE_HDP_MAP_
    unsigned int  *cpu_hdp_read_inv;
    unsigned int  *gpu_hdp_read_inv;
    int           fd;
#endif
} hdp_reg_t;

#ifndef _USE_HDP_MAP_
hdp_reg_t *rtn_hdp_flush_map(int rtn_id);
#endif

__device__ inline void
hdp_copy(hdp_reg_t* out, hdp_reg_t *in){
#ifdef _USE_HDP_MAP_
    out->gpu_hdp_read_inv   = in->gpu_hdp_read_inv;
#endif
    out->gpu_hdp_flush      = in->gpu_hdp_flush;
}

__device__ inline void
hdp_flush (hdp_reg_t *hdp_regs)
{
   *(hdp_regs->gpu_hdp_flush) = HDP_FLUSH_VAL;
}

__host__ inline void
hdp_flush (hdp_reg_t *hdp_regs)
{
    *(hdp_regs->cpu_hdp_flush) =  HDP_FLUSH_VAL;
}

#ifdef _USE_HDP_MAP_
__device__ inline void
hdp_read_inv (hdp_reg_t *hdp_regs)
{
 *(hdp_regs->gpu_hdp_read_inv) =  HDP_READ_INV_VAL;
}


__host__ inline void
hdp_read_inv (hdp_reg_t *hdp_regs)
{
   *(hdp_regs->cpu_hdp_read_inv) =  HDP_READ_INV_VAL;
}

inline void
hdp_map(hdp_reg_t * hdp_regs, int dev_id)
{
    std::string name = "/dev/hdp_umap_";
    name += std::to_string(dev_id);

    printf("IB name= %s \n",name.c_str());
    hdp_regs->fd = open(name.c_str(), O_RDWR);
    if (hdp_regs->fd == -1) {
        printf("could not open the HDP dev module \n");
    }

    hdp_regs->cpu_hdp_read_inv =
        (unsigned int*)mmap(NULL, getpagesize(), PROT_READ | PROT_WRITE,
                            MAP_SHARED, hdp_regs->fd, HDP_READ_PAOFF);
    hdp_regs->cpu_hdp_flush =
        (unsigned int*)mmap(NULL, getpagesize(), PROT_READ | PROT_WRITE,
                            MAP_SHARED, hdp_regs->fd, HDP_FLUSH_PAOFF);

    hdp_regs->cpu_hdp_read_inv =
        hdp_regs->cpu_hdp_read_inv + (HDP_READ_OFF /4);

    hdp_regs->cpu_hdp_flush = hdp_regs->cpu_hdp_flush    + (HDP_FLUSH_OFF /4);

}

inline hdp_reg_t *
hdp_map_all()
{
    int hdp_count = 0;
    hipGetDeviceCount(&hdp_count);

    hdp_reg_t *hdp_regs;
    hipCheck(hipHostMalloc((void**) &hdp_regs, sizeof(hdp_reg_t) * hdp_count),
             "Cannot allocate HDP host memory");

    // Set HIP mode that registers HDP registers to the VM space of all
    // devices, so that each device can flush/inv other devices HDP registers.
    // No need to use IPC for this, every processes data gets flushed.
    for (int i = 0; i < hdp_count; i++) {
        for (int j = 0; j < hdp_count; j++) {
            if (j != i)
                hipDeviceEnablePeerAccess(j, 0);
        }
        hdp_map(&hdp_regs[i], i);
    }
    return hdp_regs;
}

#endif
#endif
