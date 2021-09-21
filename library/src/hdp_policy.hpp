/******************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LIBRARY_SRC_HDP_POLICY_HPP_
#define LIBRARY_SRC_HDP_POLICY_HPP_

#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include "config.h"  // NOLINT(build/include_subdir)
#include "util.hpp"

/*
 * Base class for HDP policies.
 */
class HdpBasePolicy {
 protected:
    unsigned int *cpu_hdp_flush = nullptr;
    unsigned int *gpu_hdp_flush = nullptr;

 public:
    static const int HDP_FLUSH_VAL = 0x01;

    __device__
    void hdp_flush() {
        STORE(gpu_hdp_flush, HDP_FLUSH_VAL);
    }

    __host__ void
    hdp_flush() {
        *cpu_hdp_flush =  HDP_FLUSH_VAL;
    }

    __host__ unsigned int*
    get_hdp_flush_addr() const {
        return cpu_hdp_flush;
    }
};

/*
 * HDP management via the hdp_umap kernel module.
 */
class HdpMapPolicy : public HdpBasePolicy {
    const int FIJI_HDP_FLUSH = 0x5480;
    const int FIJI_HDP_READ = 0x2f4c;
    const int VEGA10_HDP_FLUSH = 0x385c;
    const int VEGA10_HDP_READ = 0x3fc4;

// #ifdef __HIP_ARCH_GFX900__
    const int HDP_FLUSH = VEGA10_HDP_FLUSH;
    const int HDP_READ = VEGA10_HDP_READ;
// #elif __HIP_ARCH_GFX803__
//    const int HDP_FLUSH = FIJI_HDP_FLUSH;
//    const int HDP_READ = FIJI_HDP_READ;
// #else
//    #error "Unknown GPU. RTN requires Fiji or Vega GPUs"
// #endif

    int fd = -1;
    int hdp_flush_off = 0;
    int hdp_flush_pa_off = 0;
    int hdp_read_off = 0;
    int hdp_read_pa_off = 0;

 public:
    __device__
    HdpMapPolicy() {
    }

    HdpMapPolicy();
};

/*
 * HDP management via ROCm HDP APIs.
 */
class HdpRocmPolicy : public HdpBasePolicy {
 public:
    __device__
    HdpRocmPolicy() {
    }

    HdpRocmPolicy();

    hsa_amd_hdp_flush_t *hdp = nullptr;
};

/*
 * Select which one of our HDP policies to use at compile time.
 */
#ifdef USE_HDP_MAP
typedef HdpMapPolicy HdpPolicy;
#else
typedef HdpRocmPolicy HdpPolicy;
#endif

#endif  // LIBRARY_SRC_HDP_POLICY_HPP_
