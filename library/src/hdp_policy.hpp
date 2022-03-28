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

#ifndef ROCSHMEM_LIBRARY_SRC_HDP_POLICY_HPP
#define ROCSHMEM_LIBRARY_SRC_HDP_POLICY_HPP

#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include "config.h"  // NOLINT(build/include_subdir)
#include "util.hpp"

namespace rocshmem {

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
    __device__ void
    flushCoherency(){
        hdp_flush();
    }
};

/*
 * No HDP management is needed
 * */
class NoHdpPolicy : public HdpBasePolicy{
   public:
    __device__
    void hdp_flush() {
    }

    __host__ void
    hdp_flush() {
    }

    __host__ unsigned int*
    get_hdp_flush_addr() const {
        return 0;
    }

    __device__ void
    flushCoherency(){
        __roc_flush();
    }

    __device__
    NoHdpPolicy() {
    }

    NoHdpPolicy();

};

/*
 * HDP management via the hdp_umap kernel module.
 */
class HdpMapPolicy : public HdpBasePolicy {
    const int FIJI_HDP_FLUSH = 0x5480;
    const int FIJI_HDP_READ = 0x2f4c;
    const int VEGA10_HDP_FLUSH = 0x385c;
    const int VEGA10_HDP_READ = 0x3fc4;

    const int HDP_FLUSH = VEGA10_HDP_FLUSH;
    const int HDP_READ = VEGA10_HDP_READ;

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
#ifdef USE_CACHED
typedef NoHdpPolicy HdpPolicy;
#else
#ifdef USE_HDP_MAP
typedef HdpMapPolicy HdpPolicy;
#else
typedef HdpRocmPolicy HdpPolicy;
#endif
#endif

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_HDP_POLICY_HPP
