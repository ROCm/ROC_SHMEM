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

#ifndef LIBRARY_SRC_HDP_POLICY_HPP_
#define LIBRARY_SRC_HDP_POLICY_HPP_

#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include "config.h"  // NOLINT(build/include_subdir)
#include "src/util.hpp"

namespace rocshmem {

class HdpRocmPolicy {
 public:
  HdpRocmPolicy() { set_hdp_flush_ptr(); }

  __host__ void hdp_flush() { *hdp_flush_ptr_ = HDP_FLUSH_VAL; }

  __host__ unsigned int* get_hdp_flush_ptr() const { return hdp_flush_ptr_; }

  __device__ void hdp_flush() { STORE(hdp_flush_ptr_, HDP_FLUSH_VAL); }

  __device__ void flushCoherency() { hdp_flush(); }

  static const int HDP_FLUSH_VAL{0x01};

 private:
  void set_hdp_flush_ptr() {
    int hip_dev_id{};
    CHECK_HIP(hipGetDevice(&hip_dev_id));
    CHECK_HIP(hipDeviceGetAttribute(reinterpret_cast<int*>(&hdp_flush_ptr_),
                                    hipDeviceAttributeHdpMemFlushCntl,
                                    hip_dev_id));
  }

  unsigned int* hdp_flush_ptr_{nullptr};
};

class NoHdpPolicy {
 public:
  NoHdpPolicy() = default;

  __host__ void hdp_flush() {}

  __host__ unsigned int* get_hdp_flush_ptr() const { return nullptr; }

  __device__ void hdp_flush() {}

  __device__ void flushCoherency() { __roc_flush(); }
};

/*
 * Select which one of our HDP policies to use at compile time.
 */
#ifdef USE_COHERENT_HEAP
typedef NoHdpPolicy HdpPolicy;
#else
typedef HdpRocmPolicy HdpPolicy;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_HDP_POLICY_HPP_
