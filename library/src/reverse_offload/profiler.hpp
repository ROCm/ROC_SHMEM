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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_PROFILER_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_PROFILER_HPP_

#include <array>
#include <cassert>

#include "config.h"  // NOLINT(build/include_subdir)
#include "src/device_proxy.hpp"
#include "src/memory/hip_allocator.hpp"
#include "src/stats.hpp"

namespace rocshmem {

enum ro_net_stats {
  WAITING_ON_SLOT = 0,
  THREAD_FENCE_1,
  THREAD_FENCE_2,
  WAITING_ON_HOST,
  PACK_QUEUE,
  SHMEM_WAIT,
  RO_NUM_STATS
};

#ifdef PROFILE
typedef Stats<RO_NUM_STATS> ROStats;
#else
typedef NullStats<RO_NUM_STATS> ROStats;
#endif

template <typename ALLOCATOR>
class ProfilerProxy {
  static constexpr size_t MAX_NUM_BLOCKS{65536};

  using ProxyT = DeviceProxy<ALLOCATOR, ROStats, MAX_NUM_BLOCKS>;

 public:
  explicit ProfilerProxy(size_t num_blocks) : num_elem_{num_blocks} {
    assert(num_blocks <= MAX_NUM_BLOCKS);

    auto *stat{proxy_.get()};
    assert(stat);

    // TODO(bpotter) This may need to be aligned properly for placement new
    for (size_t i{0}; i < num_elem_; i++) {
      new (stat + i) ROStats();
    }
  }

  ~ProfilerProxy() {
    auto *stat{proxy_.get()};
    assert(stat);

    for (size_t i{0}; i < num_elem_; i++) {
      (stat + i)->~ROStats();
    }
  }

  __host__ __device__ ROStats *get(size_t i = 0) {
    assert(i < num_elem_);
    return proxy_.get() + i;
  }

 private:
  ProxyT proxy_{};

  size_t num_elem_{0};
};

using ProfilerProxyT = ProfilerProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_PROFILER_HPP_
