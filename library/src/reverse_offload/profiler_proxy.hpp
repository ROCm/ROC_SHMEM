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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_PROFILER_PROXY_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_PROFILER_PROXY_HPP

#include <array>
#include <cassert>

#include "device_proxy.hpp"
#include "ro_net_internal.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class ProfilerProxy {
    static constexpr size_t MAX_NUM_BLOCKS {65536};

    using ProxyT = DeviceProxy<ALLOCATOR, ROStats, MAX_NUM_BLOCKS>;

  public:
    /**
     * Placement new the memory which is allocated by proxy_
     */
    ProfilerProxy(size_t num_blocks)
        : num_elem_ {num_blocks} {
        assert(num_blocks <= MAX_NUM_BLOCKS);

        auto *stat {proxy_.get()};
        assert(stat);

        // TODO: @Brandon This may need to be aligned properly for placement new
        for (size_t i {0}; i < num_elem_; i++) {
            new (stat + i) ROStats();
        }
    }

    /**
     * Since placement new is called in the constructor, then
     * delete must be called manually.
     */
    ~ProfilerProxy() {
        auto *stat {proxy_.get()};
        assert(stat);

        for (size_t i {0}; i < num_elem_; i++) {
            (stat + i)->~ROStats();
        }
    }

    /**
     * @brief Provide access to the memory referenced by the proxy
     */
    __host__ __device__
    ROStats*
    get(size_t i = 0) {
        assert(i < num_elem_);
        return proxy_.get() + i;
    }

  private:
    /**
     * @brief Memory managed by the lifetime of this object
     */
    ProxyT proxy_ {};

    /**
     * @brief Number stats in profiler (on per block basis)
     */
    size_t num_elem_ {0};
};

using ProfilerProxyT = ProfilerProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_PROFILER_PROXY_HPP
