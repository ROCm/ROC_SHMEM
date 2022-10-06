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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_WIN_POOL_BITMASK_PROXY_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_WIN_POOL_BITMASK_PROXY_HPP

#include "window_info.hpp"
#include "device_proxy.hpp"
#include "mpi_transport.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class WinPoolBitmaskProxy {
    static constexpr size_t MAX_NUM_CTXS {32};

    using ProxyT = DeviceProxy<ALLOCATOR, int, MAX_NUM_CTXS>;

  public:
    /*
     * Placement new the memory which is allocated by proxy_
     */
    WinPoolBitmaskProxy() {
        size_t bitmask_size_bytes {MAX_NUM_CTXS * sizeof(int)};
        memset(proxy_.get(), 0, bitmask_size_bytes);
    }

    /*
     * Since placement new is called in the constructor, then
     * delete must be called manually.
     */
    ~WinPoolBitmaskProxy() {
    }

    /*
     * @brief Provide access to the memory referenced by the proxy
     */
    __host__ __device__
    int*
    get() {
        return proxy_.get();
    }

  private:
    /*
     * @brief Memory managed by the lifetime of this object
     */
    ProxyT proxy_ {};
};

using WinPoolBitmaskProxyT = WinPoolBitmaskProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_WIN_POOL_BITMASK_PROXY_HPP
