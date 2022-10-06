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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_DESC_PROXY_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_DESC_PROXY_HPP

#include "ro_net_internal.hpp"
#include "device_proxy.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class QueueDescProxy {
    static constexpr size_t MAX_NUM_BLOCKS {65536};
    static constexpr size_t MAX_THREADS_PER_BLOCK {1024};
    static constexpr size_t MAX_THREADS {MAX_NUM_BLOCKS * MAX_THREADS_PER_BLOCK};

    using ProxyT = DeviceProxy<ALLOCATOR, queue_desc_t, MAX_NUM_BLOCKS>;
    using ProxyStatusT = DeviceProxy<ALLOCATOR, char, MAX_THREADS>;

  public:
    QueueDescProxy() {
        auto *status {proxy_status_.get()};
        size_t status_bytes {sizeof(char) * MAX_THREADS};
        memset(status, 0, status_bytes);

        auto *queue_descs {proxy_.get()};
        for (size_t i {0}; i < MAX_NUM_BLOCKS; i++) {
            queue_descs[i].read_idx = 0;
            queue_descs[i].write_idx = 0;
            queue_descs[i].status = status + i * MAX_THREADS_PER_BLOCK;
        }
    }

    /*
     * @brief Provide access to the memory referenced by the proxy
     */
    __host__ __device__
    queue_desc_t*
    get() {
        return proxy_.get();
    }

  private:
    /**
     * @brief Memory managed by the lifetime of this object
     */
    ProxyT proxy_ {};

    /**
     * @brief Holds the status bytes
     *
     * There is a status variable for each thread in a block. We just
     * over-allocate for the maximum block size.
     *
     * Status always goes in device memory to prevent polling for
     * completion over PCIe
     */
    ProxyStatusT proxy_status_ {};
};

using QueueDescProxyT = QueueDescProxy<HIPAllocatorFinegrained>;

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_DESC_PROXY_HPP
