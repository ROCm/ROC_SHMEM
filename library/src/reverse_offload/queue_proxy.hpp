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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_PROXY_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_PROXY_HPP

#include "ro_net_internal.hpp"
#include "device_proxy.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class QueueProxy {
    static constexpr size_t MAX_NUM_BLOCKS {65536};
    static constexpr size_t QUEUE_SIZE {1024};
    static constexpr size_t TOTAL_QUEUE_ELEMENTS {QUEUE_SIZE * MAX_NUM_BLOCKS};

    using ProxyT = DeviceProxy<ALLOCATOR, queue_element_t*, MAX_NUM_BLOCKS>;
    using ProxyPerBlockT = DeviceProxy<ALLOCATOR, queue_element_t, TOTAL_QUEUE_ELEMENTS>;

  public:
    /**
     * @brief Initializes a c-style array of circular queues.
     *
     * The circular queues are indexed using the device block-id so that each
     * each block has its own queue.
     */
    QueueProxy() {
        auto **queue_array {queue_proxy_.get()};
        auto *per_block_queue {per_block_queue_proxy_.get()};
        for (size_t i {0}; i < MAX_NUM_BLOCKS; i++) {
            queue_array[i] = per_block_queue + i * QUEUE_SIZE;
        }
        size_t total_queue_element_bytes {sizeof(queue_element_t) * TOTAL_QUEUE_ELEMENTS};
        memset(per_block_queue, 0, total_queue_element_bytes);
    }

    /*
     * @brief Provide access to the memory referenced by the proxy
     */
    __host__ __device__
    queue_element_t**
    get() {
        return queue_proxy_.get();
    }

  private:
    /**
     * @brief Memory managed by the lifetime of this object
     */
    ProxyT queue_proxy_ {};

    /**
     * @brief Holds the queue elements
     */
    ProxyPerBlockT per_block_queue_proxy_ {};
};

using QueueProxyT = QueueProxy<HIPHostAllocator>;

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_PROXY_HPP
