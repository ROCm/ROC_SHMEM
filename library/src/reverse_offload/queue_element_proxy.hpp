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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_ELEMENT_PROXY_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_ELEMENT_PROXY_HPP

#include "device_proxy.hpp"
#include "ro_net_internal.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class QueueElementProxy {
    using ProxyT = DeviceProxy<ALLOCATOR, queue_element_t>;

  public:
    /*
     * Placement new the memory which is allocated by proxy_
     */
    QueueElementProxy() {
        new (proxy_.get()) queue_element_t();
    }

    /*
     * Since placement new is called in the constructor, then
     * delete must be called manually.
     */
    ~QueueElementProxy() {
        proxy_.get()->~queue_element_t();
    }

    /*
     * @brief Provide access to the memory referenced by the proxy
     */
    __host__ __device__
    queue_element_t*
    get() {
        return proxy_.get();
    }

  private:
    /*
     * @brief Memory managed by the lifetime of this object
     */
    ProxyT proxy_ {};
};

using QueueElementProxyT = QueueElementProxy<PosixAligned64Allocator>;

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_ELEMENT_PROXY_HPP
