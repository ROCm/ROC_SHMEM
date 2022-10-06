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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_DEFAULT_CONTEXT_PROXY_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_DEFAULT_CONTEXT_PROXY_HPP

#include <hip/hip_runtime_api.h>

#include <roc_shmem.hpp>
#include "context_ro_device.hpp"
#include "device_proxy.hpp"
#include "hip_allocator.hpp"

namespace rocshmem {

template <typename ALLOCATOR>
class DefaultContextProxy {
    using ProxyT = DeviceProxy<ALLOCATOR, ROContext>;

  public:
    DefaultContextProxy() = default;

    /*
     * Placement new the memory which is allocated by proxy_
     */
    DefaultContextProxy(Backend *backend)
        : constructed_{true} {
        auto *ctx {proxy_.get()};

        // TODO: @Brandon fix this pass by reference for the backend
        new (ctx) ROContext(backend, 0);

        /*
         * The device code will reference the default context through a global
         * variable since the context will not be passed as a handle to the
         * library API. To make the default context available, we need to copy
         * it over to the device constant space using the code below.
         */
        int *symbol_address {nullptr};

        CHECK_HIP(hipGetSymbolAddress((void**)&symbol_address,
                                      HIP_SYMBOL(ROC_SHMEM_CTX_DEFAULT)));

        CHECK_HIP(hipMemcpy(symbol_address,
                            &ctx,
                            sizeof(ctx),
                            hipMemcpyDefault));
    }

    /*
     * Since placement new is called in the constructor, then
     * delete must be called manually.
     */
    ~DefaultContextProxy() {
        if (constructed_) {
            proxy_.get()->~ROContext();
        }
    }

    DefaultContextProxy(const DefaultContextProxy& other) = delete;

    DefaultContextProxy&
    operator=(const DefaultContextProxy& other) = delete;

    DefaultContextProxy(DefaultContextProxy&& other) = default;

    DefaultContextProxy&
    operator=(DefaultContextProxy&& other) = default;

    /*
     * @brief Provide access to the memory referenced by the proxy
     */
    __host__ __device__
    Context*
    get() {
        return proxy_.get();
    }

  private:
    /*
     * @brief Memory managed by the lifetime of this object
     */
    ProxyT proxy_ {};

    /*
     * @brief denotes if an objects was constructed in proxy
     */
    bool constructed_ {false};
};

using DefaultContextProxyT = DefaultContextProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_DEFAULT_CONTEXT_PROXY_HPP
