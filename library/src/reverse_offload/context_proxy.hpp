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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_PROXY_HPP_

#include "include/roc_shmem.hpp"
#include "src/device_proxy.hpp"
#include "src/memory/hip_allocator.hpp"
#include "src/reverse_offload/context_ro_device.hpp"

namespace rocshmem {

class ROBackend;

template <typename ALLOCATOR>
class DefaultContextProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, ROContext>;

 public:
  DefaultContextProxy() = default;

  /*
   * Placement new the memory which is allocated by proxy_
   */
  explicit DefaultContextProxy(ROBackend* backend) : constructed_{true} {
    auto ctx{proxy_.get()};
    new (ctx) ROContext(reinterpret_cast<Backend*>(backend), -1);
    roc_shmem_ctx_t local {ctx, nullptr};
    set_internal_ctx(&local);
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

  DefaultContextProxy& operator=(const DefaultContextProxy& other) = delete;

  DefaultContextProxy(DefaultContextProxy&& other) = default;

  DefaultContextProxy& operator=(DefaultContextProxy&& other) = default;

  /*
   * @brief Provide access to the memory referenced by the proxy
   */
  __host__ __device__ Context* get() { return proxy_.get(); }

 private:
  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};

  /*
   * @brief denotes if an objects was constructed in proxy
   */
  bool constructed_{false};
};

using DefaultContextProxyT = DefaultContextProxy<HIPAllocator>;

template <typename ALLOCATOR>
class BlockContextProxy {
  static constexpr size_t MAX_NUM_BLOCKS{65536};
  using ProxyT = DeviceProxy<ALLOCATOR, ROContext, MAX_NUM_BLOCKS>;

 public:
  BlockContextProxy() = default;

  explicit BlockContextProxy(ROBackend* backend) : constructed_{true} {
    auto* ctx{proxy_.get()};
    for (size_t i{0}; i < MAX_NUM_BLOCKS; i++) {
      auto ctx_p {&ctx[i]};
      new (ctx_p) ROContext(reinterpret_cast<Backend*>(backend), i);
    }
  }

  ~BlockContextProxy() {
    if (constructed_) {
      auto* ctx{proxy_.get()};
      for (size_t i{0}; i < MAX_NUM_BLOCKS; i++) {
        auto ctx_p {&ctx[i]};
        ctx_p->~ROContext();
      }
    }
  }

  BlockContextProxy(const BlockContextProxy& other) = delete;

  BlockContextProxy& operator=(const BlockContextProxy& other) = delete;

  BlockContextProxy(BlockContextProxy&& other) = default;

  BlockContextProxy& operator=(BlockContextProxy&& other) = default;

  __host__ __device__ Context* get() { return proxy_.get(); }

 private:
  /*
   * @brief Memory managed by the lifetime of this object
   */
  ProxyT proxy_{};

  /*
   * @brief denotes if an objects was constructed in proxy
   */
  bool constructed_{false};
};

using BlockContextProxyT = BlockContextProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_PROXY_HPP_
