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

#ifndef LIBRARY_SRC_GPU_IB_MEMORY_BUILDER_POLICY_HPP_
#define LIBRARY_SRC_GPU_IB_MEMORY_BUILDER_POLICY_HPP_

#include <hip/hip_runtime.h>

#include <utility>

namespace rocshmem {

class GPUIBContext;

class MemoryBuilderPolicyWrapper {
 public:
  __device__ MemoryBuilderPolicyWrapper() = default;

  __device__ ~MemoryBuilderPolicyWrapper() {
    if (wrapped_policy_) {
      delete wrapped_policy_;
    }
  }

  template <typename T>
  __device__ MemoryBuilderPolicyWrapper(T&& policy)
      : wrapped_policy_(new Wrapper<T>(std::forward<T>(policy))) {}

  __device__ void operator()(GPUIBContext* context) {
    return (*wrapped_policy_)(context);
  }

 private:
  class PolicyBase {
   public:
    __device__ virtual void operator()(GPUIBContext* context) = 0;

    __device__ virtual ~PolicyBase() {}
  };

  template <typename T>
  class Wrapper : public PolicyBase {
   public:
    __device__ Wrapper(const T& t) : wrapped_policy_(t) {}

    __device__ void operator()(GPUIBContext* context) override {
      return wrapped_policy_(context);
    }

   private:
    T wrapped_policy_;
  };

  PolicyBase* wrapped_policy_;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_MEMORY_BUILDER_POLICY_HPP_
