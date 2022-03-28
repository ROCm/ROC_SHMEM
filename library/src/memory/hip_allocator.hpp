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

#ifndef ROCSHMEM_LIBRARY_SRC_HIP_ALLOCATOR_HPP
#define ROCSHMEM_LIBRARY_SRC_HIP_ALLOCATOR_HPP

/**
 * @file hip_allocator.hpp
 *
 * @brief Contains HIP wrapper class for memory allocator
 */

#include <hip/hip_runtime_api.h>

#include "memory_allocator.hpp"

namespace rocshmem {

class HIPAllocator : public MemoryAllocator
{
  public:
    HIPAllocator()
        : MemoryAllocator(hipMalloc, hipFree)
    {
    }
};

class HIPAllocatorFinegrained : public MemoryAllocator
{
  public:
    HIPAllocatorFinegrained()
        : MemoryAllocator(hipExtMallocWithFlags, hipFree)
    {
    }
};

template <class T>
class StdAllocatorHIP
{
  public:
    typedef T value_type;

    StdAllocatorHIP () = default;

    template <class U>
    constexpr StdAllocatorHIP(const StdAllocatorHIP <U>&) noexcept {
    }

    [[nodiscard]] T* allocate(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }

        T* p {nullptr};
        allocator_.allocate(reinterpret_cast<void**>(&p), n * sizeof(T));
        if (p) {
            return p;
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, size_t n) noexcept {
        allocator_.deallocate(p);
    }

  private:
    HIPAllocatorFinegrained allocator_ {};
};

template <class T, class U>
bool operator==(const StdAllocatorHIP <T>&,
                const StdAllocatorHIP <U>&) {
    return true;
}

template <class T, class U>
bool operator!=(const StdAllocatorHIP <T>&,
                const StdAllocatorHIP <U>&) {
    return false;
}

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_HIP_ALLOCATOR_HPP
