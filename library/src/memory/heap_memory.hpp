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

#ifndef ROCSHMEM_LIBRARY_SRC_HEAP_MEMORY_HPP
#define ROCSHMEM_LIBRARY_SRC_HEAP_MEMORY_HPP

#include <memory>

/**
 * @file heap_memory.hpp
 *
 * @brief Contains heap memory class
 *
 * @note The heap memory class owns the symmetric heap memory allocation
 */

namespace rocshmem {

template <typename ALLOCATOR>
class HeapMemory {
  public:
    /**
     * @brief Primary constructor type
     *
     * Uses default heap size specified in class body.
     */
    HeapMemory() :
      HeapMemory(gibibyte_) {
    }

    /**
     * @brief Secondary constructor type
     *
     * @param[in] User-specified size used as heap size
     */
    HeapMemory(size_t size)
    : size_{size} {
        char* temp;
        allocator_.allocate(reinterpret_cast<void**>(&temp), size_);
        std::unique_ptr<char, Deleter> up {temp};
        up_ = std::move(up);
    }

    /**
     * @brief Accessor for heap ptr
     *
     * @return Raw memory pointer
     */
    char*
    get_ptr() {
        return up_.get();
    }

    /**
     * @brief Accessor for heap size
     *
     * @return Heap size
     */
    size_t
    get_size() {
        return size_;
    }

  private:
    /**
     * @brief Template type member with allocate and deallocate methods.
     */
    ALLOCATOR allocator_ {};

    /**
     * @brief Wrap deallocator into a functor for up_ template.
     */
    class Deleter {
      public:
        void operator()(void* x) {
            a_.deallocate(x);
        }

      private:
        ALLOCATOR a_;
    };

    /**
     * @brief Owning pointer to heap memory.
     */
    std::unique_ptr<char, Deleter> up_ {nullptr};

    /**
     * @brief Named constant for a gibibyte.
     */
    static constexpr size_t gibibyte_ {1 << 30};

    /**
     * @brief Size of heap memory.
     */
    size_t size_ {gibibyte_};
};

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_HEAP_MEMORY_HPP
