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

#ifndef LIBRARY_SRC_MEMORY_DEV_MONO_LINEAR_HPP_
#define LIBRARY_SRC_MEMORY_DEV_MONO_LINEAR_HPP_

#include <cassert>

#include "src/memory/shmem_allocator_strategy.hpp"
#include "src/util.hpp"

/**
 * @file dev_mono_linear.hpp
 *
 * @brief Contains an allocator strategy for the heap.
 *
 * This strategy returns memory chunks by monotonically increasing a pointer.
 */

namespace rocshmem {

template <typename HM_T>
class DevMonoLinear : public ShmemAllocatorStrategy {
 public:
  /**
   * @brief Required for default construction of other objects
   *
   * @note Not intended for direct usage.
   */
  DevMonoLinear() = default;

  /**
   * @brief Primary constructor type
   *
   * @param[in] Raw pointer to heap memory type
   */
  explicit DevMonoLinear(HM_T* heap_mem)
      : heap_mem_{heap_mem}, current_ptr_{heap_mem_->get_ptr()} {}

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   */
  void alloc(char** ptr, size_t request_size) override {
    assert(ptr);
    *ptr = nullptr;

    if (!request_size) {
      return;
    }

    char* heap_end{heap_mem_->get_ptr() + heap_mem_->get_size()};

    if (current_ptr_ + request_size < heap_end) {
      *ptr = current_ptr_;
      current_ptr_ += request_size;
    }
  }

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in, out] Address of raw pointer (&pointer_to_char)
   * @param[in] Size in bytes of memory allocation
   */
  __device__ void alloc(char** ptr, size_t request_size) override {
    if (is_thread_zero_in_block()) {
      assert(ptr);
      *ptr = nullptr;

      if (!request_size) {
        return;
      }

      char* heap_end{heap_mem_->get_ptr() + heap_mem_->get_size()};

      if (current_ptr_ + request_size < heap_end) {
        *ptr = current_ptr_;
        current_ptr_ += request_size;
      }
    }
  }

  /**
   * @brief Frees memory from the heap
   *
   * Released memory ignored.
   *
   * @param[in] Raw pointer to heap memory
   */
  __host__ void free([[maybe_unused]] char* ptr) override {}

  /**
   * @brief Frees memory from the heap
   *
   * Released memory ignored.
   *
   * @param[in] Raw pointer to heap memory
   */
  __device__ void free([[maybe_unused]] char* ptr) override {}

  /**
   * @brief Return pointer to monotonic linear ptr
   *
   * @return Raw pointer to heap memory.
   */
  __host__ __device__ char* current() { return current_ptr_; }

 private:
  /**
   * The pointer is used to keep reference to heap memory.
   */
  HM_T* heap_mem_{nullptr};

  /**
   * The pointer is used to track the next allocation point.
   */
  char* current_ptr_{nullptr};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_DEV_MONO_LINEAR_HPP_
