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

#ifndef LIBRARY_SRC_MEMORY_SLAB_HEAP_HPP_
#define LIBRARY_SRC_MEMORY_SLAB_HEAP_HPP_

#include "src/memory/dev_mono_linear.hpp"
#include "src/memory/heap_memory.hpp"
#include "src/memory/heap_type.hpp"
#include "src/memory/notifier.hpp"
#include "src/sync/abql_block_mutex.hpp"

/**
 * @file slab_heap.hpp
 *
 * @brief Contains a heap used to allocate library objects
 *
 * The slab heap is used internally by the library
 */

namespace rocshmem {

class SlabHeap {
  /**
   * @brief Helper type for allocation strategy
   */
  using STRAT_T = DevMonoLinear<HEAP_T>;

  /**
   * @brief Helper type for notifier
   */
  using NOTIFIER_PROXY_T = NotifierProxy<HIPAllocator>;

  /**
   * @brief Helper type for notifier
   */
  using MUTEX_PROXY_T = ABQLBlockMutexProxy<HIPAllocator>;

 public:
  /**
   * @brief Primary constructor
   */
  SlabHeap();

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in,out] A pointer to memory handle
   * @param[in] Size in bytes of memory allocation
   */
  void malloc(void** ptr, size_t size);

  /**
   * @brief Allocates memory from the heap
   *
   * @param[in,out] A pointer to memory handle
   * @param[in] Size in bytes of memory allocation
   */
  __device__ void malloc(void** ptr, size_t size);

  /**
   * @brief Frees memory from the heap
   *
   * @param[in] Raw pointer to heap memory
   */
  __host__ __device__ void free(void* ptr);

  /**
   * @brief
   *
   * @param[in]
   * @param[in]
   *
   * @return
   */
  void* realloc(void* ptr, size_t size);

  /**
   * @brief
   *
   * @param[in]
   * @param[in]
   *
   * @return
   */
  void* malign(size_t alignment, size_t size);

  /**
   * @brief Accessor for heap base ptr
   *
   * @return Pointer to base of my heap
   */
  char* get_base_ptr();

  /**
   * @brief Accessor for heap size
   *
   * @return Amount of bytes in heap
   */
  size_t get_size();

  /**
   * @brief Accessor for heap usage
   *
   * @return Amount of used bytes in heap
   */
  size_t get_used();

  /**
   * @brief Accessor for heap available
   *
   * @return Amount of available bytes in heap
   */
  size_t get_avail();

 private:
  /**
   * @brief Heap memory object
   */
  HEAP_T heap_mem_{};

  /**
   * @brief Allocation strategy object
   */
  STRAT_T strat_{&heap_mem_};

  /**
   * @brief Notifier proxy to share information between threads.
   *
   * Need this object to share allocation information between the
   * leader thread (that does allocation) and the follower threads
   * (who need the allocation address).
   */
  NOTIFIER_PROXY_T notifier_{};

  /**
   * @brief Mutex to access the heap mutator methods.
   */
  MUTEX_PROXY_T mutex_;
};

template <typename ALLOCATOR>
class SlabHeapProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, SlabHeap, 1>;

 public:
  /*
   * Placement new the memory which is allocated by proxy_
   */
  SlabHeapProxy() { new (proxy_.get()) SlabHeap(); }

  /*
   * Since placement new is called in the constructor, then
   * delete must be called manually.
   */
  ~SlabHeapProxy() { proxy_.get()->~SlabHeap(); }

  __host__ __device__ SlabHeap* get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_MEMORY_SLAB_HEAP_HPP_
