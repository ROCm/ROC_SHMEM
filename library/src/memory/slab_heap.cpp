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

#include "src/memory/slab_heap.hpp"

#include <sstream>

#include "src/util.hpp"

namespace rocshmem {

SlabHeap::SlabHeap() {
  if (auto slab_size_cstr = getenv("ROC_SHMEM_SLAB_SIZE")) {
    std::stringstream sstream(slab_size_cstr);
    size_t slab_size;
    sstream >> slab_size;
    heap_mem_ = HEAP_T{slab_size};
    strat_ = STRAT_T{&heap_mem_};
  }
}

void SlabHeap::malloc(void** ptr, size_t size) {
  strat_.alloc(reinterpret_cast<char**>(ptr), size);
}

__device__ void SlabHeap::malloc(void** ptr, size_t size) {
  /*
   * Grab the mutex from the proxy object which owns it.
   */
  auto mutex{mutex_.get()};

  /*
   * Take the ticketed lock.
   *
   * The lock is held jointly by all threads in the block.
   */
  auto ticket{mutex->lock()};

  /*
   * Perform allocation and verify it worked.
   *
   * Allocation should only be run by only one thread in the
   * strategy code.
   */
  char** ptr_c{reinterpret_cast<char**>(ptr)};
  strat_.alloc(ptr_c, size);
  __threadfence();

  /*
   * The notifier works with uint64_t for the address broadcasts
   * between threads (as type erasure for the pointer arithmetic).
   */
  uint64_t ptr_deref_u64{reinterpret_cast<uint64_t>(*ptr)};

  /*
   * Notify other threads in block about the allocation result.
   */
  auto notifier{notifier_.get()};
  notifier->write(ptr_deref_u64);
  uint64_t notification_u64{notifier->read()};
  notifier->done();

  /*
   * Write to the ptr parameter (to return it back up the call stack).
   */
  char* read_value_c{reinterpret_cast<char*>(notification_u64)};
  *ptr_c = read_value_c;

  /*
   * Release the lock with our ticket number.
   */
  mutex->unlock(ticket);
}

__host__ __device__ void SlabHeap::free(void* ptr) {
  if (!ptr) {
    return;
  }
  strat_.free(reinterpret_cast<char*>(ptr));
}

void* SlabHeap::realloc(void* ptr, size_t size) { return nullptr; }

void* SlabHeap::malign(size_t alignment, size_t size) { return nullptr; }

char* SlabHeap::get_base_ptr() { return heap_mem_.get_ptr(); }

size_t SlabHeap::get_size() { return heap_mem_.get_size(); }

size_t SlabHeap::get_used() { return strat_.current() - get_base_ptr(); }

size_t SlabHeap::get_avail() { return get_size() - get_used(); }

}  // namespace rocshmem
