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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_BLOCK_HANDLE_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_BLOCK_HANDLE_HPP_

#include "src/ipc_policy.hpp"
#include "src/hdp_policy.hpp"
#include "src/reverse_offload/profiler.hpp"
#include "src/reverse_offload/queue.hpp"

namespace rocshmem {

struct BlockHandle {
  ROStats profiler{};
  queue_element_t *queue{nullptr};
  uint64_t queue_size{QUEUE_SIZE};
  unsigned int *barrier_ptr{nullptr};
  volatile uint64_t read_index{};
  volatile uint64_t write_index{};
  volatile uint64_t *host_read_index{};
  volatile char *status{nullptr};
  char *g_ret{nullptr};
  atomic_ret_t atomic_ret{};
  IpcImpl ipc{};
  HdpPolicy hdp{};
  volatile uint64_t lock{};
};

template <typename ALLOCATOR>
class DefaultBlockHandleProxy {
  using ProxyT = DeviceProxy<ALLOCATOR, BlockHandle>;

 public:
  DefaultBlockHandleProxy() = default;

  DefaultBlockHandleProxy(unsigned *barrier_ptr, char *g_ret, atomic_ret_t *atomic_ret,
                   Queue *queue, IpcImpl *ipc_policy, HdpPolicy *hdp_policy) {
    // TODO(bpotter): create a default queue for this queue descriptor
    auto queue_descriptor{queue->descriptor(0)};
    auto block_handle{proxy_.get()};
    block_handle->profiler.resetStats();
    block_handle->queue = queue->elements(0);
    block_handle->queue_size = queue->size();
    block_handle->barrier_ptr = barrier_ptr;
    block_handle->read_index = queue_descriptor->read_index;
    block_handle->write_index = queue_descriptor->write_index;
    block_handle->host_read_index = &queue_descriptor->read_index;
    block_handle->status = queue_descriptor->status;
    block_handle->g_ret = g_ret;
    block_handle->atomic_ret.atomic_base_ptr = atomic_ret->atomic_base_ptr;
    block_handle->atomic_ret.atomic_counter = 0;
    block_handle->ipc.ipc_bases = ipc_policy->ipc_bases;
    block_handle->ipc.shm_size = ipc_policy->shm_size;
    block_handle->hdp = *hdp_policy;
    block_handle->lock = 0;
  }

  __host__ __device__ BlockHandle *get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

using DefaultBlockHandleProxyT = DefaultBlockHandleProxy<HIPAllocator>;

template <typename ALLOCATOR>
class BlockHandleProxy {
  static constexpr size_t MAX_NUM_BLOCKS{65536};
  using ProxyT = DeviceProxy<ALLOCATOR, BlockHandle, MAX_NUM_BLOCKS>;

 public:
  BlockHandleProxy() = default;

  BlockHandleProxy(unsigned *barrier_ptr, char *g_ret, atomic_ret_t *atomic_ret,
                   Queue *queue, IpcImpl *ipc_policy, HdpPolicy *hdp_policy) {
    for (size_t i{0}; i < MAX_NUM_BLOCKS; i++) {
      auto queue_descriptor{queue->descriptor(i)};
      auto block_handle{&proxy_.get()[i]};
      block_handle->profiler.resetStats();
      block_handle->queue = queue->elements(i);
      block_handle->queue_size = queue->size();
      block_handle->barrier_ptr = barrier_ptr;
      block_handle->read_index = queue_descriptor->read_index;
      block_handle->write_index = queue_descriptor->write_index;
      block_handle->host_read_index = &queue_descriptor->read_index;
      block_handle->status = queue_descriptor->status;
      block_handle->g_ret = g_ret;
      block_handle->atomic_ret.atomic_base_ptr = atomic_ret->atomic_base_ptr;
      block_handle->atomic_ret.atomic_counter = 0;
      block_handle->ipc.ipc_bases = ipc_policy->ipc_bases;
      block_handle->ipc.shm_size = ipc_policy->shm_size;
      block_handle->hdp = *hdp_policy;
      block_handle->lock = 0;
    }
  }

  __host__ __device__ BlockHandle *get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};
};

using BlockHandleProxyT = BlockHandleProxy<HIPAllocator>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_BLOCK_HANDLE_HPP_
