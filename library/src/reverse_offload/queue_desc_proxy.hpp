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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_DESC_PROXY_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_DESC_PROXY_HPP_

#include "src/device_proxy.hpp"

namespace rocshmem {

typedef struct queue_desc {
  /**
   * Read index for the queue. Rarely read by the GPU when it thinks the
   * queue might be full, but the GPU normally uses a local copy that lags
   * behind the true read_index.
   */
  uint64_t read_index;
  char padding1[56];
  /**
   * Write index for the queue. Never accessed by CPU, since it uses the
   * valid bit in the packet itself to determine whether there is data to
   * consume. The GPU has a local copy of the write_index that it uses, but it
   * does write the local index to this location when the kernel completes
   * in case the queue needs to be reused without resetting all the pointers
   * to zero.
   */
  uint64_t write_index;
  char padding2[56];
  /**
   * This bit is used by the GPU to wait on a blocking operation. The initial
   * value is 0. When a GPU enqueues a blocking operation, it waits for this
   * value to resolve to 1, which is set by the CPU when the blocking
   * operation completes. The GPU then resets status back to zero. There is
   * a separate status variable for each work-item in a work-group
   */
  char *status;
  char padding3[56];
} __attribute__((__aligned__(64))) queue_desc_t;

template <typename ALLOCATOR>
class QueueDescProxy {
  static constexpr size_t MAX_NUM_BLOCKS{65536};
  static constexpr size_t MAX_THREADS_PER_BLOCK{1024};
  static constexpr size_t MAX_THREADS{MAX_NUM_BLOCKS * MAX_THREADS_PER_BLOCK};
  using ProxyT = DeviceProxy<ALLOCATOR, queue_desc_t, MAX_NUM_BLOCKS>;
  using ProxyStatusT = DeviceProxy<ALLOCATOR, char, MAX_THREADS>;

 public:
  QueueDescProxy() {
    auto *status{proxy_status_.get()};
    size_t status_bytes{sizeof(char) * MAX_THREADS};
    memset(status, 0, status_bytes);

    auto *queue_descs{proxy_.get()};
    for (size_t i{0}; i < MAX_NUM_BLOCKS; i++) {
      queue_descs[i].read_index = 0;
      queue_descs[i].write_index = 0;
      queue_descs[i].status = status + i * MAX_THREADS_PER_BLOCK;
    }
  }

  __host__ __device__ queue_desc_t *get() { return proxy_.get(); }

 private:
  ProxyT proxy_{};

  ProxyStatusT proxy_status_{};
};

using QueueDescProxyT = QueueDescProxy<HIPAllocatorFinegrained>;

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_QUEUE_DESC_PROXY_HPP_
