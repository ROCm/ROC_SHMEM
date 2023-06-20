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

#ifndef LIBRARY_SRC_GPU_IB_SEGMENT_BUILDER_HPP_
#define LIBRARY_SRC_GPU_IB_SEGMENT_BUILDER_HPP_

#include <infiniband/mlx5dv.h>

#include "src/gpu_ib/connection_policy.hpp"
#include "src/gpu_ib/infiniband_structs.hpp"
#include "src/util.hpp"

namespace rocshmem {

class SegmentBuilder {
 public:
  __device__ SegmentBuilder(uint64_t wqe_idx, void *base);

  __device__ void update_cntrl_seg(uint8_t opcode, uint16_t wqe_idx,
                                   uint32_t ctrl_qp_sq, uint64_t ctrl_sig,
                                   ConnectionImpl *connection_policy,
                                   bool zero_byte_rd);

  __device__ void update_connection_seg(int pe,
                                        ConnectionImpl *connection_policy);

  __device__ void update_atomic_data_seg(uint64_t atomic_data,
                                         uint64_t atomic_cmp);

  __device__ void update_rdma_seg(uintptr_t *raddr, uint32_t rkey);

  __device__ void update_inl_data_seg(uintptr_t *laddr, int32_t size);

  __device__ void update_data_seg(uintptr_t *laddr, int32_t size,
                                  uint32_t lkey);

 private:
  const int SEGMENTS_PER_WQE = 4;

  mlx5_segment *seg_ptr;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_SEGMENT_BUILDER_HPP_
