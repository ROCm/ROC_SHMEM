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

#include "connection_policy.hpp"

#include <infiniband/mlx5dv.h>

#include "config.h"  // NOLINT(build/include_subdir)
#include "dynamic_connection.hpp"
#include "queue_pair.hpp"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

namespace rocshmem {

RCConnectionImpl::RCConnectionImpl(Connection* conn,
                                   uint32_t* _vec_rkey) {
}

DCConnectionImpl::DCConnectionImpl(Connection* conn,
                                   uint32_t* _vec_rkey)
  : vec_dct_num(static_cast<DynamicConnection*>(conn)->get_vec_dct_num()),
    vec_rkey(_vec_rkey),
    vec_lids(static_cast<DynamicConnection*>(conn)->get_vec_lids()) {
}

__device__ uint32_t
RCConnectionImpl::getNumWqesImpl(uint8_t opcode) {
    return 1;
}

__device__ uint32_t
DCConnectionImpl::getNumWqesImpl(uint8_t opcode) {
    // FIXME: We assume all threads in wave are performing ATOMIC ops.
    // While this might be common, we do not have such restriction
    // so need to be fixed.
    // Since OFED 5.2, a DC segments uses 48bytes - so with or without
    // atomic we need 2 wqes.
    // return 2;
    return (opcode == MLX5_OPCODE_ATOMIC_FA ||
            opcode == MLX5_OPCODE_ATOMIC_CS) ? 2 : 1;
}

__device__ bool
RCConnectionImpl::updateConnectionSegmentImpl(ib_mlx5_base_av_t* wqe, int pe) {
    return false;
}

__device__ bool
DCConnectionImpl::updateConnectionSegmentImpl(ib_mlx5_base_av_t* wqe, int pe) {
    wqe->dqp_dct = vec_dct_num[pe];
    wqe->rlid = vec_lids[pe];
    return true;
}

__device__ void
RCConnectionImpl::setRkeyImpl(uint32_t* rkey, int pe) {
}

__device__ void
DCConnectionImpl::setRkeyImpl(uint32_t* rkey, int pe) {
    *rkey = vec_rkey[pe];
}

}  // namespace rocshmem
