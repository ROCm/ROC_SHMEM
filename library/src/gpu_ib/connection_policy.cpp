/******************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#include "config.h"

#include "queue_pair.hpp"
#include "connection_policy.hpp"
#include "dynamic_connection.hpp"

#include <infiniband/mlx5dv.h>

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

RCConnectionImpl::RCConnectionImpl(Connection *conn,
                                   uint32_t *_vec_rkey)
{ }

DCConnectionImpl::DCConnectionImpl(Connection *conn,
                                   uint32_t *_vec_rkey)
  : vec_dct_num(static_cast<DynamicConnection *>(conn)->get_vec_dct_num()),
    vec_rkey(_vec_rkey),
    vec_lids(static_cast<DynamicConnection *>(conn)->get_vec_lids())
{ }

__device__ uint32_t
RCConnectionImpl::getNumWqesImpl(uint8_t opcode)
{
    return 1;
}

__device__ uint32_t
DCConnectionImpl::getNumWqesImpl(uint8_t opcode)
{
    // FIXME: we assume all threads in wave are performing ATOMIC ops
    // while this might be common, we do no thave such restriction
    // so need to be fixed
    return (opcode == MLX5_OPCODE_ATOMIC_FA ||
            opcode == MLX5_OPCODE_ATOMIC_CS) ? 2 : 1;
}

__device__ bool
RCConnectionImpl::updateConnectionSegmentImpl(mlx5_base_av *wqe, int pe)
{ return false; }

__device__ bool
DCConnectionImpl::updateConnectionSegmentImpl(mlx5_base_av *wqe, int pe)
{
    wqe->dqp_dct = vec_dct_num[pe];
    wqe->rlid = vec_lids[pe];
    return true;
}

__device__ void
RCConnectionImpl::setRkeyImpl(uint32_t &rkey, int pe)
{ }

__device__ void
DCConnectionImpl::setRkeyImpl(uint32_t &rkey, int pe)
{
    rkey = vec_rkey[pe];
}
