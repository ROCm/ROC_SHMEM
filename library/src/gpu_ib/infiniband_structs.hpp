/******************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LIBRARY_SRC_GPU_IB_INFINIBAND_STRUCTS_HPP_
#define LIBRARY_SRC_GPU_IB_INFINIBAND_STRUCTS_HPP_

#include <infiniband/mlx5dv.h>

typedef struct ib_mlx5_base_av {
    uint64_t dc_key;
    uint32_t dqp_dct;
    uint8_t stat_rate_sl;
    uint8_t fl_mlid;
    uint16_t rlid;
} ib_mlx5_base_av_t;

union mlx5_segment {
    mlx5_wqe_ctrl_seg ctrl_seg;
    mlx5_wqe_raddr_seg raddr_seg;
    mlx5_wqe_atomic_seg atomic_seg;
    mlx5_wqe_data_seg data_seg;
    mlx5_wqe_inl_data_seg inl_data_seg;
    ib_mlx5_base_av_t base_av;
};

#endif  // LIBRARY_SRC_GPU_IB_INFINIBAND_STRUCTS_HPP_
