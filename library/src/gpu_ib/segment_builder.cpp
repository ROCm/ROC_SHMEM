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

#include "segment_builder.hpp"

#include "endian.hpp"

__device__
SegmentBuilder::SegmentBuilder(uint64_t wqe_idx,
                               void *base) {
    mlx5_segment *base_ptr = static_cast<mlx5_segment*>(base);
    size_t segment_offset = SEGMENTS_PER_WQE * wqe_idx;
    seg_ptr = &base_ptr[segment_offset];
}

__device__ void
SegmentBuilder::update_cntrl_seg(uint8_t opcode,
                                 uint16_t wqe_idx,
                                 uint32_t ctrl_qp_sq,
                                 uint64_t ctrl_sig,
                                 ConnectionImpl *connection_policy,
                                 bool zero_byte_rd) {
    mlx5_wqe_ctrl_seg *ctrl_seg = &seg_ptr->ctrl_seg;

    ctrl_seg->opmod_idx_opcode = (opcode << 24) | (wqe_idx << 8);

    uint32_t DS = 2;
    if (zero_byte_rd == false) {
        DS = (opcode == MLX5_OPCODE_RDMA_WRITE ||
              opcode == MLX5_OPCODE_RDMA_READ) ? 3 : 4;
    }

    DS += connection_policy->wqeCntrlOffset();

    ctrl_seg->qpn_ds = (DS << 24) | ctrl_qp_sq;

    ctrl_seg->signature = ctrl_sig;

    ctrl_seg->fm_ce_se = ctrl_sig >> 24;

    ctrl_seg->imm = ctrl_sig >> 32;

    seg_ptr++;
}

__device__ void
SegmentBuilder::update_atomic_data_seg(uint64_t atomic_data,
                                       uint64_t atomic_cmp) {
    mlx5_wqe_atomic_seg *atomic_seg = &seg_ptr->atomic_seg;

    swap_endian_store(reinterpret_cast<uint64_t*>(&atomic_seg->swap_add),
                      atomic_data);

    swap_endian_store(reinterpret_cast<uint64_t*>(&atomic_seg->compare),
                      atomic_cmp);

    seg_ptr++;
}

__device__ void
SegmentBuilder::update_rdma_seg(uintptr_t *raddr,
                                uint32_t rkey) {
    mlx5_wqe_raddr_seg *raddr_seg = &seg_ptr->raddr_seg;

    raddr_seg->rkey = rkey;

    swap_endian_store(reinterpret_cast<uint64_t*>(&raddr_seg->raddr),
                      reinterpret_cast<uint64_t>(raddr));

    seg_ptr++;
}

__device__ void
SegmentBuilder::update_data_seg(uintptr_t *laddr,
                                int32_t size,
                                uint32_t lkey) {
    if (laddr == nullptr) {
        return;
    }

    mlx5_wqe_data_seg *data_seg = &seg_ptr->data_seg;

    data_seg->lkey = lkey;

    swap_endian_store(&data_seg->byte_count,
                      size & 0x7FFFFFFFU);

    swap_endian_store(reinterpret_cast<uint64_t*>(&data_seg->addr),
                      reinterpret_cast<uint64_t>(laddr));

    seg_ptr++;
}

__device__ void
SegmentBuilder::update_inl_data_seg(uintptr_t *laddr,
                                    int32_t size) {
    mlx5_wqe_inl_data_seg *inl_data_seg = &seg_ptr->inl_data_seg;

    swap_endian_store(&inl_data_seg->byte_count,
                      (size & 0x3FF) | 0x80000000);

    // Assume fence HDP flush
    // TODO(khamidou): Rework fence interface to avoid this
    if (!laddr) {
        uint8_t flush_val = 1;
        memcpy(inl_data_seg + 1, &flush_val, sizeof(flush_val));
    } else {
        memcpy(inl_data_seg + 1, laddr, size);
    }

    seg_ptr++;
}

__device__ void
SegmentBuilder::
update_connection_seg(int pe, ConnectionImpl *conn_policy) {
    if (conn_policy->updateConnectionSegmentImpl(&seg_ptr->base_av, pe)) {
        seg_ptr++;
    }
}
