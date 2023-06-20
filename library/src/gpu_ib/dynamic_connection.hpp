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

#ifndef LIBRARY_SRC_GPU_IB_DYNAMIC_CONNECTION_HPP_
#define LIBRARY_SRC_GPU_IB_DYNAMIC_CONNECTION_HPP_

#include "src/gpu_ib/connection.hpp"

namespace rocshmem {

class DynamicConnection : public Connection {
 public:
  explicit DynamicConnection(GPUIBBackend* backend);

  ~DynamicConnection() override;

  void get_remote_conn(int* remote_conn) override;

  void post_wqes() override;

  void initialize_rkey_handle(uint32_t** heap_rkey_handle,
                                ibv_mr* mr) override;

  void free_rkey_handle(uint32_t* heap_rkey_handle) override;

  uint32_t* get_vec_dct_num() const { return vec_dct_num; }

  uint16_t* get_vec_lids() const { return vec_lids; }

 private:
  InitQPState initqp(uint8_t port) override;

  RtrState rtr(dest_info_t* dest, uint8_t port) override;

  RtsState rts(dest_info_t* dest) override;

  QPInitAttr qpattr(ibv_qp_cap cap) override;

  void connect_dci(ibv_qp* qp, uint8_t port);

  void create_dct(int32_t* dct_num, ibv_cq* cq, ibv_srq* srq, uint8_t port);

  ibv_qp_init_attr_ex dct_qp_init_attr(ibv_cq* cq, ibv_srq* srq,
                                       uint8_t port) const;

  mlx5dv_qp_init_attr dct_dv_init_attr();

  void dc_get_av(ibv_ah* ah, mlx5_wqe_av* mlx5_av);

  void set_dgram_seg(mlx5_wqe_datagram_seg* dc_seg, uint64_t dc_key,
                     uint32_t dct_num, uint8_t ext, mlx5_wqe_av* av);

  void set_data_seg(mlx5_wqe_data_seg* data_seg, uint32_t lkey);

  void post_dv_dc_wqe(int remote_conn);

  void create_qps_1() override;

  void create_qps_2(int port, int my_rank,
                      ibv_port_attr* ib_port_att) override;

  void create_qps_3(int port, ibv_qp* qp, int offset,
                      ibv_port_attr* ib_port_att) override;

  ibv_qp* create_qp_0(ibv_context* context,
                      ibv_qp_init_attr_ex* qp_attr) override;

  void allocate_dynamic_members(int num_wg) override;

  void free_dynamic_members() override;

  void initialize_1(int port, int num_wg) override;

  void initialize_wr_fields(ibv_send_wr* wr, ibv_ah* ah, int dc_key) override;

  int get_sq_dv_offset(int pe_idx, int32_t num_qps, int wg_idx) override;

  int num_dcis{1};

  int num_dct{1};

  static constexpr int DC_IB_KEY{0x1ee7a330};

  uint32_t* dcts_num{nullptr};

  uint16_t* lids{nullptr};

  mlx5_wqe_av mlx5_av{};

  ibv_ah* ah{nullptr};

  ibv_srq* srq{nullptr};

  ibv_cq* dct_cq{nullptr};

  uint32_t* vec_dct_num{nullptr};

  uint16_t* vec_lids{nullptr};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_DYNAMIC_CONNECTION_HPP_
