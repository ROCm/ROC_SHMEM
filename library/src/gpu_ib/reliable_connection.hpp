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

#ifndef LIBRARY_SRC_GPU_IB_RELIABLE_CONNECTION_HPP_
#define LIBRARY_SRC_GPU_IB_RELIABLE_CONNECTION_HPP_

#include <vector>

#include "src/gpu_ib/connection.hpp"

namespace rocshmem {

class ReliableConnection : public Connection {
 public:
  explicit ReliableConnection(GPUIBBackend* backend);

  ~ReliableConnection() override;

  void get_remote_conn(int* remote_conn) override;

  void post_wqes() override;

  void initialize_rkey_handle(uint32_t** heap_rkey_handle,
                                ibv_mr* mr) override;

  void free_rkey_handle(uint32_t* heap_rkey_handle) override;

 private:
  InitQPState initqp(uint8_t port) override;

  RtrState rtr(dest_info_t* dest, uint8_t port) override;

  RtsState rts(dest_info_t* dest) override;

  QPInitAttr qpattr(ibv_qp_cap cap) override;

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

  int get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx) override;

  std::vector<dest_info_t> all_qp;

  void post_dv_rc_wqe(int remote_conn);
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_RELIABLE_CONNECTION_HPP_
