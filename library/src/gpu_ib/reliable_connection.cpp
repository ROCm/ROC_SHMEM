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

#include "src/gpu_ib/reliable_connection.hpp"

#include <mpi.h>

#include "src/gpu_ib/backend_ib.hpp"

namespace rocshmem {

ReliableConnection::ReliableConnection(GPUIBBackend* b) : Connection(b, 0) {}

ReliableConnection::~ReliableConnection() {}

Connection::InitQPState ReliableConnection::initqp(uint8_t port) {
  InitQPState init{};

  init.exp_qp_attr.qp_access_flags =
      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
      IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  init.exp_qp_attr.port_num = port;

  init.exp_attr_mask |= IBV_QP_ACCESS_FLAGS;

  return init;
}

Connection::RtrState ReliableConnection::rtr(dest_info_t* dest, uint8_t port) {
  RtrState rtr{};

  rtr.exp_qp_attr.dest_qp_num = dest->qpn;
  rtr.exp_qp_attr.rq_psn = dest->psn;
  rtr.exp_qp_attr.ah_attr.port_num = port;
  if (ib_state->portinfo.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    rtr.exp_qp_attr.ah_attr.dlid = dest->lid;
  } else {
    rtr.exp_qp_attr.ah_attr.is_global = 1;
    rtr.exp_qp_attr.ah_attr.grh.dgid = dest->gid;
    rtr.exp_qp_attr.ah_attr.grh.sgid_index = 0;
    rtr.exp_qp_attr.ah_attr.grh.hop_limit = 1;
  }

  rtr.exp_attr_mask |= IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                       IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  return rtr;
}

Connection::RtsState ReliableConnection::rts(dest_info_t* dest) {
  RtsState rts{};

  rts.exp_qp_attr.sq_psn = dest->psn;

  rts.exp_attr_mask |= IBV_QP_SQ_PSN;

  return rts;
}

ibv_qp* ReliableConnection::create_qp_0(ibv_context* context,
                                        ibv_qp_init_attr_ex* qp_attr) {
  return ibv_create_qp_ex(context, qp_attr);
}

void ReliableConnection::create_qps_1() { }

void ReliableConnection::create_qps_2(int port, int my_rank,
                                        ibv_port_attr* ib_port_att) { }

void ReliableConnection::create_qps_3(int port, ibv_qp* qp, int offset,
                                        ibv_port_attr* ib_port_att) {
  init_qp_status(qp, port);

  all_qp[offset].lid = ib_port_att->lid;
  all_qp[offset].qpn = qp->qp_num;
  all_qp[offset].psn = 0;
  union ibv_gid gid;
  ibv_query_gid(ib_state->context, port, 0, &gid);
  all_qp[offset].gid = gid;
}

void ReliableConnection::get_remote_conn(int* remote_conn) {
  *remote_conn = backend->num_pes;
}

void ReliableConnection::allocate_dynamic_members(int num_blocks) {
  all_qp.resize(backend->num_pes * num_blocks);
}

void ReliableConnection::free_dynamic_members() {
}

void ReliableConnection::initialize_1(int port, int num_blocks) {
  MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t) * num_blocks, MPI_CHAR,
               all_qp.data(), sizeof(dest_info_t) * num_blocks, MPI_CHAR,
               backend->thread_comm);

  for (int i = 0; i < qps.size(); i++) {
    change_status_rtr(qps[i], &all_qp[i], port);
  }

  MPI_Barrier(backend->thread_comm);

  for (int i = 0; i < qps.size(); i++) {
    change_status_rts(qps[i], &all_qp[i]);
  }
}

void ReliableConnection::initialize_rkey_handle(uint32_t** heap_rkey_handle,
                                                  ibv_mr* mr) {
  CHECK_HIP(
      hipHostMalloc(heap_rkey_handle, sizeof(uint32_t) * backend->num_pes));
  (*heap_rkey_handle)[backend->my_pe] = mr->rkey;
}

void ReliableConnection::free_rkey_handle(uint32_t* heap_rkey_handle) {
  CHECK_HIP(hipHostFree(heap_rkey_handle));
}

Connection::QPInitAttr ReliableConnection::qpattr(ibv_qp_cap cap) {
  QPInitAttr qpattr(cap);
  qpattr.attr.qp_type = IBV_QPT_RC;
  return qpattr;
}

void ReliableConnection::post_dv_rc_wqe(int remote_conn) {
  mlx5_wqe_ctrl_seg* ctrl;
  mlx5_wqe_raddr_seg* rdma;
  mlx5_wqe_data_seg* data;

  for (int i = 0; i < remote_conn; i++) {
    int num_blocks = backend->num_blocks_;
    for (int j = 0; j < num_blocks; j++) {
      int qp_index = i * num_blocks + j;
      uint64_t* ptr = get_address_sq(qp_index);

      const uint16_t nb_post = 1;  // 4 * sq_size;
      for (uint16_t index = 0; index < nb_post; index++) {
        uint8_t op_mod = 0;
        uint8_t op_code = 8;
        uint32_t qp_num = qps[qp_index]->qp_num;
        uint8_t fm_ce_se = 0;
        uint8_t ds = 3;
        ctrl = reinterpret_cast<mlx5_wqe_ctrl_seg*>(ptr);
        mlx5dv_set_ctrl_seg(ctrl, index, op_code, op_mod, qp_num, fm_ce_se, ds,
                            0, 0);
        ptr = ptr + 2;

        rdma = reinterpret_cast<mlx5_wqe_raddr_seg*>(ptr);
        const auto& heap_bases = backend->heap.get_heap_bases();
        auto temp = heap_bases[(backend->my_pe + 1) % 2];
        uint64_t r_address = reinterpret_cast<uint64_t>(temp);
        uint32_t rkey = backend->networkImpl.heap_rkey[i];
        set_rdma_seg(rdma, r_address, rkey);
        ptr = ptr + 2;

        data = reinterpret_cast<mlx5_wqe_data_seg*>(ptr);
        uint32_t lkey = backend->networkImpl.heap_mr->lkey;
        temp = heap_bases[backend->my_pe];
        uint64_t address = reinterpret_cast<uint64_t>(temp);
        mlx5dv_set_data_seg(data, 1, lkey, address);
        ptr = ptr + 4;
      }
    }
  }
}

// TODO(bpotter): remove redundancies with the other derived class
void ReliableConnection::post_wqes() {
  int remote_conn;
  get_remote_conn(&remote_conn);
  post_dv_rc_wqe(remote_conn);
}

void ReliableConnection::initialize_wr_fields(ibv_send_wr* wr, ibv_ah* ah,
                                              int dc_key) {}

int ReliableConnection::get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx) {
  return pe_idx * num_qps + wg_idx;
}

}  // namespace rocshmem
