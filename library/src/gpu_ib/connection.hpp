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

#ifndef LIBRARY_SRC_GPU_IB_CONNECTION_HPP_
#define LIBRARY_SRC_GPU_IB_CONNECTION_HPP_

#include <infiniband/verbs.h>

extern "C" {
#include <infiniband/mlx5dv.h>
}

#include <vector>

#include "include/roc_shmem.hpp"
#include "src/gpu_ib/connection_policy.hpp"

namespace rocshmem {

class GPUIBBackend;
class QueuePair;

class Connection {
 protected:
  typedef struct ib_state {
    struct ibv_context* context;
    struct ibv_pd* pd;
    struct ibv_mr* mr;
    struct ibv_port_attr portinfo;
  } ib_state_t;

  typedef struct dest_info {
    int lid;
    int qpn;
    int psn;
    union ibv_gid gid;
  } dest_info_t;

  typedef struct heap_info {
    void* base_heap;
    uint32_t rkey;
  } heap_info_t;

  struct sq_post_dv_t {
    uint64_t segments[16];
    uint32_t current_sq;
    uint16_t wqe_idx;
  };

  class State {
   public:
    ibv_qp_attr exp_qp_attr{};
    uint64_t exp_attr_mask{};
  };

  class InitQPState : public State {
   public:
    InitQPState() {
      exp_qp_attr.qp_state = IBV_QPS_INIT;
      exp_qp_attr.qp_access_flags =
          IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

      exp_attr_mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT;
    }
  };

  class RtrState : public State {
   public:
    RtrState() {
      exp_qp_attr.qp_state = IBV_QPS_RTR;
      exp_qp_attr.path_mtu = IBV_MTU_4096;
      exp_qp_attr.ah_attr.sl = 1;
      exp_qp_attr.max_dest_rd_atomic = 1;
      exp_qp_attr.min_rnr_timer = 12;

      exp_attr_mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU;
    }
  };

  class RtsState : public State {
   public:
    RtsState() {
      exp_qp_attr.qp_state = IBV_QPS_RTS;
      exp_qp_attr.timeout = 14;
      exp_qp_attr.retry_cnt = 7;
      exp_qp_attr.rnr_retry = 7;
      exp_qp_attr.max_rd_atomic = 1;

      exp_attr_mask = IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                      IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
    }
  };

  class QPInitAttr {
   public:
    explicit QPInitAttr(ibv_qp_cap cap) {
      attr.cap = cap;
      attr.sq_sig_all = 0;
    }
    ibv_qp_init_attr_ex attr{};
  };

 public:
  Connection(GPUIBBackend* backend, int key_offset);

  virtual ~Connection();

  void initialize(int num_block);

  void finalize();

  virtual void post_wqes() = 0;

  void reg_mr(void* ptr, size_t size, ibv_mr** mr, bool is_managed);

  virtual void get_remote_conn(int* remote_conn) = 0;

  unsigned total_number_connections();

  virtual void initialize_rkey_handle(uint32_t** heap_rkey_handle,
                                        ibv_mr* mr) = 0;

  virtual void free_rkey_handle(uint32_t* heap_rkey_handle) = 0;

  void initialize_gpu_policy(ConnectionImpl** conn, uint32_t* heap_rkey);

  /*
   * Populate a QueuePair for use on the GPU from the internal IB state.
   */
  void init_gpu_qp_from_connection(QueuePair* qp, int conn_num);

 protected:
  Connection() = default;

  virtual InitQPState initqp(uint8_t port) = 0;

  virtual RtrState rtr(dest_info_t* dest, uint8_t port) = 0;

  virtual RtsState rts(dest_info_t* dest) = 0;

  virtual QPInitAttr qpattr(ibv_qp_cap cap) = 0;

  void init_qp_status(ibv_qp* qp, uint8_t port);

  void change_status_rtr(ibv_qp* qp, dest_info_t* dest, uint8_t port);

  void change_status_rts(ibv_qp* qp, dest_info_t* dest);

  void create_qps(uint8_t port, int my_rank, ibv_port_attr* ib_port_att);

  template <typename T>
  void try_to_modify_qp(ibv_qp* qp, T state);

  virtual void create_qps_1() = 0;

  virtual void create_qps_2(int port, int my_rank,
                              ibv_port_attr* ib_port_att) = 0;

  virtual void create_qps_3(int port, ibv_qp* qp, int offset,
                              ibv_port_attr* ib_port_att) = 0;

  virtual ibv_qp* create_qp_0(ibv_context* context,
                              ibv_qp_init_attr_ex* qp_attr) = 0;

  virtual void allocate_dynamic_members(int num_block) = 0;

  virtual void free_dynamic_members() = 0;

  virtual void initialize_1(int port, int num_block) = 0;

  virtual void initialize_wr_fields(ibv_send_wr* wr, ibv_ah* ah,
                                    int dc_key) = 0;

  virtual int get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx) = 0;

  void set_sq_dv(int num_block, int wg_idx, int pe_idx);

  /*
   * ibv interface functions must be static.
   */
  static void* buf_alloc(ibv_pd* pd, void* pd_context, size_t size,
                         size_t alignment, uint64_t resource_type);

  static void buf_release(ibv_pd* pd, void* pd_context, void* ptr,
                          uint64_t resource_type);

  void init_parent_domain_attr(ibv_parent_domain_init_attr* attr);

  void set_rdma_seg(mlx5_wqe_raddr_seg* rdma, uint64_t address, uint32_t rkey);

  uint64_t* get_address_sq(int i);

  ibv_cq* create_cq(ibv_context* context, ibv_pd* pd, int cqe);

  ibv_qp* create_qp(ibv_pd* pd, ibv_context* context,
                    ibv_qp_init_attr_ex* qp_attr, ibv_cq* rcq);

  /*
   * TODO: Remove this eventually. Goal is to have backend delegate
   * connection stuff to this class, while this class knows nothing about
   * GPUs or backends.
   */
  GPUIBBackend* backend{nullptr};

  uint32_t sq_size{1024};

  ib_state_t* ib_state{nullptr};

  const int key_offset{0};

  sq_post_dv_t* sq_post_dv{nullptr};

  std::vector<ibv_cq*> cqs;

  std::vector<ibv_qp*> qps;

  uint64_t counter_wqe{0};

  static int use_gpu_mem;

  static int coherent_cq;

  int cq_use_gpu_mem{1};

  int sq_use_gpu_mem{1};

 private:
  void init_shmem_handle();

  void ib_init(ibv_device* ib_dev, uint8_t port);

  char* requested_dev{nullptr};

  ibv_device** dev_list{nullptr};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_CONNECTION_HPP_
