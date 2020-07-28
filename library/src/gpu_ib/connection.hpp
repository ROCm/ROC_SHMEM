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

#ifndef __CONNECTION_HPP__
#define __CONNECTION_HPP__

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>

extern "C"{
#include <infiniband/mlx5dv.h>
}

#include <roc_shmem.hpp>

#include "connection_policy.hpp"

class GPUIBBackend;
class QueuePair;

class Connection
{
  private:
    using Status = roc_shmem_status_t;

  protected:
    typedef struct ib_state {
        struct ibv_context *context;
        struct ibv_pd *pd;
        struct ibv_mr *mr;
        struct ibv_port_attr portinfo;
    } ib_state_t;

    typedef struct dest_info {
        int lid;
        int qpn;
        int psn;
    } dest_info_t;

    typedef struct heap_info {
        void *base_heap;
        uint32_t rkey;
    } heap_info_t;

    struct sq_post_dv_t {
        uint64_t segments[8];
        uint32_t current_sq;
        uint16_t wqe_idx;
    };

    class State
    {
      public:
        ibv_exp_qp_attr exp_qp_attr {};
        uint64_t exp_attr_mask {};
    };

    class InitQPState : public State
    {
      public:
        InitQPState()
        {
            exp_qp_attr.qp_state = IBV_QPS_INIT;

            exp_attr_mask = IBV_EXP_QP_STATE |
                            IBV_EXP_QP_PKEY_INDEX |
                            IBV_EXP_QP_PORT;
        }
    };

    class RtrState : public State
    {
      public:
        RtrState()
        {
            exp_qp_attr.qp_state = IBV_QPS_RTR;
            exp_qp_attr.path_mtu = IBV_MTU_4096;
            exp_qp_attr.ah_attr.sl = 1;
            exp_qp_attr.max_dest_rd_atomic = 1;
            exp_qp_attr.min_rnr_timer = 12;

            exp_attr_mask = IBV_EXP_QP_STATE |
                            IBV_EXP_QP_AV |
                            IBV_EXP_QP_PATH_MTU;
        }
    };

    class RtsState : public State
    {
      public:
        RtsState()
        {
            exp_qp_attr.qp_state = IBV_QPS_RTS;
            exp_qp_attr.timeout = 14;
            exp_qp_attr.retry_cnt = 7;
            exp_qp_attr.rnr_retry = 7;
            exp_qp_attr.max_rd_atomic = 1;

            exp_attr_mask = IBV_EXP_QP_STATE |
                            IBV_EXP_QP_TIMEOUT |
                            IBV_EXP_QP_RETRY_CNT |
                            IBV_EXP_QP_RNR_RETRY |
                            IBV_EXP_QP_MAX_QP_RD_ATOMIC;
        }
    };

    class QPInitAttr
    {
      public:
        explicit QPInitAttr(ibv_qp_cap cap)
        {
            attr.cap = cap;
            attr.sq_sig_all = 1;
        }
        ibv_exp_qp_init_attr attr {};
    };

  public:
    Connection(GPUIBBackend *backend, int key_offset);

    virtual ~Connection();

    Status initialize(int num_wg);

    Status finalize();

    virtual void post_wqes() = 0;

    Status reg_mr(void *ptr, size_t size, ibv_mr **mr);

    Status init_mpi_once();

    virtual roc_shmem_status_t get_remote_conn(int &remote_conn) = 0;

    unsigned total_number_connections();

    virtual roc_shmem_status_t
    initialize_rkey_handle(uint32_t **heap_rkey_handle, ibv_mr *mr) = 0;

    roc_shmem_status_t
    initialize_gpu_policy(ConnectionImpl **conn, uint32_t *heap_rkey);

    /*
     * Populate a QueuePair for use on the GPU from the internal IB state.
     */
    roc_shmem_status_t
    init_gpu_qp_from_connection(QueuePair &qp, int conn_num);

  protected:
    Connection() = default;

    virtual InitQPState initqp(uint8_t port) = 0;

    virtual RtrState rtr(dest_info_t *dest, uint8_t port) = 0;

    virtual RtsState rts(dest_info_t *dest) = 0;

    virtual QPInitAttr qpattr(ibv_qp_cap cap) = 0;

    Status init_qp_status(ibv_qp *qp, uint8_t port);

    Status change_status_rtr(ibv_qp *qp, dest_info_t *dest, uint8_t port);

    Status change_status_rts(ibv_qp *qp, dest_info_t *dest);

    Status create_qps(uint8_t port, int rtn_id,
                      int my_rank, ibv_port_attr *ib_port_att);

    template <typename T>
    Status
    try_to_modify_qp(ibv_qp *qp, T state);

    virtual roc_shmem_status_t
    create_qps_1() = 0;

    virtual roc_shmem_status_t
    create_qps_2(int port, int my_rank,
                 ibv_port_attr *ib_port_att) = 0;

    virtual roc_shmem_status_t
    create_qps_3(int port, ibv_qp *qp, int offset,
                 ibv_port_attr *ib_port_att) = 0;

    virtual roc_shmem_status_t allocate_dynamic_members(int num_wg) = 0;

    virtual roc_shmem_status_t free_dynamic_members() = 0;

    virtual roc_shmem_status_t
    initialize_1(int port,
                 int num_wg) = 0;

    virtual void
    initialize_wr_fields(ibv_exp_send_wr &wr,
                         ibv_ah *ah, int dc_key) = 0;

    virtual int
    get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx) = 0;

    roc_shmem_status_t
    set_sq_dv(int num_wgs, int wg_idx, int pe_idx);

    /*
     * ibv interface functions must be static.
     */
    static ibv_exp_peer_buf *
    buf_alloc(ibv_exp_peer_buf_alloc_attr *attr);

    static int buf_release(ibv_exp_peer_buf *pb);

    static uint64_t
    register_va(void *start, size_t length, uint64_t rtn_id,
                ibv_exp_peer_buf *pb);

    static int unregister_va(uint64_t target_id, uint64_t rtn_id);

    void init_peer_attr(ibv_exp_peer_direct_attr *attr1, int rtn_id);

    roc_shmem_status_t
    post_send(ibv_qp *qp, ibv_exp_send_wr *wr, ibv_exp_send_wr **bad_wr);

    roc_shmem_status_t
    cpu_post_wqe(ibv_qp *qp, void* addr, uint32_t lkey,
                 void* remote_addr, uint32_t rkey, size_t size,
                 ibv_ah *ah, int dc_key);

    ibv_cq *
    create_cq(ibv_context *context, int cqe,
              void *cq_context, ibv_comp_channel *channel,
              int comp_vector, int rtn_id);

    ibv_qp *
    create_qp(ibv_pd *pd, ibv_context *context,
              ibv_exp_qp_init_attr *qp_attr, ibv_cq * rcq,
              int rtn_id);

    /*
     * TODO: Remove this eventually.  Goal is to have backend delegate
     * connection stuff to this class, while this class knows nothing about
     * GPUs or backends.
     */
    GPUIBBackend *backend;

    uint32_t sq_size = 1024;

    const size_t RTN_MAX_INLINE_SIZE = 128;

    ib_state_t *ib_state = nullptr;

    ibv_exp_peer_direct_attr peers_attr[32];

    const int key_offset = 0;

    sq_post_dv_t* sq_post_dv = nullptr;

    std::vector<ibv_cq *> cqs;

    std::vector<ibv_qp *> qps;

    uint64_t counter_wqe = 0;

    static int use_gpu_mem;

    int cq_use_gpu_mem = 0;

    int sq_use_gpu_mem = 0;

  private:
    Status init_shmem_handle();

    Status ib_init(ibv_device *ib_dev, uint8_t port);

    char *requested_dev = nullptr;
};

#endif // __CONNECTION_HPP__
