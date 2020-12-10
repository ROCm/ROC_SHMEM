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

#include "connection.hpp"

#include <vector>

#include "mpi.h"
#include "util.hpp"
#include "queue_pair.hpp"
#include "backend.hpp"

int Connection::use_gpu_mem = 0;

Connection::Connection(GPUIBBackend *b, int k)
  : backend(b), key_offset(k)
{
    char *value = nullptr;

    if ((value = getenv("ROC_SHMEM_USE_IB_HCA"))) {
        requested_dev = value;
    }

    if ((value = getenv("ROC_SHMEM_SQ_SIZE"))) {
        sq_size = atoi(value);
    }

    if ((value = getenv("ROC_SHMEM_USE_CQ_GPU_MEM")) != nullptr) {
        cq_use_gpu_mem = atoi(value);
    }

    if ((value = getenv("ROC_SHMEM_USE_SQ_GPU_MEM")) != nullptr) {
        sq_use_gpu_mem = atoi(value);
    }
}

Connection::~Connection()
{
    delete ib_state;
}

Status
Connection::reg_mr(void *ptr, size_t size, ibv_mr **mr)
{
    *mr = ibv_reg_mr(ib_state->pd, ptr, size,
                    IBV_EXP_ACCESS_LOCAL_WRITE |
                    IBV_EXP_ACCESS_REMOTE_WRITE |
                    IBV_EXP_ACCESS_REMOTE_READ |
                    IBV_EXP_ACCESS_REMOTE_ATOMIC);

    if (*mr == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }
    return Status::ROC_SHMEM_SUCCESS;
}

unsigned
Connection::total_number_connections()
{
    int connections;
    get_remote_conn(connections);
    return backend->num_wg * connections;
}

Status
Connection::initialize(int num_wg)
{
    Status status;

    status = allocate_dynamic_members(num_wg);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return status;
    }

    int ib_devices {0};
    dev_list = ibv_get_device_list(&ib_devices);
    if (dev_list == nullptr){
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    struct ibv_device *ib_dev = dev_list[0];
    if (requested_dev != nullptr) {
        for (int i = 0; i < ib_devices; i++) {
            const char *select_dev = ibv_get_device_name(dev_list[i]);
            if (strstr(select_dev, requested_dev) != nullptr) {
                ib_dev = dev_list[i];
                break;
            }
        }
    }

    uint8_t port = 1;
    status = ib_init(ib_dev, port);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return status;
    }

    int rtn_id = 0;

    int ib_fork_err = ibv_fork_init();
    if (ib_fork_err != 0)
        printf("error: ibv)fork_init  failed \n");

    sq_post_dv = static_cast<sq_post_dv_t*>(
        malloc(sizeof(sq_post_dv_t) * total_number_connections()));

    if (sq_post_dv == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    status = create_qps(port, rtn_id, backend->my_pe,
                        &ib_state->portinfo);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return status;
    }

    status = initialize_1(port, num_wg);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return status;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    free_dynamic_members();

    return Status::ROC_SHMEM_SUCCESS;
}

Status
Connection::finalize()
{
    ibv_free_device_list(dev_list);

    int ret = ibv_dereg_mr(backend->heap_mr);
    if (ret) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    ibv_dereg_mr(backend->hdp_mr);
    ibv_dereg_mr(backend->mr);

    MPI_Finalize();

    return Status::ROC_SHMEM_SUCCESS;
}

Status
Connection::ib_init(struct ibv_device *ib_dev,
                    uint8_t port)
{
    ib_state = new ib_state_t;
    if (!ib_state) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    ib_state->context = ibv_open_device(ib_dev);
    if (!ib_state->context) {
        delete ib_state;
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    ib_state->pd = ibv_alloc_pd(ib_state->context);
    if (!ib_state->pd) {
        delete ib_state;
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    ibv_query_port(ib_state->context, port, &ib_state->portinfo);

    return Status::ROC_SHMEM_SUCCESS;
}

template <typename StateType>
Status
Connection::try_to_modify_qp(ibv_qp *qp, StateType state)
{
    if (ibv_exp_modify_qp(qp, &state.exp_qp_attr, state.exp_attr_mask))
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
Connection::init_qp_status(ibv_qp *qp, uint8_t port)
{
    return try_to_modify_qp<InitQPState>(qp, initqp(port));
}

/**
 * rtr stands for 'ready to receive'
 */
Status
Connection::change_status_rtr(ibv_qp *qp, dest_info_t *dest, uint8_t port)
{
    return try_to_modify_qp<RtrState>(qp, rtr(dest, port));
}

/**
 * rts stands for 'ready to send'
 */
Status
Connection::change_status_rts(ibv_qp *qp, dest_info_t *dest)
{
    return try_to_modify_qp<RtsState>(qp, rts(dest));
}

Status
Connection::create_qps(uint8_t port, int rtn_id,
                       int my_rank, ibv_port_attr *ib_port_att)
{
    Status status;
    status = create_qps_1();
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    ibv_qp_cap cap {};
    cap.max_send_wr = sq_size;
    cap.max_send_sge = 1;
    cap.max_inline_data = 4;

    QPInitAttr qp_init_attr = qpattr(cap);

    size_t qp_size = total_number_connections();
    cqs.resize(qp_size);
    qps.resize(qp_size);

    for (auto &entry : cqs) {
        entry = create_cq(ib_state->context,
                          qp_init_attr.attr.cap.max_send_wr,
                          nullptr,
                          nullptr,
                          0,
                          rtn_id);
        if (!entry) {
            return Status::ROC_SHMEM_UNKNOWN_ERROR;
        }
    }

    status = create_qps_2(port, my_rank, ib_port_att);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    for (int i = 0; i < qps.size(); i++) {
            qps[i] = create_qp(ib_state->pd,
                               ib_state->context,
                               &qp_init_attr.attr,
                               cqs[i],
                               rtn_id);
        if (!qps[i]) {
            return Status::ROC_SHMEM_UNKNOWN_ERROR;
        }

        status = create_qps_3(port, qps[i], i, ib_port_att);
        if (status != Status::ROC_SHMEM_SUCCESS) {
            return Status::ROC_SHMEM_UNKNOWN_ERROR;
        }
    }
    return Status::ROC_SHMEM_SUCCESS;
}

Status
Connection::init_mpi_once()
{
    static std::mutex init_mutex;
    const std::lock_guard<std::mutex> lock(init_mutex);

    int init_done = 0;
    if (MPI_Initialized(&init_done) == MPI_SUCCESS){
       if (init_done) return Status::ROC_SHMEM_SUCCESS;
    }

    if (MPI_Init(nullptr, nullptr) != MPI_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    return Status::ROC_SHMEM_SUCCESS;
}

Status
Connection::initialize_gpu_policy(ConnectionImpl **conn,
                                  uint32_t *heap_rkey)
{
    CHECK_HIP(hipMalloc((void **) conn, sizeof(ConnectionImpl)));
    new (*conn) ConnectionImpl(this, heap_rkey);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
Connection::post_send(ibv_qp *qp, ibv_exp_send_wr *wr,
                      ibv_exp_send_wr **bad_wr)
{
    assert(qp);
    assert(wr);

    if (ibv_exp_post_send(qp, wr, bad_wr))
        return Status::ROC_SHMEM_UNKNOWN_ERROR;

    return Status::ROC_SHMEM_SUCCESS;
}

Status
Connection::cpu_post_wqe(ibv_qp *qp, void* addr, uint32_t lkey,
                         void* remote_addr, uint32_t rkey, size_t size,
                         ibv_ah *ah, int dc_key)
{
    counter_wqe++;
    ibv_sge list;
    list.addr = (uintptr_t) addr;
    list.length = size;
    list.lkey = lkey;

    ibv_exp_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t) counter_wqe;
    wr.sg_list = &list;
    wr.num_sge = 1;
    wr.exp_opcode = IBV_EXP_WR_RDMA_WRITE;
    wr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    wr.wr.rdma.remote_addr  = (int64_t) remote_addr;
    wr.wr.rdma.rkey = rkey;

    initialize_wr_fields(wr, ah, dc_key);

    ibv_exp_send_wr *bad_ewr;
    return post_send(qp, &wr, &bad_ewr);
}

ibv_exp_peer_buf *
Connection::buf_alloc(ibv_exp_peer_buf_alloc_attr *attr)
{
    assert(attr);
    ibv_exp_peer_buf * peer_buf =
        static_cast<ibv_exp_peer_buf*>(malloc(sizeof(ibv_exp_peer_buf)));

    if (peer_buf == nullptr) {
        printf("error, could not allocate memory \n");
        return nullptr;
    }

    peer_buf->comp_mask = 0;
    peer_buf->length = attr->length;

    if (use_gpu_mem) {
        void * dev_ptr;
        CHECK_HIP(hipSetDevice( attr->peer_id));
        CHECK_HIP(hipExtMallocWithFlags((void**)&dev_ptr, attr->length,
                                        hipDeviceMallocFinegrained));
        peer_buf->addr = dev_ptr;
    } else {
        free(peer_buf);
        return nullptr;
    }
    return peer_buf;
}

int
Connection::buf_release(ibv_exp_peer_buf *pb)
{
    assert(pb);
    free(pb->addr);
    pb->addr = nullptr;
    return 0;
}

uint64_t
Connection::register_va(void *start, size_t length, uint64_t rtn_id,
                        ibv_exp_peer_buf *pb)
{
    CHECK_HIP(hipSetDevice(rtn_id));

    void * gpu_ptr = nullptr;
    rocm_memory_lock_to_fine_grain(start, length, &gpu_ptr, rtn_id);

    return (uint64_t) (start);
}

int
Connection::unregister_va(uint64_t target_id, uint64_t rtn_id)
{
    CHECK_HIP(hipSetDevice(rtn_id));
    CHECK_HIP(hipHostUnregister ( (void*) target_id ));
    return 0;
}

void
Connection::init_peer_attr(ibv_exp_peer_direct_attr *attr1, int rtn_id)
{
    // TODO: need to cache this for better perf
    attr1->peer_id = rtn_id;
    attr1->buf_alloc = Connection::buf_alloc;
    attr1->buf_release = Connection::buf_release;
    attr1->register_va = Connection::register_va;
    attr1->unregister_va = Connection::unregister_va;

    attr1->caps = (IBV_EXP_PEER_OP_STORE_DWORD_CAP    |
                   IBV_EXP_PEER_OP_STORE_QWORD_CAP    |
                   IBV_EXP_PEER_OP_FENCE_CAP          |
                   IBV_EXP_PEER_OP_POLL_AND_DWORD_CAP |
                   IBV_EXP_PEER_OP_POLL_GEQ_DWORD_CAP);

    attr1->peer_dma_op_map_len = RTN_MAX_INLINE_SIZE;
    attr1->comp_mask = IBV_EXP_PEER_DIRECT_VERSION;
    attr1->version = 1; // EXP verbs requires to be set to 1
}

ibv_cq *
Connection::create_cq(ibv_context *context, int cqe,
                      void *cq_context, ibv_comp_channel *channel,
                      int comp_vector, int rtn_id)
{
    use_gpu_mem = cq_use_gpu_mem;
    ibv_exp_peer_direct_attr* peer_attr = &peers_attr[rtn_id];

    init_peer_attr(peer_attr,  rtn_id);

    ibv_exp_cq_init_attr attr;
    memset(&attr, 0, sizeof(ibv_exp_cq_init_attr));
    attr.comp_mask = IBV_EXP_CQ_INIT_ATTR_PEER_DIRECT;
    attr.flags = 0; // see ibv_exp_cq_create_flags
    attr.res_domain = nullptr;
    attr.peer_direct_attrs = peer_attr;

    ibv_cq *cq = ibv_exp_create_cq(context, cqe, cq_context, channel,
                                   comp_vector, &attr);
    if (!cq) {
        printf("error in ibv_exp_create_cq, %d  %s\n", errno, strerror(errno));
        return nullptr;
    }

    return cq;
}

Status
Connection::init_gpu_qp_from_connection(QueuePair &gpu_qp, int conn_num)
{
    int rtn_id = 0;
    use_gpu_mem = cq_use_gpu_mem;

    mlx5dv_cq cq_out;
    mlx5dv_obj mlx_obj;
    mlx_obj.cq.in = cqs[conn_num];
    mlx_obj.cq.out = &cq_out;

    mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_CQ);
    gpu_qp.cq_log_size = log2(cq_out.cqe_cnt);
    gpu_qp.cq_size = cq_out.cqe_cnt;

    void *gpu_ptr = nullptr;
    if (use_gpu_mem) {
        gpu_qp.current_cq_q = (mlx5_cqe64 *) cq_out.buf;
    } else {
        rocm_memory_lock_to_fine_grain((void*) cq_out.buf,
                                       cq_out.cqe_cnt * 64, &gpu_ptr, 0);
        gpu_qp.current_cq_q = (mlx5_cqe64 *) gpu_ptr;
    }

    rocm_memory_lock_to_fine_grain((void*) cq_out.dbrec, 64, &gpu_ptr, 0);
    gpu_qp.dbrec_cq = (volatile uint32_t*) gpu_ptr;

    use_gpu_mem = sq_use_gpu_mem;

    mlx5dv_qp qp_out;
    mlx_obj.qp.in = qps[conn_num];
    mlx_obj.qp.out = &qp_out;

    mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);

    gpu_qp.max_nwqe = (qp_out.sq.wqe_cnt);

    volatile uint32_t *dbrec_send = qp_out.dbrec + 1;

    if (use_gpu_mem) {
        gpu_qp.current_sq = (uint64_t *) qp_out.sq.buf;
        gpu_qp.dbrec_send = (volatile uint32_t*) dbrec_send;

    } else {
        rocm_memory_lock_to_fine_grain((void*) qp_out.sq.buf,
            qp_out.sq.wqe_cnt *64, &gpu_ptr, rtn_id);

        gpu_qp.current_sq = (uint64_t *) gpu_ptr;

        rocm_memory_lock_to_fine_grain((void*) dbrec_send, 32, &gpu_ptr,
                                       rtn_id);
        gpu_qp.dbrec_send = (volatile uint32_t*) gpu_ptr;
    }

    gpu_qp.threadImpl.setDBval(*((uint64_t *) qp_out.sq.buf));

    rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size,
                                   &gpu_ptr, rtn_id);

    gpu_qp.db = (uint64_t*) gpu_ptr;

    memcpy(sq_post_dv[conn_num].segments, qp_out.sq.buf, 64);

    sq_post_dv[conn_num].wqe_idx = 0;
    sq_post_dv[conn_num].current_sq = 0;

    uint32_t ctrl_qp_sq = ((uint32_t*)(sq_post_dv[conn_num].segments))[1];
    /*
     * Keep the BE-byte order representation of the qp_id that was populated
     * during the initial round of CPU wqe posting.  Remove the DS byte that
     * will be calculated dynamically by the GPU.
     */
    gpu_qp.ctrl_qp_sq = ctrl_qp_sq & 0xFFFFFF;
    gpu_qp.ctrl_sig = ((uint64_t*)(sq_post_dv[conn_num].segments))[1];
    gpu_qp.rkey = ((uint32_t*)(sq_post_dv[conn_num].segments))[6 + key_offset];
    gpu_qp.lkey = ((uint32_t*)(sq_post_dv[conn_num].segments))[9 + key_offset];

    return Status::ROC_SHMEM_SUCCESS;
}

ibv_qp *
Connection::create_qp(ibv_pd *pd, ibv_context *context,
                      ibv_exp_qp_init_attr *qp_attr, ibv_cq * cq,
                      int rtn_id)
{
    use_gpu_mem = sq_use_gpu_mem;

    int ret = 0;
    ibv_qp *qp = nullptr;
    ibv_cq *tx_cq = nullptr;
    ibv_exp_peer_direct_attr *peer_attr =  &peers_attr[rtn_id];

    assert(pd);
    assert(context);
    assert(qp_attr);

    init_peer_attr(peer_attr, rtn_id);

    qp_attr->send_cq = cq;
    qp_attr->recv_cq = cq;
    qp_attr->pd = pd;
    qp_attr->comp_mask |= IBV_EXP_QP_INIT_ATTR_PD;
    qp_attr->comp_mask |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
    qp_attr->exp_create_flags |= IBV_EXP_QP_CREATE_IGNORE_SQ_OVERFLOW ;

    qp_attr->comp_mask |= IBV_EXP_QP_INIT_ATTR_PEER_DIRECT;
    qp_attr->peer_direct_attrs = peer_attr;

    qp = ibv_exp_create_qp(context, qp_attr);

    if (!qp) {
        printf("error ibv_exp_create_qp failed %d \n", errno);
        ibv_destroy_qp(qp);
        ibv_destroy_cq(tx_cq);
    }

    return qp;
}
