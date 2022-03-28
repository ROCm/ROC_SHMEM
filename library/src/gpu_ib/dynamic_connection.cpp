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

#include "dynamic_connection.hpp"
#include "backend_ib.hpp"

#include "mpi.h"  // NOLINT(build/include_subdir)

namespace rocshmem {

DynamicConnection::DynamicConnection(GPUIBBackend* b)
    : Connection(b, 4) {
    char* value = nullptr;

    if ((value = getenv("ROC_SHMEM_NUM_DCIs"))) {
        num_dcis = atoi(value);
    }

    if ((value = getenv("ROC_SHMEM_NUM_DCT"))) {
        num_dct = atoi(value);
    }
}

DynamicConnection::~DynamicConnection() {
    CHECK_HIP(hipFree(vec_lids));
    CHECK_HIP(hipFree(vec_dct_num));
}

ibv_qp_init_attr_ex
DynamicConnection::dct_qp_init_attr(ibv_cq* cq,
                                 ibv_srq* srq,
                                 uint8_t port) const {
    ibv_qp_init_attr_ex attr {};

    attr.comp_mask = IBV_QP_INIT_ATTR_PD;
    attr.pd = ib_state->pd;
    attr.recv_cq = cq;
    attr.send_cq = cq;
    attr.srq = srq;
    attr.qp_type = IBV_QPT_DRIVER;

    return attr;
}

mlx5dv_qp_init_attr
DynamicConnection::dct_dv_init_attr() {
    mlx5dv_qp_init_attr dv_attr {};
    dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    dv_attr.dc_init_attr.dct_access_key = DC_IB_KEY;

    return dv_attr;
}

Connection::InitQPState
DynamicConnection::initqp(uint8_t port) {
    InitQPState initqp {};

    initqp.exp_qp_attr.port_num = port;
    initqp.exp_qp_attr.pkey_index = 0;
    initqp.exp_qp_attr.qp_access_flags = 0;

    return initqp;
}

Connection::RtrState
DynamicConnection::rtr(dest_info_t* dest,
                       uint8_t port) {
    RtrState rtr {};

    rtr.exp_qp_attr.ah_attr.is_global = 1;
    rtr.exp_qp_attr.ah_attr.port_num = port;

    rtr.exp_qp_attr.max_dest_rd_atomic = 0;
    rtr.exp_qp_attr.min_rnr_timer = 0;

    return rtr;
}

Connection::RtsState
DynamicConnection::rts(dest_info_t* dest) {
    RtsState rts {};
    rts.exp_attr_mask |=IBV_QP_SQ_PSN;
    return rts;
}

Status
DynamicConnection::connect_dci(ibv_qp* qp, uint8_t port) {
    Status status;
    status = init_qp_status(qp, port);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    status = change_status_rtr(qp, nullptr, port);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    status = change_status_rts(qp, nullptr);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }
    return Status::ROC_SHMEM_SUCCESS;
}

/*
 * create a DCT and get is to ready state
 */
Status
DynamicConnection::create_dct(int32_t* dct_num,
                              ibv_cq* cq,
                              ibv_srq* srq,
                              uint8_t port) {
    Status status;

    auto init_attr = dct_qp_init_attr(cq, srq, port);
    auto dv_attr = dct_dv_init_attr();
    auto dct = mlx5dv_create_qp(ib_state->context, &init_attr, &dv_attr);

    if (dct == nullptr) {
        printf("Failed to create dct \n");
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    ibv_qp_attr qp_attr {};
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.port_num = port;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_LOCAL_WRITE |
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_ATOMIC;

    int attr_mask =  IBV_QP_STATE |
                     IBV_QP_PKEY_INDEX |
                     IBV_QP_PORT |
                     IBV_QP_ACCESS_FLAGS;

    int ret = ibv_modify_qp(dct, &qp_attr, attr_mask);
    if (ret) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.min_rnr_timer = 7;
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.hop_limit = 1;
    qp_attr.ah_attr.grh.traffic_class = 0;
    qp_attr.ah_attr.grh.sgid_index = 0;
    qp_attr.ah_attr.port_num = port;

    attr_mask = IBV_QP_STATE |
                IBV_QP_MIN_RNR_TIMER |
                IBV_QP_AV |
                IBV_QP_PATH_MTU;

    ret = ibv_modify_qp(dct, &qp_attr, attr_mask);
    if (ret) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    *dct_num = dct->qp_num;
    return Status::ROC_SHMEM_SUCCESS;;
}

/*
 * @brief create a qp (DCI qp) using DEVX
 */
ibv_qp*
DynamicConnection::create_qp_0(ibv_context* context,
                               ibv_qp_init_attr_ex* qp_attr) {
    ibv_qp* qp;
    qp_attr->qp_type = IBV_QPT_DRIVER;

    mlx5dv_qp_init_attr dv_attr {};
    dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;
    dv_attr.dc_init_attr.dct_access_key = DC_IB_KEY;

    qp = mlx5dv_create_qp(context, qp_attr, &dv_attr);

    return qp;
}

Status
DynamicConnection::create_qps_1() {
    ibv_srq_init_attr srq_init_attr {};
    srq_init_attr.attr.max_wr = 1;
    srq_init_attr.attr.max_sge = 1;

    srq = ibv_create_srq(ib_state->pd, &srq_init_attr);
    if (!srq) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    dct_cq = ibv_create_cq(ib_state->context, 100, nullptr, nullptr, 0);
    if (!dct_cq) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    return Status::ROC_SHMEM_SUCCESS;
}

Status
DynamicConnection::create_qps_2(int port, int my_rank,
                                ibv_port_attr* ib_port_att) {
    for (int i = 0; i < num_dct; i++) {
        int32_t dct_num;
        Status status;
        status = create_dct(&dct_num, dct_cq, srq, port);
        if (status != Status::ROC_SHMEM_SUCCESS) {
            return Status::ROC_SHMEM_UNKNOWN_ERROR;
        }
        dct_num = htobe32(dct_num);
        dcts_num[my_rank * num_dct + i] = dct_num;
    }
    lids[my_rank] = htobe16(ib_port_att->lid);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
DynamicConnection::create_qps_3(int port, ibv_qp* qp, int offset,
                                ibv_port_attr* ib_port_att) {
    return connect_dci(qp, port);
}

Status
DynamicConnection::get_remote_conn(int* remote_conn) {
    *remote_conn = num_dcis;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
DynamicConnection::allocate_dynamic_members(int num_wg) {
    size_t num_pes_size_bytes = sizeof(uint16_t) * backend->num_pes;
    lids = reinterpret_cast<uint16_t*>(malloc(num_pes_size_bytes));
    if (lids == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    size_t num_dcts = num_dct * backend->num_pes;
    size_t num_dcts_size_bytes = sizeof(uint32_t) * num_dcts;
    dcts_num = reinterpret_cast<uint32_t*>(malloc(num_dcts_size_bytes));
    if (dcts_num == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    return Status::ROC_SHMEM_SUCCESS;
}

/*
 * get the wqe_av information from tyhe ibv_ah
 * rely on DEVX to extract the AV. We use the AV to create
 * the DC segment
 */
Status
DynamicConnection::dc_get_av(ibv_ah* ah,
                             mlx5_wqe_av* mlx5_av) {
    mlx5dv_obj dv;
    mlx5dv_ah dah;

    dv.ah.in = ah;
    dv.ah.out = &dah;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_AH);

    memcpy(mlx5_av, dah.av, sizeof(mlx5_wqe_av));
    return Status::ROC_SHMEM_SUCCESS;
}

Status
DynamicConnection::free_dynamic_members() {
    free(lids);
    free(dcts_num);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
DynamicConnection::initialize_1(int port,
                                int num_wg) {
    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(int32_t) * num_dct,
                  MPI_CHAR,
                  dcts_num,
                  sizeof(int32_t) * num_dct,
                  MPI_CHAR,
                  backend->thread_comm);

    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(int16_t),
                  MPI_CHAR,
                  lids,
                  sizeof(int16_t),
                  MPI_CHAR,
                  backend->thread_comm);

    hipStream_t stream;
    CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&vec_dct_num),
                        sizeof(int32_t) * num_dct * backend->num_pes));

    CHECK_HIP(hipMemcpyAsync(vec_dct_num,
                             dcts_num,
                             sizeof(int32_t) * num_dct * backend->num_pes,
                             hipMemcpyHostToDevice,
                             stream));

    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&vec_lids),
                        sizeof(int16_t) * backend->num_pes));

    CHECK_HIP(hipMemcpyAsync(vec_lids,
                             lids,
                             sizeof(int16_t) * backend->num_pes,
                             hipMemcpyHostToDevice,
                             stream));


    struct ibv_ah_attr ah_attr;
    memset(&ah_attr, 0, sizeof(ah_attr));

    ah_attr.is_global = 1;
    ah_attr.dlid = ib_state->portinfo.lid;
    ah_attr.sl = 1;
    ah_attr.src_path_bits = 0;
    ah_attr.port_num = port;

    ah = ibv_create_ah(ib_state->pd, &ah_attr);
    if (ah == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    dc_get_av(ah, &mlx5_av);

    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipStreamDestroy(stream));
    return Status::ROC_SHMEM_SUCCESS;
}

Status
DynamicConnection::initialize_rkey_handle(uint32_t** heap_rkey_handle,
                                          ibv_mr* mr) {
    CHECK_HIP(hipMalloc(heap_rkey_handle,
                        sizeof(uint32_t) * backend->num_pes));
    (*heap_rkey_handle)[backend->my_pe] = htobe32(mr->rkey);
    return Status::ROC_SHMEM_SUCCESS;
}

void
DynamicConnection::free_rkey_handle(uint32_t* heap_rkey_handle) {
     CHECK_HIP(hipFree(heap_rkey_handle));
}

Connection::QPInitAttr
DynamicConnection::qpattr(ibv_qp_cap cap) {
    QPInitAttr qpattr(cap);
    return qpattr;
}

/*
 * Create and write the DC segment to SQ.
 * We get all the info needed from the mlx5_wqe_av that we extract from ibv_ah.
 */
void
DynamicConnection::set_dgram_seg(mlx5_wqe_datagram_seg* dc_seg,
                                 uint64_t dc_key,
                                 uint32_t dct_num,
                                 uint8_t ext,
                                 mlx5_wqe_av* mlx5_av) {
    dc_seg->av.key.dc_key = htobe64(dc_key);
    dc_seg->av.dqp_dct = htobe32(((uint32_t) ext << 31 | dct_num));
    dc_seg->av.stat_rate_sl = mlx5_av->stat_rate_sl;
    dc_seg->av.fl_mlid = mlx5_av->fl_mlid;
    dc_seg->av.rlid = mlx5_av->rlid;
}

/*
 * create a DC wqe and post it to the SQ
 * we rely on mlx5dv functions to ceate the ctrl and data
 * segments but we use our own function to write teh DC and rdma segments
 */
void
DynamicConnection::post_dv_dc_wqe(int remote_conn) {
    mlx5_wqe_ctrl_seg* ctrl;
    mlx5_wqe_datagram_seg* dc_seg;
    mlx5_wqe_raddr_seg* rdma;
    mlx5_wqe_data_seg* data;

    for (int i = 0; i < remote_conn; i++) {
        uint64_t* ptr = get_address_sq(i);

        const uint32_t nb_post = 4 * sq_size;
        for (uint16_t index = 0; index < nb_post; index++) {
            uint8_t op_mod = 0;
            uint8_t op_code = 8;
            uint32_t qp_num = qps[i]->qp_num;
            uint8_t fm_ce_se = 0;
            uint8_t ds = 4;
            ctrl = reinterpret_cast<mlx5_wqe_ctrl_seg*>(ptr);
            mlx5dv_set_ctrl_seg(ctrl,
                                index,
                                op_code,
                                op_mod,
                                qp_num,
                                fm_ce_se,
                                ds,
                                0,
                                0);
            ptr = ptr + 2;

            uint32_t dct_num = dcts_num[i];
            uint8_t ext = 1;
            dc_seg = reinterpret_cast<mlx5_wqe_datagram_seg*>(ptr);
            set_dgram_seg(dc_seg,
                          (uint64_t)DC_IB_KEY,
                          dct_num,
                          ext,
                          &mlx5_av);
            ptr = ptr + 2;

            uint64_t address = 0;
            uint32_t rkey = 0;
            rdma = reinterpret_cast<mlx5_wqe_raddr_seg*>(ptr);
            set_rdma_seg(rdma, address, rkey);
            ptr = ptr + 2;

            uint32_t lkey = backend->networkImpl.heap_mr->lkey;
            data = reinterpret_cast<mlx5_wqe_data_seg*>(ptr);
            mlx5dv_set_data_seg(data, 1, lkey, 0);
            ptr = ptr + 2;
        }
    }
}

// TODO(bpotter): remove redundancies with the other derived class
void
DynamicConnection::post_wqes() {
    int remote_conn;
    get_remote_conn(&remote_conn);
    remote_conn *= backend->num_wg;
    post_dv_dc_wqe(remote_conn);
}

void
DynamicConnection::initialize_wr_fields(ibv_send_wr* wr,
                                        ibv_ah* ah,
                                        int dc_key) {
}

int
DynamicConnection::get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx) {
    return wg_idx;
}

}  // namespace rocshmem
