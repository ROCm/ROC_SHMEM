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

#include "dynamic_connection.hpp"
#include "backend.hpp"

#include "mpi.h"

DynamicConnection::DynamicConnection(GPUIBBackend *b)
    : Connection(b, 4)
{
    char *value = nullptr;

    if ((value = getenv("ROC_SHMEM_NUM_DCIs"))) {
        num_dcis = atoi(value);
    }

    if ((value = getenv("ROC_SHMEM_NUM_DCT"))) {
        num_dct = atoi(value);
    }
}

DynamicConnection::~DynamicConnection()
{
}

bool
DynamicConnection::transport_enabled()
{
    ibv_exp_device_attr dattr {};
    dattr.comp_mask = IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS |
                      IBV_EXP_DEVICE_DC_RD_REQ |
                      IBV_EXP_DEVICE_DC_RD_RES;

    if (ibv_exp_query_device(ib_state->context, &dattr)) {
        // Could not query device extended attributes.
        return false;
    }

    if (!(dattr.exp_device_cap_flags & IBV_EXP_DEVICE_DC_TRANSPORT)) {
        // Dynamic connection transport not enabled.
        return false;
    }

    return true;
}

bool
DynamicConnection::status_good(ibv_exp_dct *dct)
{
    if (!dct) {
        return false;
    }

    ibv_exp_dct_attr dcqattr {};
    if (ibv_exp_query_dct(dct, &dcqattr)) {
        return false;
    }

    if (dcqattr.dc_key != DC_IB_KEY) {
        return false;
    }

    if (dcqattr.state != IBV_EXP_DCT_STATE_ACTIVE) {
        return false;
    }

    return true;
}

ibv_exp_dct_init_attr
DynamicConnection::dct_init_attr(ibv_cq *cq,
                                 ibv_srq *srq,
                                 uint8_t port) const
{
    ibv_exp_dct_init_attr attr {};

    attr.pd = ib_state->pd;
    attr.cq = cq;
    attr.srq = srq;
    attr.dc_key = DC_IB_KEY;
    attr.port = port;
    attr.mtu = IBV_MTU_4096;
    attr.access_flags = IBV_EXP_ACCESS_REMOTE_WRITE |
                        IBV_EXP_ACCESS_LOCAL_WRITE |
                        IBV_EXP_ACCESS_REMOTE_READ |
                        IBV_EXP_ACCESS_REMOTE_ATOMIC;
    attr.min_rnr_timer = 7;
    attr.hop_limit = 1;
    attr.inline_size = 4;

    return attr;
}

Connection::InitQPState
DynamicConnection::initqp(uint8_t port)
{
    InitQPState initqp {};

    initqp.exp_qp_attr.dct_key = DC_IB_KEY;
    initqp.exp_qp_attr.port_num = port;

    initqp.exp_attr_mask |= IBV_EXP_QP_DC_KEY;

    return initqp;
}

Connection::RtrState
DynamicConnection::rtr(dest_info_t *dest, uint8_t port)
{
    RtrState rtr {};

    rtr.exp_qp_attr.dct_key = DC_IB_KEY;
    rtr.exp_qp_attr.ah_attr.port_num = port;

    return rtr;
}

Connection::RtsState
DynamicConnection::rts(dest_info_t *dest)
{
    return RtsState();
}

roc_shmem_status_t
DynamicConnection::connect_dci(ibv_qp *qp, uint8_t port)
{
    roc_shmem_status_t status;
    status = init_qp_status(qp, port);
    if (status != ROC_SHMEM_SUCCESS) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    status = change_status_rtr(qp, nullptr, port);
    if (status != ROC_SHMEM_SUCCESS) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    status = change_status_rts(qp, nullptr);
    if (status != ROC_SHMEM_SUCCESS) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
DynamicConnection::create_dct(int32_t &dct_num,
                              ibv_cq *cq,
                              ibv_srq *srq,
                              uint8_t port)
{
    if (!transport_enabled()) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    auto init_attr = dct_init_attr(cq, srq, port);
    auto dct = ibv_exp_create_dct(ib_state->context, &init_attr);

    if (!status_good(dct)) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    dct_num = dct->dct_num;

    return ROC_SHMEM_SUCCESS;;
}

roc_shmem_status_t
DynamicConnection::create_qps_1()
{
    ibv_srq_init_attr srq_init_attr {};
    srq_init_attr.attr.max_wr = 1;
    srq_init_attr.attr.max_sge = 1;

    srq = ibv_create_srq(ib_state->pd, &srq_init_attr);
    if (!srq) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    dct_cq = ibv_create_cq(ib_state->context, 100, nullptr, nullptr, 0);
    if (!dct_cq) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
DynamicConnection::create_qps_2(int port, int my_rank,
                                ibv_port_attr *ib_port_att)
{
    for (int i = 0; i < num_dct; i++) {
        int32_t dct_num;
        roc_shmem_status_t status;
        status = create_dct(dct_num, dct_cq, srq, port);
        if (status != ROC_SHMEM_SUCCESS) {
            return ROC_SHMEM_UNKNOWN_ERROR;
        }
        dct_num = htobe32(dct_num);
        dcts_num[my_rank * num_dct + i] = dct_num;
    }
    lids[my_rank] = htobe16(ib_port_att->lid);

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
DynamicConnection::create_qps_3(int port, ibv_qp *qp, int offset,
                                ibv_port_attr *ib_port_att)
{
    return connect_dci(qp, port);
}

roc_shmem_status_t
DynamicConnection::get_remote_conn(int &remote_conn)
{
    remote_conn = num_dcis;

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
DynamicConnection::allocate_dynamic_members(int num_wg)
{
    lids = (uint16_t*) malloc(sizeof(uint16_t) * backend->num_pes);
    if (lids == nullptr) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    size_t num_dcts = num_dct * backend->num_pes;
    dcts_num = (uint32_t*) malloc(sizeof(uint32_t) * num_dcts);
    if (dcts_num == nullptr) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
DynamicConnection::free_dynamic_members()
{
    free(lids);
    free(dcts_num);
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
DynamicConnection::initialize_1(int port,
                                int num_wg)
{
    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(int32_t) * num_dct,
                  MPI_CHAR,
                  dcts_num,
                  sizeof(int32_t) * num_dct,
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(int16_t),
                  MPI_CHAR,
                  lids,
                  sizeof(int16_t),
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    CHECK_HIP(hipMalloc((void**) &vec_dct_num,
                        sizeof(int32_t) * num_dct * backend->num_pes));

    CHECK_HIP(hipMemcpy(vec_dct_num,
                        dcts_num,
                        sizeof(int32_t) * num_dct * backend->num_pes,
                        hipMemcpyHostToDevice));

    CHECK_HIP(hipMalloc((void**) &vec_lids,
                        sizeof(int16_t) * backend->num_pes));

    CHECK_HIP(hipMemcpy(vec_lids,
                        lids,
                        sizeof(int16_t) *  backend->num_pes,
                        hipMemcpyHostToDevice));

    struct ibv_ah_attr ah_attr;
    memset(&ah_attr, 0, sizeof(ah_attr));

    ah_attr.is_global     = 0;
    ah_attr.dlid          = ib_state->portinfo.lid;
    ah_attr.sl            = 1;
    ah_attr.src_path_bits = 0;
    ah_attr.port_num      = port;

    ah = ibv_create_ah(ib_state->pd, &ah_attr);
    if (ah == nullptr) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
DynamicConnection::initialize_rkey_handle(uint32_t **heap_rkey_handle,
                                          ibv_mr *mr)
{
    CHECK_HIP(hipMalloc(heap_rkey_handle,
                        sizeof(uint32_t) * backend->num_pes));
    (*heap_rkey_handle)[backend->my_pe]  = htobe32(mr->rkey);
    return ROC_SHMEM_SUCCESS;
}

Connection::QPInitAttr
DynamicConnection::qpattr(ibv_qp_cap cap)
{
    QPInitAttr qpattr(cap);
    qpattr.attr.qp_type = IBV_EXP_QPT_DC_INI;
    return qpattr;
}

// TODO: remove redundancies with the other derived class
void
DynamicConnection::post_wqes()
{
    int remote_conn;
    get_remote_conn(remote_conn);

    int lkey = backend->lkey;

    for (auto qp : qps) {
        cpu_post_wqe(qp, nullptr, lkey, nullptr, 0, 10, ah, DC_IB_KEY);
        uint32_t nb_post = 4 * sq_size;
        for (int k = 0; k < nb_post - 1; k++) {
            cpu_post_wqe(qp, nullptr, lkey, nullptr, 0, 10, ah, DC_IB_KEY);
        }
    }
}

void
DynamicConnection::initialize_wr_fields(ibv_exp_send_wr &wr,
                                        ibv_ah *ah, int dc_key)
{
    wr.dc.ah = ah;
    wr.dc.dct_number = 0;
    wr.dc.dct_access_key = dc_key;
}

int
DynamicConnection::get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx)
{
    return wg_idx;
}
