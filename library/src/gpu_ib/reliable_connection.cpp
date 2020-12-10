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

#include "reliable_connection.hpp"
#include "backend.hpp"

#include "mpi.h"

ReliableConnection::ReliableConnection(GPUIBBackend *b)
    : Connection(b, 0)
{
}

ReliableConnection::~ReliableConnection()
{
}

Connection::InitQPState
ReliableConnection::initqp(uint8_t port)
{
    InitQPState init {};

    init.exp_qp_attr.qp_access_flags = IBV_EXP_ACCESS_REMOTE_WRITE |
                                       IBV_EXP_ACCESS_LOCAL_WRITE  |
                                       IBV_EXP_ACCESS_REMOTE_READ  |
                                       IBV_EXP_ACCESS_REMOTE_ATOMIC;
    init.exp_qp_attr.port_num = port;

    init.exp_attr_mask |= IBV_EXP_QP_ACCESS_FLAGS;

    return init;
}

Connection::RtrState
ReliableConnection::rtr(dest_info_t *dest, uint8_t port)
{
    RtrState rtr {};

    rtr.exp_qp_attr.dest_qp_num = dest->qpn;
    rtr.exp_qp_attr.rq_psn = dest->psn;
    rtr.exp_qp_attr.ah_attr.dlid = dest->lid;
    rtr.exp_qp_attr.ah_attr.port_num = port;

    rtr.exp_attr_mask |= IBV_EXP_QP_DEST_QPN           |
                         IBV_EXP_QP_RQ_PSN             |
                         IBV_EXP_QP_MAX_DEST_RD_ATOMIC |
                         IBV_EXP_QP_MIN_RNR_TIMER;

    return rtr;
}

Connection::RtsState
ReliableConnection::rts(dest_info_t *dest)
{
    RtsState rts {};

    rts.exp_qp_attr.sq_psn = dest->psn;

    rts.exp_attr_mask |= IBV_EXP_QP_SQ_PSN;

    return rts;
}

Status
ReliableConnection::create_qps_1()
{
    return Status::ROC_SHMEM_SUCCESS;
}

Status
ReliableConnection::create_qps_2(int port, int my_rank,
                                 ibv_port_attr *ib_port_att)
{
    return Status::ROC_SHMEM_SUCCESS;
}

Status
ReliableConnection::create_qps_3(int port, ibv_qp *qp, int offset,
                                 ibv_port_attr *ib_port_att)
{
    Status status = init_qp_status(qp, port);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    all_qp[offset].lid = ib_port_att->lid;
    all_qp[offset].qpn = qp->qp_num;
    all_qp[offset].psn = 0;

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ReliableConnection::get_remote_conn(int &remote_conn)
{
    remote_conn = backend->num_pes;

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ReliableConnection::allocate_dynamic_members(int num_wg)
{
    all_qp.resize(backend->num_pes * num_wg);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
ReliableConnection::free_dynamic_members()
{
    return Status::ROC_SHMEM_SUCCESS;
}

Status
ReliableConnection::initialize_1(int port, int num_wg)
{
    MPI_Alltoall(MPI_IN_PLACE,
                 sizeof(dest_info_t) * num_wg,
                 MPI_CHAR,
                 all_qp.data(),
                 sizeof(dest_info_t) * num_wg,
                 MPI_CHAR,
                 MPI_COMM_WORLD);

    Status status;

    for (int i = 0; i < qps.size(); i++) {
        status = change_status_rtr(qps[i], &all_qp[i], port);
        if (status != Status::ROC_SHMEM_SUCCESS) {
            return Status::ROC_SHMEM_UNKNOWN_ERROR;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < qps.size(); i++) {
        status = change_status_rts(qps[i], &all_qp[i]);
        if (status != Status::ROC_SHMEM_SUCCESS) {
            return Status::ROC_SHMEM_UNKNOWN_ERROR;
        }
    }
    return Status::ROC_SHMEM_SUCCESS;
}

Status
ReliableConnection::initialize_rkey_handle(uint32_t **heap_rkey_handle,
                                           ibv_mr *mr)
{
    CHECK_HIP(hipHostMalloc(heap_rkey_handle,
                            sizeof(uint32_t) * backend->num_pes));
    (*heap_rkey_handle)[backend->my_pe]  = mr->rkey;

    return Status::ROC_SHMEM_SUCCESS;
}

void
ReliableConnection::free_rkey_handle(uint32_t *heap_rkey_handle)
{
    CHECK_HIP(hipHostFree(heap_rkey_handle));
}

Connection::QPInitAttr
ReliableConnection::qpattr(ibv_qp_cap cap)
{
    QPInitAttr qpattr(cap);
    qpattr.attr.qp_type = IBV_QPT_RC;
    return qpattr;
}

// TODO: remove redundancies with the other derived class
void
ReliableConnection::post_wqes()
{
    int remote_conn;
    get_remote_conn(remote_conn);

    int num_wg = backend->num_wg;
    int lkey = backend->lkey;

    for (int i = 0; i < remote_conn; i++) {
        uint32_t rkey = backend->heap_rkey[i];
        for (int j = 0; j < num_wg; j++) {
            int index = i * num_wg + j;
            cpu_post_wqe(qps[index],
                         nullptr,
                         lkey,
                         nullptr,
                         rkey,
                         10,
                         nullptr,
                         0);

            uint32_t nb_post = 4 * sq_size;
            for (int k = 0; k < nb_post - 1; k++) {
                cpu_post_wqe(qps[index],
                             nullptr,
                             lkey,
                             nullptr,
                             rkey,
                             10,
                             nullptr,
                             0);
            }
        }
    }
}

void
ReliableConnection::initialize_wr_fields(ibv_exp_send_wr &wr,
                                         ibv_ah *ah, int dc_key)
{ }

int
ReliableConnection::get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx)
{
    return pe_idx * num_qps + wg_idx;
}
