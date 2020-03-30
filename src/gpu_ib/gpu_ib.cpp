/******************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <endian.h>

#include <roc_shmem.hpp>
#include <mpi.h>

#include "gpu_ib_internal.hpp"
#include "backend.hpp"

/* ----------------- Internal API ----------------------*/
static roc_shmem_status_t
init_qp_status(ibv_qp *qp, int port)
{
    ibv_exp_qp_attr qp_attr {};

    qp_attr.qp_state        = IBV_QPS_INIT;
    qp_attr.port_num        = port;

#ifdef _USE_DC_
    qp_attr.dct_key         = DC_IB_KEY;
#else
    qp_attr.qp_access_flags = IBV_EXP_ACCESS_REMOTE_WRITE |
                              IBV_EXP_ACCESS_LOCAL_WRITE  |
                              IBV_EXP_ACCESS_REMOTE_READ  |
                              IBV_EXP_ACCESS_REMOTE_ATOMIC;
#endif

    uint64_t exp_attr_mask = IBV_EXP_QP_STATE      |
                             IBV_EXP_QP_PKEY_INDEX |
                             IBV_EXP_QP_PORT;
#ifdef _USE_DC_
    exp_attr_mask |= IBV_EXP_QP_DC_KEY;
#else
    exp_attr_mask |= IBV_EXP_QP_ACCESS_FLAGS;
#endif

    if (ibv_exp_modify_qp(qp, &qp_attr, exp_attr_mask)) {
        printf("Failed to modify QP to INIT  %s\n", strerror(errno));
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    return ROC_SHMEM_SUCCESS;
}

static roc_shmem_status_t
change_status_rtr(ibv_qp *qp, dest_info_t *rem_dest, int ib_port)
{
    ibv_exp_qp_attr attr {};

    attr.qp_state              = IBV_QPS_RTR;
    attr.path_mtu              = IBV_MTU_4096;
    attr.max_dest_rd_atomic    = 1;
    attr.min_rnr_timer         = 12;
    attr.ah_attr.sl            = 1;
    attr.ah_attr.port_num      = ib_port;

#ifndef _USE_DC_ // NOTE: IF_NOT_DEF
    attr.dest_qp_num  = rem_dest->qpn;
    attr.rq_psn       = rem_dest->psn;
    attr.ah_attr.dlid = rem_dest->lid;
#endif

#ifdef _USE_DC_
    attr.dct_key      = DC_IB_KEY;
#endif

    uint64_t exp_attr_mask = IBV_EXP_QP_STATE    |
                             IBV_EXP_QP_AV       |
                             IBV_EXP_QP_PATH_MTU;

#ifndef _USE_DC_ // NOTE: IF_NOT_DEF
    exp_attr_mask |= IBV_EXP_QP_DEST_QPN           |
                     IBV_EXP_QP_RQ_PSN             |
                     IBV_EXP_QP_MAX_DEST_RD_ATOMIC |
                     IBV_EXP_QP_MIN_RNR_TIMER;
#endif

    if (ibv_exp_modify_qp(qp, &attr, exp_attr_mask)) {
        printf("Failed to modify QP to RTR %s\n", strerror(errno));
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    return ROC_SHMEM_SUCCESS;
}

static roc_shmem_status_t
change_status_rts(ibv_qp *qp, dest_info_t *my_dest)
{
    ibv_exp_qp_attr attr {};

    attr.qp_state       = IBV_QPS_RTS;
    attr.timeout        = 14;
    attr.retry_cnt      = 7;
    attr.rnr_retry      = 7; /* infinite */
    attr.max_rd_atomic  = 1;

#ifndef _USE_DC_
    attr.sq_psn         = my_dest->psn;
#endif

    uint64_t exp_attr_mask = IBV_EXP_QP_STATE            |
                             IBV_EXP_QP_TIMEOUT          |
                             IBV_EXP_QP_RETRY_CNT        |
                             IBV_EXP_QP_RNR_RETRY        |
                             IBV_EXP_QP_MAX_QP_RD_ATOMIC;

#ifndef _USE_DC_
    exp_attr_mask |= IBV_EXP_QP_SQ_PSN;
#endif

    if (ibv_exp_modify_qp(qp, &attr, exp_attr_mask)) {
        printf("Failed to modify QP to RTS %s\n", strerror(errno));
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    return ROC_SHMEM_SUCCESS;
}

#ifdef _USE_DC_
static void
connect_dci (struct ibv_qp *qp, int port)
{
    init_qp_status(qp, port);
    change_status_rtr(qp, NULL, port);
    change_status_rts(qp, NULL);
}

static int
create_dct(context_ib_t *ctx, ibv_cq *cq, ibv_srq *srq, int port)
{
    ibv_exp_device_attr dattr;
    int err;
    memset(&dattr,0,sizeof(ibv_exp_device_attr));
    dattr.comp_mask = IBV_EXP_DEVICE_ATTR_EXP_CAP_FLAGS |
                IBV_EXP_DEVICE_DC_RD_REQ |
                IBV_EXP_DEVICE_DC_RD_RES;

    err = ibv_exp_query_device(ctx->context, &dattr);
    if (err) {
        printf("couldn't query device extended attributes\n");
        return -1;
    } else {
        if (!(dattr.exp_device_cap_flags & IBV_EXP_DEVICE_DC_TRANSPORT)) {
            printf("DC transport not enabled\n");
            return -1;
        }
    }

    ibv_exp_dct_init_attr init_attr;

    memset(&init_attr, 0, sizeof(init_attr));

    init_attr.pd               = ctx->pd;
    init_attr.cq               = cq;
    init_attr.srq              = srq;//NULL;
    init_attr.dc_key           = DC_IB_KEY;
    init_attr.port             = port;
    init_attr.mtu              = IBV_MTU_4096;
    init_attr.access_flags     = IBV_EXP_ACCESS_REMOTE_WRITE |
                                 IBV_EXP_ACCESS_LOCAL_WRITE |
                                 IBV_EXP_ACCESS_REMOTE_READ |
                                 IBV_EXP_ACCESS_REMOTE_ATOMIC;

    init_attr.min_rnr_timer    = 7;
    init_attr.hop_limit        = 1;
    init_attr.inline_size      = 4;
    init_attr.tclass           = 0;
    init_attr.flow_label       = 0;
    init_attr.pkey_index           = 0;
    init_attr.gid_index            = 0;

    ibv_exp_dct * dct = NULL;

    dct = ibv_exp_create_dct(ctx->context, &init_attr);

    if (dct == NULL) {
        printf("Failed to created DC target %s\n", strerror(errno));
    }

    ibv_exp_dct_attr dcqattr;
    memset(&dcqattr,0,sizeof(ibv_exp_dct_attr));

    err = ibv_exp_query_dct(dct, &dcqattr);
        if (err) {
            printf("query dct failed\n");
            return -1;
        } else if (dcqattr.dc_key != DC_IB_KEY) {
            printf("queried dckry (0x%llx) is different then provided at "
                   "create (0x%llx)\n", (unsigned long long)dcqattr.dc_key,
                   (unsigned long long)DC_IB_KEY);
            return -1;
        } else if (dcqattr.state != IBV_EXP_DCT_STATE_ACTIVE) {
            printf("state is not active %d\n", dcqattr.state);
            return -1;
        }

    return  dct->dct_num;
}
#endif

static roc_shmem_status_t
create_qps(context_ib_t *ctx, dest_info_t *all_qp, int port, int rtn_id,
           int num_qp, int comm_size, int my_rank,
           struct ibv_port_attr *ib_port_att, int SQ_size,
           RTNGlobalHandle *rtn_handle)
{
    int i,j;

    rtn_cq_t **tab_rcq;
    tab_rcq = (rtn_cq_t **) malloc(sizeof(rtn_cq_t*) * num_qp*comm_size);

#ifdef _USE_DC_
    struct ibv_srq_init_attr srq_init_attr;
    memset(&srq_init_attr, 0, sizeof(srq_init_attr));
    srq_init_attr.attr.max_wr  = 1;
    srq_init_attr.attr.max_sge = 1;

    srq = ibv_create_srq(ctx->pd, &srq_init_attr);
    if (!srq) {
        fprintf(stderr, "Error, ibv_create_srq() failed\n");
        free(tab_rcq);
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    dct_cq = ibv_create_cq(ctx->context, 100, NULL, NULL, 0);
    if (!dct_cq) {
        fprintf(stderr, "Error, ibv_create_cq() failed\n");
        free(tab_rcq);
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
#endif

    struct ibv_qp_cap cap;
    cap.max_send_wr  = SQ_size;
    cap.max_recv_wr  = 0;
    cap.max_send_sge = 1;
    cap.max_inline_data = 4;

    struct ibv_exp_qp_init_attr attr;
    memset(&attr, 0, sizeof(ibv_exp_qp_init_attr));
    attr.send_cq = 0;
    attr.recv_cq = 0;
    attr.srq     = NULL;
    attr.cap     = cap;
#ifdef _USE_DC_
    attr.qp_type = IBV_EXP_QPT_DC_INI;
#else
    attr.qp_type = IBV_QPT_RC;
#endif
    attr.sq_sig_all = 1;

    for (j= 0; j < comm_size; j++) {
        for (i = 0; i < num_qp; i++) {
            int offset = j * num_qp + i;
            tab_rcq[offset] = rtn_create_cq(ctx->context, attr.cap.max_send_wr,
                NULL, NULL, 0, rtn_id);

            if (!tab_rcq[offset]) {
                printf(" Error : Failed to create RTN_CQ num {%d, %d}\n", i,j);
                return ROC_SHMEM_UNKNOWN_ERROR;
            }
        }
    }

#ifdef _USE_DC_
    for (i = 0; i < num_dct; i++) {
        //all_qp[my_rank*num_dct +i].dct_num =
        //  create_dct(ctx, rtn_get_cq(tab_rcq[i]), srq, port);
        int32_t dct_num = create_dct(ctx, dct_cq, srq, port);
        dct_num = htobe32(dct_num);
        //dct_num = dct_num >>8;
        dcts_num[my_rank * num_dct + i]= dct_num;
    }
    lids[my_rank] = htobe16(ib_port_att->lid);

#endif

    for (j= 0; j < comm_size; j++) {
        for (i = 0; i < num_qp; i++) {
            int offset = j * num_qp + i;
            rtn_qp_t *rqp =
                rtn_create_qp(ctx->pd, ctx->context, &attr, tab_rcq[offset],
                              rtn_id, i , j, rtn_handle);
            if (rqp == NULL) {
                printf(" Error: Failed to create RTN_QP num [%d. %d] \n",i ,j);
                return ROC_SHMEM_UNKNOWN_ERROR;
            }

            ibv_qp *qp;
            qp = rtn_get_qp(rqp);

#ifndef _USE_DC_
            init_qp_status(qp, port);

            all_qp[offset].lid = ib_port_att->lid;
            all_qp[offset].qpn = qp->qp_num;
            all_qp[offset].psn = 0;
#else
            connect_dci (qp, port);
#endif
        }
    }
    return ROC_SHMEM_SUCCESS;
}

static context_ib_t*
ib_init_ctx(struct ibv_device *ib_dev, int ib_port)
{
    context_ib_t *ctx;

    ctx = (context_ib_t *) malloc(sizeof(context_ib_t));
    if (!ctx)
        return NULL;

    ctx->context = ibv_open_device(ib_dev);
    if (!ctx->context) {
        printf("Error : Couldn't get context \n");
    }

    ctx->pd = ibv_alloc_pd(ctx->context);

    if (!ctx->pd) {
        printf("Error : Couldn't allocate PD\n");
    }

    ibv_query_port (ctx->context, ib_port, &ctx->portinfo);

    return ctx;
}

void
GPUIBBackend::thread_cpu_post_wqes()
{
    roc_shmem *shmem_handle =
        reinterpret_cast<roc_shmem*>(backend_handle);

    uint32_t nb_post = 0;
    int remote_conn;
    rtn_qp_t * rqp_t;
    int num_wg= shmem_handle->num_wg;
#ifdef _USE_DC_
    remote_conn = num_dcis;
#else
    remote_conn = num_pes;
#endif
    int lkey = shmem_handle->lkey;
    for(int i= 0; i < remote_conn; i++)
    {
#ifndef _USE_DC_
        uint32_t rkey= shmem_handle->heap_rkey[i];
#endif
        for(int j = 0; j < num_wg; j++) {
            rqp_t = rtn_get_rqp(shmem_handle->rtn_handle, j, i);
            //nb_post = last_post;
            if (first_time == true) {
                nb_post = (4*SQ_size);
#ifndef _USE_DC_
                rtn_cpu_post_wqe(rqp_t, NULL, lkey, NULL, rkey, 10, NULL, 0);
#else
                rtn_cpu_post_wqe(rqp_t, NULL, lkey, NULL, 0, 10, ah,
                                 DC_IB_KEY);
#endif

                rtn_set_sq_dv(shmem_handle->rtn_handle, j, i);
#ifndef _USE_GPU_UPDATE_SQ_
                for (int k = 0; k < nb_post; k++) {
                    rtn_post_wqe_dv(shmem_handle->rtn_handle, j, i, true);
                    //rtn_cpu_post_wqe(rqp_t, NULL, lkey, NULL, rkey, 10, NULL,
                    //                 0);
                }
#else
                for(int k=0; k < nb_post-1; k++)
                {
#ifndef _USE_DC_
                    rtn_cpu_post_wqe(rqp_t, NULL, lkey, NULL, rkey, 10, NULL,
                                     0);
#else
                    rtn_cpu_post_wqe(rqp_t, NULL, lkey, NULL, 0, 10, ah,
                                     DC_IB_KEY);
#endif
                }
#endif
            }
#ifndef _USE_GPU_UPDATE_SQ_
            else {
                nb = rtn_get_sq_counter(rqp_t);
                nb_post =  (nb - last_nb_post[i * num_wg + j]);
                //abs(last_post - nb_post);
                last_nb_post[i * num_wg + j] = nb;

                for (int k = 0; k < nb_post; k++) {
                    rtn_post_wqe_dv(shmem_handle->rtn_handle, j, i, false);
                }
                printf("nb_post %d nb %d \n", nb_post, nb);
                fflush(stdout);
            }
#endif
        }
    }
}

void
GPUIBBackend::thread_func(int sleep_time)
{
    roc_shmem *shmem_handle =
        reinterpret_cast<roc_shmem*>(backend_handle);

    //last_post = SQ_size;
    while (shmem_handle->thread_done == false) {
        thread_cpu_post_wqes();
        if (first_time == true) {
            first_time = false;
        }
        sleep(sleep_time);
    }
}

/* ----------------- External API ----------------------*/
roc_shmem_status_t
GPUIBBackend::net_free(void *ptr)
{
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::dump_backend_stats()
{
    roc_shmem *shmem_handle =
        reinterpret_cast<roc_shmem*>(backend_handle);

    return rtn_dump_backend_stats(shmem_handle->rtn_handle,
                                  globalStats.getStat(NUM_FINALIZE));
}

roc_shmem_status_t
GPUIBBackend::reset_backend_stats()
{
    roc_shmem *shmem_handle =
        reinterpret_cast<roc_shmem*>(backend_handle);

    return rtn_reset_backend_stats(shmem_handle->rtn_handle);
}

roc_shmem_status_t
GPUIBBackend::pre_init()
{
    int comm_size, my_rank;
    int flag = 0;
    struct roc_shmem *shmem_handle;
    CHECK_HIP(hipMalloc((void**) &shmem_handle, sizeof(struct roc_shmem)));

    backend_handle = shmem_handle;
    type = GPU_IB_BACKEND;

    MPI_Initialized(&flag);

    if (!flag) {
        MPI_Init(NULL, NULL);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    my_pe = my_rank;
    num_pes = comm_size;

    pre_init_done = 1;

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::init(int num_wg)
{
    context_ib_t *ctx;
    struct ibv_device **dev_list = NULL;
    struct ibv_device *ib_dev = NULL;
    int comm_size, my_rank, ib_devices, remote_conn;
    char * requested_dev = NULL;
    struct roc_shmem *shmem_handle;
    RTNGlobalHandle * rtn_handle;
    uint32_t *hdp_rkey;
    uintptr_t *hdp_address;

    int port = 1;
    int rtn_id = 0;
    int ret = 0;

    if (pre_init_done == 0){
        pre_init();
    }

    shmem_handle = reinterpret_cast<roc_shmem*>(backend_handle);

    comm_size = num_pes;
    my_rank   = my_pe;

    //MPI_Init(NULL, NULL);
    //MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    //MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    char *value = NULL;
    value = getenv("ROC_SHMEM_USE_IB_HCA");
    if (value != NULL) {
        requested_dev = value;
    }

    value = getenv("ROC_SHMEM_HEAP_SIZE");
    if (value != NULL) {
        heap_size = atoi(value);
    }

    value = getenv("ROC_SHMEM_SLEEP");
    if (value != NULL) {
        sleep_thread = atoi(value);
    }
    value = getenv("ROC_SHMEM_SQ_SIZE");
    if (value != NULL) {
        SQ_size = atoi(value);
    }
#ifdef _USE_DC_
    value = getenv("ROC_SHMEM_NUM_DCIs");
    if (value != NULL) {
        num_dcis = atoi(value);
    }

    value = getenv("ROC_SHMEM_NUM_DCT");
    if (value != NULL) {
        num_dct = atoi(value);
    }
#endif

    dev_list = ibv_get_device_list (&ib_devices);
    if (dev_list == NULL){
        printf("ibv_get_device_list returned NULL \n");
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    ib_dev = dev_list[0];
    if (requested_dev != NULL) {
        int i;
        for (i = 0; i < ib_devices; i++) {
            const char *select_dev = ibv_get_device_name(dev_list[i]);
            if (strstr(select_dev, requested_dev) != NULL) {
                ib_dev = dev_list[i];
                break;
            }
        }
        if (i == ib_devices) {
            ib_dev = dev_list[0];
        }
    }

    ctx = ib_init_ctx(ib_dev, port);

#ifdef _USE_DC_
    remote_conn = num_dcis;
    dest_info_t * all_qp = NULL;

    lids =(uint16_t*) malloc(sizeof(uint16_t) * comm_size);
    if(lids == NULL ){
        printf("ERROR: Could not allocate memory for LIDs information \n");
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    dcts_num =(uint32_t*) malloc(sizeof(uint32_t) * num_dct * comm_size);
    if(dcts_num == NULL ){
        printf("ERROR: Could not allocate memory for DCTs information \n");
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

#else
    remote_conn = comm_size;

    dest_info_t * all_qp =
            (dest_info_t*) malloc(sizeof(dest_info_t)* comm_size * num_wg );
    if(all_qp == NULL ){
        printf("ERROR: Could not allocate memory for QP information \n");
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

#endif
    uint32_t *host_hdp_cpy;
    uint64_t *host_hdp_address_cpy;

    rtn_handle = rtn_init(rtn_id, num_wg, remote_conn, ctx->pd);
    if (rtn_handle == NULL) {
        printf("ERROR: Failed to initialize RTN \n");
        free(all_qp);
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    // exchange the hdp info for a fence implementation
    CHECK_HIP(hipMalloc((void**) &hdp_rkey,
                sizeof(uint32_t) *comm_size));
    CHECK_HIP(hipMalloc((void**) &hdp_address,
                sizeof(uintptr_t) *comm_size));

    host_hdp_cpy = (uint32_t *)malloc(sizeof(uint32_t) * comm_size);
    if (host_hdp_cpy == NULL)
        printf("ERROR: malloc failed \n");
    host_hdp_address_cpy = (uint64_t *)malloc(sizeof(uint64_t) *comm_size);
    if (host_hdp_address_cpy == NULL)
        printf("ERROR: malloc failed \n");

    host_hdp_cpy[my_rank] = rtn_get_hdp_rkey(rtn_handle);
    host_hdp_address_cpy[my_rank] = rtn_get_hdp_address(rtn_handle);

    MPI_Allgather(MPI_IN_PLACE, sizeof(uint32_t),
                  MPI_CHAR, host_hdp_cpy, sizeof(uint32_t),
                  MPI_CHAR, MPI_COMM_WORLD);

    MPI_Allgather(MPI_IN_PLACE, sizeof(uintptr_t),
                  MPI_CHAR, host_hdp_address_cpy, sizeof(uintptr_t),
                  MPI_CHAR, MPI_COMM_WORLD);

    CHECK_HIP(hipMemcpy(hdp_rkey, host_hdp_cpy,
              sizeof(uint32_t) *  comm_size, hipMemcpyHostToDevice));

    CHECK_HIP(hipMemcpy(hdp_address, host_hdp_address_cpy,
              sizeof(uint64_t) *  comm_size, hipMemcpyHostToDevice));

    free(host_hdp_cpy);
    free(host_hdp_address_cpy);

    rtn_hdp_add_info(rtn_handle, hdp_rkey, hdp_address);
    // end of HDP handling

    ret = create_qps(ctx, all_qp, port, rtn_id, num_wg, remote_conn, my_rank,
                    &ctx->portinfo, SQ_size, rtn_handle);

#ifndef _USE_DC_
    MPI_Alltoall(MPI_IN_PLACE, sizeof(dest_info_t) *num_wg,
                         MPI_CHAR, all_qp, sizeof(dest_info_t) * num_wg,
                         MPI_CHAR, MPI_COMM_WORLD);

    for(int i = 0; i < num_wg; i++) {
        for (int j = 0; j < comm_size; j++) {
            ibv_qp* qp = rtn_get_qp(rtn_get_rqp(rtn_handle, i, j));
            int offset = j*num_wg+i;
            change_status_rtr(qp, &all_qp[offset], port);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < num_wg; i++) {
        for (int j = 0; j < comm_size; j++) {
            ibv_qp* qp = rtn_get_qp(rtn_get_rqp(rtn_handle, i, j));
            change_status_rts(qp, &all_qp[j * num_wg + i]);
        }
    }
#else
     MPI_Allgather(MPI_IN_PLACE, sizeof(int32_t) * num_dct,
                         MPI_CHAR, dcts_num, sizeof(int32_t) * num_dct,
                         MPI_CHAR, MPI_COMM_WORLD);

     MPI_Allgather(MPI_IN_PLACE, sizeof(int16_t),
                         MPI_CHAR, lids, sizeof(int16_t),
                         MPI_CHAR, MPI_COMM_WORLD);

    CHECK_HIP(hipMalloc((void**) &shmem_handle->vec_dct_num,
                sizeof(int32_t) * num_dct* comm_size));

    CHECK_HIP(hipMemcpy(shmem_handle->vec_dct_num, dcts_num,
                sizeof(int32_t) * num_dct* comm_size, hipMemcpyHostToDevice));

    CHECK_HIP(hipMalloc((void**) &shmem_handle->vec_lids,
                sizeof(int16_t) * comm_size));

    CHECK_HIP(hipMemcpy(shmem_handle->vec_lids, lids,
                sizeof(int16_t) *  comm_size, hipMemcpyHostToDevice));

    struct ibv_ah_attr ah_attr;
    memset(&ah_attr, 0, sizeof(ah_attr));

    ah_attr.is_global     = 0;
    ah_attr.dlid          = ctx->portinfo.lid;
    ah_attr.sl            = 1;
    ah_attr.src_path_bits = 0;
    ah_attr.port_num      = port;

    ah = ibv_create_ah(ctx->pd, &ah_attr);
    if (ah == NULL) {
        printf("ERROR: ibv_create_ah failed \n");
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
#endif

    last_nb_post = (uint32_t*) malloc(num_wg * comm_size * sizeof(uint32_t));
    if (last_nb_post == NULL)
        printf("Error: Could not allocate Host memory for last_nb_post \n");

    memset(last_nb_post, 0 , num_wg *comm_size *sizeof(uint32_t));

    MPI_Barrier(MPI_COMM_WORLD);
    free(all_qp);

    // alocate and register heap memory
    char **heap_bases, **host_bases_cpy;
    uint32_t *heap_rkey, *host_rkey_cpy;
    void     *base_heap;

    host_bases_cpy = (char **) malloc(sizeof(*host_bases_cpy) * comm_size);

    if (host_bases_cpy == NULL)
        printf("ERROR : malloc failed \n");

    host_rkey_cpy = (uint32_t*) malloc(sizeof(uint32_t*) * comm_size);

    if (host_rkey_cpy == NULL)
        printf("ERROR : malloc failed \n");

#ifndef _USE_DC_
    CHECK_HIP(hipHostMalloc((void**) &heap_rkey,
                            sizeof(uint32_t) * comm_size));
#else
    CHECK_HIP(hipMalloc((void**) &heap_rkey, sizeof(uint32_t) * comm_size));
#endif
    CHECK_HIP(hipMalloc((void**) &heap_bases, sizeof(*heap_bases) * comm_size));

    //allocate the heap on UC region
    //CHECK_HIP(hipMalloc((void**) &base_heap, heap_size));
    CHECK_HIP(hipExtMallocWithFlags((void**)&base_heap, heap_size,
                                    hipDeviceMallocFinegrained));
    //CHECK_HIP(hipHostMalloc((void**) &base_heap, heap_size));

    ibv_mr *mr = ibv_reg_mr(ctx->pd, base_heap, heap_size,
                            IBV_EXP_ACCESS_LOCAL_WRITE |
                            IBV_EXP_ACCESS_REMOTE_WRITE |
                            IBV_EXP_ACCESS_REMOTE_READ |
                            IBV_EXP_ACCESS_REMOTE_ATOMIC);
    if (mr == NULL) {
        printf("ERROR: PE [%d] Failed to register the heap at %p of size %d\n",
               my_rank, base_heap, heap_size);
    }

    heap_bases[my_rank] = (char *) base_heap;
#ifndef _USE_DC_
    heap_rkey[my_rank]  = mr->rkey;
#else
    heap_rkey[my_rank]  = htobe32(mr->rkey);
#endif

    CHECK_HIP(hipMemcpy(host_bases_cpy, heap_bases, sizeof(*heap_bases) *
                        comm_size, hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(host_rkey_cpy, heap_rkey,
                        sizeof(uint32_t*) * comm_size, hipMemcpyDeviceToHost));

    MPI_Allgather(MPI_IN_PLACE, sizeof(heap_bases), MPI_CHAR, host_bases_cpy,
                  sizeof(*heap_bases), MPI_CHAR, MPI_COMM_WORLD);

    MPI_Allgather(MPI_IN_PLACE, sizeof(uint32_t), MPI_CHAR, host_rkey_cpy,
                  sizeof(uint32_t), MPI_CHAR, MPI_COMM_WORLD);

    CHECK_HIP(hipMemcpy(heap_bases, host_bases_cpy, sizeof(*heap_bases) *
                        comm_size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(heap_rkey, host_rkey_cpy,
                        sizeof(uint32_t*) * comm_size, hipMemcpyHostToDevice));

    free(host_bases_cpy);
    free(host_rkey_cpy);

    // IPC support
#ifdef _USE_IPC_
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                        MPI_INFO_NULL, &shmcomm);

    int shm_rank, shm_size;
    MPI_Comm_size(shmcomm, &shm_size);
    MPI_Comm_rank(shmcomm, &shm_rank);

    hipIpcMemHandle_t *vec_ipc_handle;
    vec_ipc_handle = (hipIpcMemHandle_t *)malloc(sizeof(hipIpcMemHandle_t) *
                                                 shm_size);

    CHECK_HIP(hipIpcGetMemHandle(&vec_ipc_handle[shm_rank], base_heap));

    MPI_Allgather(MPI_IN_PLACE, sizeof(hipIpcMemHandle_t),
                         MPI_CHAR, vec_ipc_handle, sizeof(hipIpcMemHandle_t),
                         MPI_CHAR, shmcomm);


    char **ipc_bases;
    CHECK_HIP(hipMalloc((void**) &ipc_bases, sizeof(ipc_bases) * shm_size));
    for(int i = 0; i < shm_size; i++) {
        if (i != shm_rank) {
            CHECK_HIP(hipIpcOpenMemHandle((void**)&ipc_bases[i],
                                          vec_ipc_handle[i],
                                          hipIpcMemLazyEnablePeerAccess));
        } else {
            ipc_bases[i] = (char *) base_heap;
        }
    }
    shmem_handle->shm_size = (int8_t)shm_size;
    shmem_handle->ipc_bases = ipc_bases;

    free(vec_ipc_handle);
#endif

    shmem_handle->lkey       = mr->lkey;
    shmem_handle->num_wg     = num_wg;
    shmem_handle->heap_bases = heap_bases;
    shmem_handle->heap_rkey  = heap_rkey;
    shmem_handle->heap_mr    = mr;
    shmem_handle->rtn_handle = rtn_handle;
    shmem_handle->current_heap_offset = 0;
    shmem_handle->thread_done = false;
#ifdef _USE_DC_
    rtn_dc_add_info(rtn_handle, (uint32_t*)shmem_handle->vec_dct_num,
                    (uint16_t*)shmem_handle->vec_lids,
                    (uint32_t*)shmem_handle->heap_rkey);
#endif

    // init the collectives
    roc_shmem_collective_init();
    // pre-allocate heap space for get_p return
    roc_shmem_g_init();

    first_time = true;

    worker_thread = new std::thread(&GPUIBBackend::thread_func, this,
                                    sleep_thread);

    //thread_cpu_post_wqes(shmem_handle, true);

    // block until the thread is done with first post
    while (first_time == true) { }

    return ROC_SHMEM_SUCCESS;
}

void
GPUIBBackend::roc_shmem_collective_init()
{
    roc_shmem *shmem_handle =
        reinterpret_cast<roc_shmem*>(backend_handle);

    int comm_size = num_pes;

    int64_t *ptr = (int64_t*) shmem_handle->heap_bases[my_pe] +
        shmem_handle->current_heap_offset;

    shmem_handle->current_heap_offset = shmem_handle->current_heap_offset +
        SHMEM_BARRIER_SYNC_SIZE;

    shmem_handle->barrier_sync = ptr;

    for(int i = 0; i < comm_size; i++)
        shmem_handle->barrier_sync[i]= SHMEM_SYNC_VALUE;

    MPI_Barrier(MPI_COMM_WORLD);
}
void
GPUIBBackend::roc_shmem_g_init()
{
    roc_shmem *shmem_handle =
        (roc_shmem *) backend_handle;

    int num_wg = shmem_handle->num_wg;

    char *ptr = (char*) shmem_handle->heap_bases[my_pe] +
        shmem_handle->current_heap_offset;

    shmem_handle->current_heap_offset = shmem_handle->current_heap_offset +
        sizeof(int64_t)* MAX_WG_SIZE * num_wg;

    shmem_handle->g_ret = (char*) ptr;

    MPI_Barrier(MPI_COMM_WORLD);
}

roc_shmem_status_t
GPUIBBackend::net_malloc(void **ptr, size_t size)
{
    if (pre_init_done == 0) {
        printf("error, ROC_SHMEM is not initialized \n");
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    roc_shmem *shmem_handle =
        reinterpret_cast<roc_shmem*>(backend_handle);
    *ptr = (char*)  shmem_handle->heap_bases[my_pe] +
        shmem_handle->current_heap_offset;

    shmem_handle->current_heap_offset = shmem_handle->current_heap_offset +
        (size / sizeof(char));

    MPI_Barrier(MPI_COMM_WORLD);
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::finalize()
{
    if (pre_init_done == 0) {
        printf("error, ROC_SHMEM is not initialized \n");
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    int ret = 0;

    struct roc_shmem * shmem_handle =
        (struct roc_shmem *) backend_handle;

    shmem_handle->thread_done = true;

    worker_thread->join();
    delete worker_thread;

    ret = ibv_dereg_mr(shmem_handle->heap_mr);
    if(ret != 0)
        printf("ERROR: Failed to deregister the heap memory \n");

    void* heap_bases = shmem_handle->heap_bases;
    CHECK_HIP(hipFree(heap_bases));
#ifdef _USE_DC_
    CHECK_HIP(hipFree(shmem_handle->heap_rkey));
    CHECK_HIP(hipFree(shmem_handle->vec_dct_num));
    CHECK_HIP(hipFree(shmem_handle->vec_lids));
#else
    CHECK_HIP(hipHostFree(shmem_handle->heap_rkey));
#endif

#ifdef _USE_IPC_
    CHECK_HIP(hipFree(shmem_handle->ipc_bases));
#endif

    rtn_finalize(shmem_handle->rtn_handle, 0);
    // close ib_ctx();
    MPI_Finalize();
    pre_init_done = 0;
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::dynamic_shared(size_t *shared_bytes)
{
    uint32_t heap_usage = 0;
    uint32_t rtn_usage = 0;
    uint32_t remote_conn = 1;
#ifndef _USE_DC_
    remote_conn = num_pes;
#endif

    heap_usage = sizeof(uint64_t) * num_pes;

    //rtn_usage = sizeof(struct roc_shmem_wg) + (sizeof_rtn()*remote_conn);
    rtn_usage = (sizeof_rtn() * remote_conn);

    *shared_bytes = heap_usage + rtn_usage + sizeof(GPUIBContext);
    return ROC_SHMEM_SUCCESS;
}
