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

#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <endian.h>

#include "rtn.hpp"
#include "rtn_internal.hpp"


RTNGlobalHandle* rtn_init(int rtn_id, int num_qp, int remote_conn,
                          struct ibv_pd *pd)
{

    RTNGlobalHandle *rtn_handle;
    hdp_reg_t   *hdp_regs;
    uint64_t    *atomic_base_ptr;
    struct      ibv_mr *mr;
#ifdef _RECYCLE_QUEUE_
    uint32_t    *softohw;
    uint32_t    **hwtoqueue;
    uint32_t    *queueTocken;
#endif

    rtn_rocm_init();
    CHECK_HIP(hipSetDevice(rtn_id));

    int ib_fork_err = ibv_fork_init();
    if(ib_fork_err !=0)
        printf("error: ibv)fork_init  failed \n");

    CHECK_HIP(hipMalloc((void**)&rtn_handle,
                        sizeof(RTNGlobalHandle)* num_qp *remote_conn ));

    memset(rtn_handle, 0,  sizeof(RTNGlobalHandle)* num_qp *remote_conn);

    CHECK_HIP(hipExtMallocWithFlags((void**)&atomic_base_ptr,
                                    sizeof(uint64_t)* max_nb_atomic * num_qp,
                                    hipDeviceMallocFinegrained));

    memset(atomic_base_ptr, 0,  sizeof(uint64_t)* max_nb_atomic* num_qp);


    mr = ibv_reg_mr(pd, atomic_base_ptr, sizeof(uint64_t) * max_nb_atomic *
                    num_qp, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

    if (mr == NULL) {
        fprintf(stderr, "IB error: could not register a memory \n");
        exit(EXIT_FAILURE);
    }

    rtn_atomic_ret_t * atomic_ret;

    CHECK_HIP(hipMalloc((void**)&atomic_ret,
                        sizeof(rtn_atomic_ret_t)));

    atomic_ret->atomic_base_ptr = (void*) atomic_base_ptr;
    atomic_ret->atomic_lkey     = htobe32(mr->lkey);
#ifdef _USE_HDP_MAP_
    // mmap yhe HDP registers and get it ready for
    // ibv_reg_mr with the Peer_direct
    CHECK_HIP(hipMalloc((void**) &hdp_regs, sizeof(hdp_reg_t)));
    hdp_map(hdp_regs,rtn_id);

    void* dev_ptr;
    rtn_rocm_memory_lock_to_fine_grain(hdp_regs->cpu_hdp_read_inv,
                                       getpagesize(), &dev_ptr, rtn_id);

    hdp_regs->gpu_hdp_read_inv = (unsigned int *)dev_ptr;

    rtn_rocm_memory_lock_to_fine_grain(hdp_regs->cpu_hdp_flush, getpagesize(),
                                       &dev_ptr, rtn_id);

    hdp_regs->gpu_hdp_flush = (unsigned int *)dev_ptr;

    mr = ibv_reg_mr(pd, hdp_regs->cpu_hdp_flush, 32,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                    IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

    if (mr == NULL) {
        fprintf(stderr, "IB error: could not register a memory (hdp_flush_mr) %p %d \n",
                         hdp_regs->cpu_hdp_flush, errno);
        exit(EXIT_FAILURE);
    }

    hdp_regs->rkey    = htobe32(mr->rkey);
#else
   hdp_reg_t *rtn_hdp = rtn_hdp_flush_map(rtn_id);
   hdp_regs = rtn_hdp_flush_map(rtn_id);

   mr = ibv_reg_mr(pd, hdp_regs->cpu_hdp_flush, 32,
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                   IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);

    if (mr == NULL) {
        fprintf(stderr, "IB error: could not register a memory (rtn:hdp_flush_mr) "
                        "%p %d \n", hdp_regs->cpu_hdp_flush, errno);
        exit(EXIT_FAILURE);
    }

   hdp_regs->rkey = htobe32(mr->rkey);
#endif //end of HDP mmaping

    char *value;

    if ((value = getenv("RTN_USE_CQ_GPU_MEM")) != NULL) {
        cq_use_gpu_mem = atoi(value);
    }
    if ((value = getenv("RTN_USE_SQ_GPU_MEM")) != NULL) {
        sq_use_gpu_mem = atoi(value);
    }

    rtn_handle->rtn_sq_post_dv = (rtn_sq_post_dv_t*) malloc(
        sizeof(rtn_sq_post_dv_t)* num_qp * remote_conn);
    if( rtn_handle->rtn_sq_post_dv == NULL) {
        printf("error: could not allocate the sq_post_dv buffer \n");
    }

#ifdef _RECYCLE_QUEUE_
    if ((value = getenv("RTN_NUM_CU")) != NULL) {
        rtn_num_cu = atoi(value);
    }
    rtn_num_qp_cu = (num_qp / rtn_num_cu) > 0 ? num_qp / rtn_num_cu : 1;
    printf("rtn_num_qp_cu  %d \n", rtn_num_qp_cu);

    CHECK_HIP(hipMalloc((void**) &softohw, sizeof(uint32_t) * num_qp));
    CHECK_HIP(hipMalloc((void**) &queueTocken, sizeof(uint32_t) * num_qp));
    memset(queueTocken, 0, sizeof(uint32_t) * num_qp);

    CHECK_HIP(hipMalloc((void***) &hwtoqueue, sizeof(uint32_t*) * MAX_NUM_CU));
    for (int i = 0; i < MAX_NUM_CU; i++) {
        CHECK_HIP(hipMalloc((void**) &hwtoqueue[i],
                            sizeof(uint32_t) * (rtn_num_qp_cu + 1)));
        (hwtoqueue[i])[0] = 1;
    }
#endif

    for (int i = 0; i < num_qp * remote_conn; i++) {
        rtn_handle[i].hdp_regs = hdp_regs;
        rtn_handle[i].atomic_ret = atomic_ret;
        rtn_handle[i].sq_counter = 0;
        rtn_handle[i].cq_counter = 0;
        rtn_handle[i].quiet_counter = 0;
        rtn_handle[i].sq_overflow = false;
        rtn_handle[i].num_cqs = remote_conn;
#ifdef _RECYCLE_QUEUE_
        rtn_handle[i].softohw = softohw;
        rtn_handle[i].hwtoqueue = hwtoqueue;
        rtn_handle[i].queueTocken = queueTocken;
#endif
    }
    rtn_handle->num_qps = num_qp;
    return rtn_handle;
}


int rtn_finalize(RTNGlobalHandle *rtn, int rtn_id)
{
#ifdef _USE_HDP_MAP_
    close(rtn->hdp_regs->fd);
#endif
    CHECK_HIP(hipFree((void*)rtn->hdp_regs));
    CHECK_HIP(hipFree((void*)rtn->atomic_ret));
    return 1;
}

uint32_t rtn_get_hdp_rkey(RTNGlobalHandle *rtn_handle)
{
    return (uint32_t)rtn_handle->hdp_regs->rkey;
}

uintptr_t rtn_get_hdp_address(RTNGlobalHandle *rtn_handle)
{
    return (uintptr_t)rtn_handle->hdp_regs->cpu_hdp_flush;
}

void rtn_hdp_add_info(RTNGlobalHandle *rtn_handle, uint32_t *vec_rkey,
                      uintptr_t* vec_address)
{
    rtn_handle->hdp_rkey    = vec_rkey;
    rtn_handle->hdp_address = vec_address;
}

//-----------------------------------------------------------------------------

ibv_qp* rtn_get_qp (rtn_qp_t *rqp_t)
{
    struct rtn_qp * rqp = (struct rtn_qp*) rqp_t;
    return rqp->qp;
}

ibv_cq* rtn_get_cq(rtn_cq_t *rcq_t)
{
    struct rtn_cq *rcq = (struct rtn_cq*) rcq_t;
    return rcq->cq;
}

uint32_t rtn_get_sq_counter(rtn_qp_t *rqp_t)
{
    struct rtn_qp *rqp = (struct rtn_qp*) rqp_t;
    uint32_t lval =( (uint32_t)*(rqp->dbrec_send_H) );
    return be32toh(lval);
}


rtn_qp_t* rtn_get_rqp (RTNGlobalHandle *rtn_handle, int wg_idx, int pe_idx)
{
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
    return (rtn_qp_t*)rtn_handle[offset].rqp;
}

rtn_cq_t* rtn_get_rcq (RTNGlobalHandle *rtn_handle, int wg_idx, int pe_idx)
{
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
    return (rtn_cq_t*)rtn_handle[offset].rqp->rcq;
}

uint32_t rtn_get_cq_counter(rtn_cq_t *rcq_t)
{
    struct rtn_cq *rcq = (struct rtn_cq*) rcq_t;
    uint32_t lval =( (uint32_t)*(rcq->dbrec_cq_H) );
    return be32toh(lval);
}

void rtn_set_sq_dv(RTNGlobalHandle *rtn_handle, int wg_idx, int pe_idx)
{
#ifndef _USE_DC_
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
#else
    int offset = wg_idx;
#endif

    memcpy((void*) rtn_handle->rtn_sq_post_dv[offset].segments,
           ((uint64_t*)rtn_handle[offset].rqp->sq_base_H), 64);

    rtn_handle->rtn_sq_post_dv[offset].wqe_idx = 0;
    rtn_handle->rtn_sq_post_dv[offset].current_sq = 0;

#ifdef _USE_GPU_UPDATE_SQ_
    uint64_t ctrl_sig =
        ((uint64_t*)(rtn_handle->rtn_sq_post_dv[offset].segments))[1];
    uint32_t ctrl_qp_sq =
        ((uint32_t*)(rtn_handle->rtn_sq_post_dv[offset].segments))[1];
    ctrl_qp_sq = ctrl_qp_sq & 0xFFFFF0;

#ifndef _USE_DC_
    uint32_t rkey =
        ((uint32_t*)(rtn_handle->rtn_sq_post_dv[offset].segments))[6];
    uint32_t lkey =
        ((uint32_t*)(rtn_handle->rtn_sq_post_dv[offset].segments))[9];
#else
    uint32_t rkey =
        ((uint32_t*)(rtn_handle->rtn_sq_post_dv[offset].segments))[10];
    uint32_t lkey =
        ((uint32_t*)(rtn_handle->rtn_sq_post_dv[offset].segments))[13];
#endif

    rtn_handle[offset].lkey = lkey;
    rtn_handle[offset].rkey = rkey;
    rtn_handle[offset].ctrl_qp_sq = ctrl_qp_sq;
    rtn_handle[offset].ctrl_sig = ctrl_sig;
#endif

    //printf("%llx sig %llx  qp_sq %llx  rkey %llx lkey %llx \n",
    //       ((uint64_t*)(rtn_handle->rtn_sq_post_dv[offset].segments))[0],
    //       ctrl_sig, ctrl_qp_sq, rkey, lkey);
}

void rtn_post_wqe_dv(RTNGlobalHandle *rtn_handle, int wg_idx, int pe_idx,
                     bool flag)
{
#ifndef _USE_GPU_UPDATE_SQ_
#ifndef _USE_DC_
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
#else
    int offset = wg_idx;
#endif

    uint32_t sq_size = (rtn_handle[offset].rqp->max_nwqe);
    uint32_t current_offset =
        (((uint32_t)rtn_handle->rtn_sq_post_dv[offset].current_sq));

    uint64_t * current_sq =
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H) + (current_offset * 8);

    uint64_t wqe_idx_val =
        htobe64((uint64_t) rtn_handle->rtn_sq_post_dv[offset].wqe_idx);
    wqe_idx_val = wqe_idx_val >> 40;
    uint64_t cntl_seg =
        (rtn_handle->rtn_sq_post_dv[offset].segments[0]
        & 0x0000000000ffff00 | wqe_idx_val);
    cntl_seg = cntl_seg | rtn_handle->rtn_sq_post_dv[offset].segments[0];

    uint64_t * src =
        (uint64_t*)&(rtn_handle->rtn_sq_post_dv[offset].segments[0]);
    ((uint64_t*)current_sq)[0] = cntl_seg;

    for(int i = 1; i < 8; i++)
        ((uint64_t*)current_sq)[i]= src[i];

    //rtn_handle->rtn_sq_post_dv[offset].wqe_idx++ & 0xFFFF;
    rtn_handle->rtn_sq_post_dv[offset].wqe_idx =
        (rtn_handle->rtn_sq_post_dv[offset].wqe_idx + 1) % sq_size;
    rtn_handle->rtn_sq_post_dv[offset].current_sq =
        (rtn_handle->rtn_sq_post_dv[offset].current_sq + 1) % sq_size;
#endif

}

/*
void rtn_poll_cq_dv(RTNGlobalHandle* rtn, int wg_idx, int pe_idx)
{
    struct rtn_handle * rtn_handle = (struct rtn_handle *) rtn;
#ifndef _USE_DC_
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
#else
    int offset = wg_idx;
#endif

    uint32_t cq_size = (rtn_handle[offset].rqp->rcq->cq_size);
    rtn_handle->rtn_sq_post_dv[offset].current_cq =
        rtn_handle->rtn_sq_post_dv[offset].current_cq + 1;
    uint32_t current_offset =
        (((uint32_t)rtn_handle->rtn_sq_post_dv[offset].current_cq)) % cq_size;
    uint64_t * current_cq =
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H) +
        (current_offset * 8);

    // *uint64_t val7 = (uint64_t)((uint64_t*)current_cq[7]);

    //if((val7 &0xF000000000000000) == 0xd000000000000000){
        //error in CQ
    //  exit(-1);
    //}
    for(int i=0;i<7;i++)
        current_cq[i] = 0;
    current_cq[7] = 0xF000000000000000;
}
*/

//-----------------------------------------------------------------------------
void PRINT_SQ(RTNGlobalHandle* rtn_handle, int wg_idx, int pe_idx, int wr_idx)
{
//#ifdef DEBUG
#ifndef _USE_DC_
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
#else
    int offset = wg_idx;
#endif
    uint64_t val0 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 0];
    uint64_t val1 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 1];
    uint64_t val2 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 2];
    uint64_t val3 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 3];
    uint64_t val4 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 4];
    uint64_t val5 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 5];
    uint64_t val6 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 6];
    uint64_t val7 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->sq_base_H)[wr_idx * 8 + 7];
    uint64_t val8 = (uint64_t)*(rtn_handle[offset].rqp->dbrec_send_H);
    uint64_t val9 = (uint64_t)*(rtn_handle[offset].rqp->db_H);
    printf("SQ entry <WG %d pe %d idx %d > = 0x%lx 0x%lx 0x%lx 0x%lx "
           "0x%lx 0x%lx 0x%lx 0x%lx dbrec 0x%lx db 0x%lx address "
           "%p \n", wg_idx, pe_idx, wr_idx, val0, val1, val2, val3, val4,
           val5, val6, val7, val8, val9,
           rtn_handle[offset].rqp->dbrec_send_H);
//#endif
}

void PRINT_CQ(RTNGlobalHandle* rtn_handle, int wg_idx, int pe_idx, int cqe_idx)
{
//#ifdef DEBUG
#ifndef _USE_DC_
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
#else
    int offset = wg_idx;
#endif
    uint64_t val0 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 0];
    uint64_t val1 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 1];
    uint64_t val2 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 2];
    uint64_t val3 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 3];
    uint64_t val4 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 4];
    uint64_t val5 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 5];
    uint64_t val6 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 6];
    uint64_t val7 = (uint64_t)
        ((uint64_t*)rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 7];
    uint64_t *val8 = &(((uint64_t*)
        rtn_handle[offset].rqp->rcq->cq_q_H)[cqe_idx * 8 + 0]);
    printf("CQ entry <WG %d, pe %d, cqe %d> = 0x%lx 0x%lx 0x%lx 0x%lx "
           "0x%lx 0x%lx 0x%lx 0x%lx %p\n", wg_idx, pe_idx, cqe_idx, val0,
           val1, val2, val3, val4, val5, val6, val7, val8);
//#endif
}

void PRINT_RTN_HANDLE(RTNGlobalHandle* rtn_handle, int wg_idx, int pe_idx)
{
#ifndef _USE_DC_
    int offset = pe_idx * rtn_handle->num_qps +wg_idx;
#else
    int offset = wg_idx;
#endif
    uint32_t sq_counter = rtn_handle[offset].sq_counter;
    uint32_t cq_counter = rtn_handle[offset].cq_counter;
    void *sq_current = rtn_handle[offset].sq_current;
    void *cq_current = rtn_handle[offset].cq_current;

    printf("RTN_HANDLE info for WG %d are: sq_counter =%d cq_counter=%d "
           "sq_cur=%p cq_cur=%p\n", wg_idx, sq_counter, cq_counter, sq_current,
           cq_current);
}


#ifdef _USE_HDP_MAP_
void  rtn_hdp_inv(RTNGlobalHandle* rtn_handle)
{
    hdp_reg_t * hdp = rtn_handle[0].hdp_regs;
    hdp_read_inv(hdp);
}
#endif

void PRINT_RTN_HDP(RTNGlobalHandle* rtn_handle)
{
    hdp_reg_t * hdp = rtn_handle[0].hdp_regs;
    printf("HDP_MISC_CNTL = %04x\n", *(hdp->cpu_hdp_flush));
}

void PRINT_RTN_QUEUE_STH(RTNGlobalHandle* rtn_handle)
{
#ifdef _RECYCLE_QUEUE_
    for(int i = 0; i < rtn_handle->num_qps; i++) {
        printf("Software WG_ID mapping to CU_ID (%d , %d)\n",
               i, rtn_handle->softohw[i]);
    }
    rtn_num_qp_cu = (rtn_handle->num_qps/rtn_num_cu) > 0 ?
        rtn_handle->num_qps / rtn_num_cu : 1;
    for (int i = 0; i < rtn_num_cu; i++) {
        for (int j = 1; j < rtn_num_qp_cu + 1; j++)
            printf("Mapping Network Queues to CU_ID (cu_id %d, queue %d)\n",
                   i, (rtn_handle->hwtoqueue[i])[j]);
            fflush(stdout);
    }
#endif
}

//----------------------------------------------------------------------------
int rtn_post_send(rtn_qp_t *qp_t, struct ibv_exp_send_wr *wr,
                  struct ibv_exp_send_wr **bad_wr)
{
    int ret = 0;
    assert(qp_t);
    struct rtn_qp  *qp = ( struct rtn_qp *) qp_t;

    assert(qp->qp);
    assert(wr);
    ret = ibv_exp_post_send(qp->qp, wr, bad_wr);
    if (ret) {
        printf("error %d in rtn_post_send %s\n", ret, strerror(errno));
        goto out;
    }
out:
    return ret;
}

//-----------------------------------------------------------------------------

int rtn_cpu_post_wqe(rtn_qp_t *qp_t, void* addr, uint32_t lkey,
                     void* remote_addr, uint32_t rkey, size_t size,
                     struct ibv_ah *ah, int dc_key)
{
    counter_wqe++;
    struct ibv_sge list;
    list.addr = (uintptr_t) addr;
    list.length = size;
    list.lkey = lkey;

    struct ibv_exp_send_wr wr;
    wr.wr_id = (uint64_t) counter_wqe;
    wr.sg_list = &list;
    wr.num_sge = 1;
    wr.exp_opcode = IBV_EXP_WR_RDMA_WRITE;
    wr.exp_send_flags = IBV_EXP_SEND_SIGNALED;
    wr.wr.rdma.remote_addr  = (int64_t) remote_addr;
    wr.wr.rdma.rkey = rkey;

#ifdef _USE_DC_
    wr.dc.ah             = ah;
    wr.dc.dct_number     = 0;//counter_wqe; //1;
    wr.dc.dct_access_key = dc_key;
#endif

    struct ibv_exp_send_wr *bad_ewr;
    return rtn_post_send(qp_t, &wr, &bad_ewr);
}

void rtn_dc_add_info(RTNGlobalHandle *rtn_handle, uint32_t *vec_dct_num,
                     uint16_t *vec_lids, uint32_t *vec_rkey)
{
#ifdef _USE_DC_
    rtn_handle->vec_dct_num = vec_dct_num;
    rtn_handle->vec_lids = vec_lids;
    rtn_handle->vec_rkey = vec_rkey;
#endif
}

//-----------------------------------------------------------------------------

roc_shmem_status_t
rtn_dump_backend_stats(RTNGlobalHandle *handle, uint64_t totalFinalize)
{
#ifdef PROFILE
    int statblocks = handle->num_qps * handle->num_cqs;

    uint64_t cycles_ring_sq_db = 0;
    uint64_t cycles_update_wqe = 0;
    uint64_t cycles_poll_cq = 0;
    uint64_t cycles_next_cq = 0;
    uint64_t cycles_init = handle[statblocks - 1].profiler.getStat(INIT);
    uint64_t cycles_finalize =
        handle[statblocks - 1].profiler.getStat(FINALIZE);

    uint64_t total_rtn_quiet_count = 0;
    uint64_t total_rtn_db_count = 0;
    uint64_t total_rtn_wqe_count = 0;

    for (int i = 0; i < statblocks; i++) {
        cycles_ring_sq_db += handle[i].profiler.getStat(RING_SQ_DB);
        cycles_update_wqe += handle[i].profiler.getStat(UPDATE_WQE);
        cycles_poll_cq += handle[i].profiler.getStat(POLL_CQ);
        cycles_next_cq += handle[i].profiler.getStat(NEXT_CQ);
        total_rtn_quiet_count += handle[i].profiler.getStat(RTN_QUIET_COUNT);
        total_rtn_db_count += handle[i].profiler.getStat(RTN_DB_COUNT);
        total_rtn_wqe_count += handle[i].profiler.getStat(RTN_WQE_COUNT);
    }

    double us_ring_sq_db = (double) cycles_ring_sq_db / gpu_clock_freq_mhz;
    double us_update_wqe = (double) cycles_update_wqe / gpu_clock_freq_mhz;
    double us_poll_cq = (double) cycles_poll_cq / gpu_clock_freq_mhz;
    double us_next_cq = (double) cycles_next_cq / gpu_clock_freq_mhz;
    double us_init = (double) cycles_init / gpu_clock_freq_mhz;
    double us_finalize = (double) cycles_finalize / gpu_clock_freq_mhz;

    const int FIELD_WIDTH = 20;
    const int FLOAT_PRECISION = 2;

    printf("RTN Counts: Internal Quiets %lu DB Rings %lu WQE Posts "
           "%lu\n", total_rtn_quiet_count, total_rtn_db_count,
           total_rtn_wqe_count);

    printf("\n%*s%*s%*s%*s%*s%*s\n", FIELD_WIDTH + 1, "Init (us)",
           FIELD_WIDTH + 1, "Finalize (us)",
           FIELD_WIDTH + 1, "Ring SQ DB (us)",
           FIELD_WIDTH + 1, "Update WQE (us)",
           FIELD_WIDTH + 1, "Poll CQ (us)",
           FIELD_WIDTH + 1, "Next CQ (us)");

    printf("%*.*f %*.*f %*.*f %*.*f %*.*f %*.*f\n",
           FIELD_WIDTH, FLOAT_PRECISION, us_init / totalFinalize,
           FIELD_WIDTH, FLOAT_PRECISION, us_finalize / totalFinalize,
           FIELD_WIDTH, FLOAT_PRECISION, us_ring_sq_db / total_rtn_db_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_update_wqe / total_rtn_wqe_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_poll_cq / total_rtn_quiet_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_next_cq / total_rtn_quiet_count);
#endif
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
rtn_reset_backend_stats(RTNGlobalHandle *rtn_handle)
{
    int statblocks = rtn_handle->num_qps * rtn_handle->num_cqs;

    for (int i = 0; i < statblocks; i++)
        rtn_handle[i].profiler.resetStats();

    return ROC_SHMEM_SUCCESS;
}

uint32_t sizeof_rtn()
{
    return sizeof(QueuePair);
}
