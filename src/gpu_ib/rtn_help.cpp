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

#include "config.h"

#include <unistd.h>
#include <string.h>
#include <assert.h>

#include "rtn.hpp"
#include "rtn_internal.hpp"

static ibv_exp_peer_direct_attr peers_attr[32];

const size_t RTN_MAX_INLINE_SIZE = 128; // need to check what is the exact size

//-----------------------------------------------------------------------------

static struct ibv_exp_peer_buf *rtn_buf_alloc(ibv_exp_peer_buf_alloc_attr *attr)
{
    assert(attr);
    ibv_exp_peer_buf * peer_buf =
        (ibv_exp_peer_buf*) malloc(sizeof(ibv_exp_peer_buf));

    if (peer_buf == NULL) {
        printf("error, could not allocate memory \n");
    }

    peer_buf->comp_mask = 0;
    peer_buf->length = attr->length;

    if (use_gpu_mem == 1) {
        void * dev_ptr;
        CHECK_HIP(hipSetDevice( attr->peer_id));
        //CHECK_HIP(hipMalloc((void**)&dev_ptr, attr->length));
        CHECK_HIP(hipExtMallocWithFlags((void**)&dev_ptr, attr->length,
                                        hipDeviceMallocFinegrained));
        peer_buf->addr = dev_ptr;
    } else {
        free(peer_buf);
        return NULL;
    }
    return peer_buf;
}

static int rtn_buf_release(struct ibv_exp_peer_buf *pb)
{
    assert(pb);
    free(pb->addr);
    return 0;
}

static uint64_t rtn_register_va(void *start, size_t length, uint64_t rtn_id,
                                struct ibv_exp_peer_buf *pb)
{
    CHECK_HIP(hipSetDevice(rtn_id));
    if (use_gpu_mem == 1) {
        if (pb == IBV_EXP_PEER_IOMEMORY) {
            void * gpu_ptr= NULL;
            //rtn_rocm_memory_lock(start, length, &gpu_ptr, rtn_id);
            rtn_rocm_memory_lock_to_fine_grain(start, length, &gpu_ptr,
                                               rtn_id);
            // we could use this map to keep track of DBs and not call lock
            // twice
            //rtn_db_insert (&rtn_db_head, (uintptr_t*) start,
            //               (uintptr_t*)gpu_ptr);

        } else if(pb == NULL) {
            void *gpu_ptr = NULL;
            //CHECK_HIP(hipHostRegister((void*)start, length,
            //                  hipHostRegisterMapped));

            rtn_rocm_memory_lock_to_fine_grain(start, length, &gpu_ptr,
                                               rtn_id);
        }
        return (uint64_t)(start);
     } else {
        if (pb == IBV_EXP_PEER_IOMEMORY) {
            void * gpu_ptr= NULL;
            //rtn_rocm_memory_lock(start, length, &gpu_ptr, rtn_id);
            rtn_rocm_memory_lock_to_fine_grain(start, length, &gpu_ptr,
                                               rtn_id);
            // we could use this map to keep track of DBs and not call lock
            // twice
            //rtn_db_insert(&rtn_db_head, (uintptr_t*) start,
            //              (uintptr_t*)gpu_ptr);
        } else{
            void * gpu_ptr = NULL;
            //CHECK_HIP(hipHostRegister((void*)start, length,
            //       hipHostRegisterMapped));
            rtn_rocm_memory_lock_to_fine_grain(start, length, &gpu_ptr,
                                               rtn_id);
        }
        return (uint64_t)(start);
    }
    //return (uint64_t)(start);
}

static int rtn_unregister_va(uint64_t target_id, uint64_t rtn_id)
{
    printf("calling unregister \n");
    CHECK_HIP(hipSetDevice(rtn_id));
    CHECK_HIP(hipHostUnregister ( (void*) target_id ));
    return 0;
}

static void rtn_init_peer_attr(ibv_exp_peer_direct_attr *attr1, int rtn_id)
{
    // need to cache this for better perf
    // ibv_exp_peer_direct_attr* attr1 = &peers_attr[rtn_id];

    attr1->peer_id = rtn_id;
    attr1->buf_alloc = rtn_buf_alloc;
    attr1->buf_release = rtn_buf_release;
    attr1->register_va = rtn_register_va;
    attr1->unregister_va = rtn_unregister_va;

    attr1->caps = (IBV_EXP_PEER_OP_STORE_DWORD_CAP    |
                   IBV_EXP_PEER_OP_STORE_QWORD_CAP    |
                   IBV_EXP_PEER_OP_FENCE_CAP          |
                   IBV_EXP_PEER_OP_POLL_AND_DWORD_CAP |
                   IBV_EXP_PEER_OP_POLL_GEQ_DWORD_CAP);

    attr1->peer_dma_op_map_len = RTN_MAX_INLINE_SIZE;
    attr1->comp_mask = IBV_EXP_PEER_DIRECT_VERSION;
    attr1->version = 1; // EXP verbs requires to be set to 1
}

//-----------------------------------------------------------------------------

/* ibv_cq* rtn_get_cq(rtn_cq_t *rcq_t)
{
    struct rtn_cq *rcq = (struct rtn_cq*) rcq_t;
    return rcq->cq;
}
*/

//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

rtn_cq_t *
rtn_create_cq(struct ibv_context *context, int cqe,
              void *cq_context, struct ibv_comp_channel *channel,
              int comp_vector, int rtn_id)
{
    if (cq_use_gpu_mem == 1) {
        use_gpu_mem = 1;
    } else {
        use_gpu_mem = 0;
    }

    struct ibv_cq *cq = NULL;
    struct rtn_cq *rcq = NULL;

    CHECK_HIP(hipMalloc(&rcq, sizeof(struct rtn_cq)));

    ibv_exp_peer_direct_attr* peer_attr = &peers_attr[rtn_id]; //NULL;

    rtn_init_peer_attr(peer_attr,  rtn_id);

    ibv_exp_cq_init_attr attr;
    memset(&attr, 0, sizeof(ibv_exp_cq_init_attr));
    attr.comp_mask = IBV_EXP_CQ_INIT_ATTR_PEER_DIRECT;
    attr.flags = 0; // see ibv_exp_cq_create_flags
    attr.res_domain = NULL;
    attr.peer_direct_attrs = peer_attr;

    cq = ibv_exp_create_cq(context, cqe, cq_context, channel, comp_vector,
                           &attr);
    if (!cq) {
        printf("error in ibv_exp_create_cq, %d  %s\n", errno, strerror(errno));
    }
    rcq->cq = cq;
    return (rtn_cq_t*) rcq;
}

//-----------------------------------------------------------------------------
void rtn_get_cq_q(struct rtn_cq *rcq)
{
    if (cq_use_gpu_mem == 1) {
        use_gpu_mem = 1;
    } else {
        use_gpu_mem = 0;
    }

    assert(rcq);
    struct mlx5dv_cq cq_out;
    struct mlx5dv_obj mlx_obj;
    mlx_obj.cq.in = rcq->cq;
    mlx_obj.cq.out = &cq_out;
    void *gpu_ptr = NULL;

    mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_CQ);
    rcq->cq_log_size = log2(cq_out.cqe_cnt);
    rcq->cq_size = cq_out.cqe_cnt;

    if (use_gpu_mem == 1) {
        rcq->cq_q   = (void*) cq_out.buf;
        rcq->cq_q_H = (void*) cq_out.buf;
    } else {
        //CHECK_HIP(hipHostGetDevicePointer( &gpu_ptr, (void*)cq_out.buf, 0));
        rtn_rocm_memory_lock_to_fine_grain((void*)cq_out.buf,
                                            cq_out.cqe_cnt * 64, &gpu_ptr, 0);

        rcq->cq_q   = gpu_ptr;
        rcq->cq_q_H = (void*)cq_out.buf;
    }

    //CHECK_HIP(hipHostRegister((void*)cq_out.dbrec, 64,
    //                hipHostRegisterMapped));
    //CHECK_HIP(hipHostGetDevicePointer( &gpu_ptr, (void*)cq_out.dbrec, 0));

    rtn_rocm_memory_lock_to_fine_grain((void*)cq_out.dbrec, 64, &gpu_ptr, 0);

    rcq->dbrec_cq = (volatile uint32_t*)gpu_ptr;
    rcq->dbrec_cq_H = (volatile uint32_t*)cq_out.dbrec;
}


//-----------------------------------------------------------------------------
void rtn_get_sq(struct rtn_qp *rqp, int rtn_id)
{
    if (sq_use_gpu_mem == 1) {
        use_gpu_mem = 1;
    } else {
        use_gpu_mem = 0;
    }

    assert(rqp);
    struct mlx5dv_qp qp_out;
    struct mlx5dv_obj mlx_obj;
    mlx_obj.qp.in = rqp->qp;
    mlx_obj.qp.out = &qp_out;

    mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);

    rqp->max_nwqe = (qp_out.sq.wqe_cnt);


    void * gpu_ptr = NULL;
    volatile uint32_t *dbrec_send = qp_out.dbrec + 1;

    if (use_gpu_mem == 1) {
        rqp->sq_base        =  (void*)qp_out.sq.buf;
        rqp->sq_base_H      =  (void*)qp_out.sq.buf;
        rqp->dbrec_send     =  (volatile uint32_t*)dbrec_send;
        rqp->dbrec_send_H   =  (volatile uint32_t*)dbrec_send;

    } else {
        //CHECK_HIP(hipHostGetDevicePointer(&gpu_ptr,
        //                                  (void*)qp_out.sq.buf, 0));
        rtn_rocm_memory_lock_to_fine_grain((void*)qp_out.sq.buf,
            qp_out.sq.wqe_cnt *64, &gpu_ptr, rtn_id);

        rqp->sq_base   =  gpu_ptr;
        rqp->sq_base_H =  (void*)qp_out.sq.buf;

        //CHECK_HIP(hipHostGetDevicePointer( &gpu_ptr, (void*)dbrec_send, 0));

        rtn_rocm_memory_lock_to_fine_grain((void*)dbrec_send, 32, &gpu_ptr,
                                           rtn_id);
        rqp->dbrec_send     =  (volatile uint32_t*)gpu_ptr;
        rqp->dbrec_send_H   =  (volatile uint32_t*)dbrec_send;

    }
    //rtn_rocm_memory_lock(qp_out.bf.reg, qp_out.bf.size, &gpu_ptr, rtn_id);
    rtn_rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size,
                                       &gpu_ptr, rtn_id);
    // we could use this map table to track the DBs
    //gpu_ptr =  rtn_db_lookup(rtn_db_head, (uintptr_t*) qp_out.bf.reg);
    rqp->db = (uint64_t*)gpu_ptr;
    rqp->db_H = (uint64_t*)qp_out.bf.reg;

}
//-----------------------------------------------------------------------------
/*ibv_qp* rtn_get_qp (rtn_qp_t *rqp_t)
{
    struct rtn_qp * rqp = (struct rtn_qp*) rqp_t;
    return rqp->qp;
}*/
//-----------------------------------------------------------------------------

rtn_qp_t *
rtn_create_qp(struct ibv_pd *pd, struct ibv_context *context,
              ibv_exp_qp_init_attr *qp_attr, rtn_cq_t * rcq_t, int rtn_id,
              int qp_idx, int pe_idx, RTNGlobalHandle *rtn_handle)
{
    if (sq_use_gpu_mem == 1) {
        use_gpu_mem = 1;
    } else {
        use_gpu_mem = 0;
    }

    int ret = 0;
    struct rtn_qp *rqp = NULL;
    struct ibv_qp *qp = NULL;
    struct ibv_cq *tx_cq = NULL;
    ibv_exp_peer_direct_attr *peer_attr =  &peers_attr[rtn_id];

    assert(pd);
    assert(context);
    assert(qp_attr);
    assert(rtn_handle);

    struct rtn_cq *rcq = (struct rtn_cq*) rcq_t;

    rtn_init_peer_attr(peer_attr, rtn_id);

    CHECK_HIP(hipMalloc(&rqp, sizeof(struct rtn_qp)));

    qp_attr->send_cq = rcq->cq;
    qp_attr->recv_cq = rcq->cq;
    //qp_attr->srq = NULL;

    qp_attr->pd = pd;
    qp_attr->comp_mask |= IBV_EXP_QP_INIT_ATTR_PD;
    qp_attr->comp_mask |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
    qp_attr->exp_create_flags |= IBV_EXP_QP_CREATE_IGNORE_SQ_OVERFLOW ;

    qp_attr->comp_mask |= IBV_EXP_QP_INIT_ATTR_PEER_DIRECT;
    qp_attr->peer_direct_attrs = peer_attr;

    qp = ibv_exp_create_qp(context, qp_attr);

    int offset = rtn_handle->num_qps *pe_idx + qp_idx;

    if (!qp) {
        ret = EINVAL;
        printf("error ibv_exp_create_qp failed %d \n", errno);
        goto err;
    }

    rqp->qp  = qp;
    rqp->rcq = rcq;
    rtn_get_sq(rqp, rtn_id);
    rtn_get_cq_q(rcq);
    //if(rank ==0)
    //printf("QP creation for pe %d wg %d qp address %llx cq address %llx \n",
    //       pe_idx, qp_idx, rqp->sq_base_H, rcq->cq_q_H);
    //fflush(stdout);


    rtn_handle[offset].rqp  =  rqp;

    rtn_handle[offset].sq_current = (uint64_t *) rqp->sq_base;
    rtn_handle[offset].cq_current = (uint8_t *) rqp->rcq->cq_q;

    return (rtn_qp_t *) rqp;

err:
    ibv_destroy_qp(qp);
    ret = ibv_destroy_cq(tx_cq);
    CHECK_HIP(hipFree(rqp));
    return NULL;
}

//-----------------------------------------------------------------------------

int rtn_destroy_qp(rtn_qp_t *rqp_t)
{
    int retcode = 0;
    int ret;
    assert(rqp_t);
    struct rtn_qp *rqp = (struct rtn_qp*) rqp_t;
    assert(rqp->qp);

    ret = ibv_destroy_qp(rqp->qp);
    if (ret) {
        printf("error %d in destroy_qp\n", ret);
        retcode = ret;
    }

    CHECK_HIP(hipFree(rqp));
    return retcode;
}

int rtn_destroy_cq(rtn_cq_t *rcq_t)
{

    int retcode = 0;
    int ret;
    assert(rcq_t);
    struct rtn_cq *rcq = (struct rtn_cq*) rcq_t;
    assert(rcq->cq);

    ret = ibv_destroy_cq(rcq->cq);
    if (ret) {
        printf("error %d in rtn_destroy_cq\n", ret);
        retcode = ret;
    }
    CHECK_HIP(hipFree(rcq));

    return retcode;
}
