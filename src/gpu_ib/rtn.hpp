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

#ifndef RTN_HPP
#define RTN_HPP

#include "config.h"

//#define _USE_HDP_MAP_ 1

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>
extern "C"{
#include <infiniband/mlx5dv.h>
}

#include "util.hpp"

typedef uint64_t rtn_cq_t;
typedef uint64_t rtn_qp_t;

class RTNGlobalHandle;

rtn_cq_t* rtn_create_cq(struct ibv_context *context, int cqe,
              void *cq_context, struct ibv_comp_channel *channel,
              int comp_vector, int rtn_id);

rtn_qp_t* rtn_create_qp(struct ibv_pd *pd, struct ibv_context *context,
                        ibv_exp_qp_init_attr *qp_attr, rtn_cq_t* rcq,
                        int rtn_id, int qp_idx, int pe_idx,
                        RTNGlobalHandle *rtn_handle);

ibv_qp* rtn_get_qp(rtn_qp_t *rqp);
ibv_cq* rtn_get_cq(rtn_cq_t *rcq);
rtn_qp_t* rtn_get_rqp(RTNGlobalHandle *rtn_handle, int wg_idx, int wr_idx);
rtn_cq_t* rtn_get_rcq(RTNGlobalHandle *rtn_handle, int wg_idx, int wr_idx);
uint32_t rtn_get_sq_counter(rtn_qp_t *rqp_t);
uint32_t rtn_get_cq_counter(rtn_cq_t *rcq_t);
uint32_t rtn_get_hdp_rkey(RTNGlobalHandle *rtn_handle);
uintptr_t rtn_get_hdp_address(RTNGlobalHandle *rtn_handle);
void rtn_hdp_add_info(RTNGlobalHandle *rtn_handle, uint32_t *vec_rkey,
                      uintptr_t*vec_address);

void rtn_dc_add_info(RTNGlobalHandle *rtn_handle_e, uint32_t *vec_dct_num,
                     uint16_t *vec_lids, uint32_t *vec_rkey);

void rtn_set_sq_dv(RTNGlobalHandle *handle, int wg_idx, int pe_idx);
#ifndef _USE_GPU_UPDATE_SQ_
void rtn_post_wqe_dv(RTNGlobalHandle *handle, int wg_idx, int pe_idx,
                     bool flag);
#endif

int rtn_destroy_qp(rtn_qp_t *rqp);
int rtn_destroy_cq(rtn_cq_t *rcq);


RTNGlobalHandle* rtn_init(int rtn_id, int num_qps, int remote_conn,
                       struct ibv_pd *pd);

int rtn_finalize(RTNGlobalHandle *rtn_handle, int rtn_id);

int rtn_cpu_post_wqe(rtn_qp_t *qp_t, void* addr, uint32_t lkey,
                     void* remote_addr, uint32_t rkey, size_t size,
                     struct ibv_ah *ah, int dc_key);


int rtn_post_send(rtn_qp_t *qp_t, struct ibv_exp_send_wr *wr,
                  struct ibv_exp_send_wr **bad_wr);

__device__ void  rtn_init(RTNGlobalHandle *rtn_handle,
                          QueuePair *rtn_gpu_handle,
                          int my_pe, int comm_size);

__device__ void  rtn_finalize (RTNGlobalHandle *rtn_handle,
                               QueuePair *rtn_gpu_handle,
                               int my_pe, int comm_size);

void PRINT_SQ(RTNGlobalHandle* rtn, int wg_num, int pe_idx, int wr_idx);

void PRINT_CQ(RTNGlobalHandle* rtn, int wg_num, int pe_idx, int cqe_idx);

void PRINT_RTN_QUEUE_STH(RTNGlobalHandle* rtn);

void PRINT_RTN_HANDLE(RTNGlobalHandle* rtn, int wg_idx, int pe_idx);
void PRINT_RTN_HDP(RTNGlobalHandle* rtn);
void rtn_hdp_inv(RTNGlobalHandle* rtn);

uint32_t sizeof_rtn();

roc_shmem_status_t rtn_reset_backend_stats(RTNGlobalHandle *rtn_handle_e);
roc_shmem_status_t rtn_dump_backend_stats(RTNGlobalHandle *handle,
                                          uint64_t numFinalize);

#endif
