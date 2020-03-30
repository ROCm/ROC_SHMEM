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

#ifndef RTN_INTERNAL_HPP
#define RTN_INTERNAL_HPP

#include "config.h"

#include "rtn.hpp"
#include"hdp_helper.hpp"
#include"rtn_rocm.hpp"
#include "thread_policy.hpp"

//runime parameters
int use_gpu_mem = 0;
int cq_use_gpu_mem = 0;
int sq_use_gpu_mem = 0;
int max_nb_atomic = 4096;
uint64_t counter_wqe =0;

constexpr int RTN_INLINE_THRESHOLD = 8;

const int CQ_ENTRY_BYTES = 64;
const int CQ_HEADER_BYTE_OFFSET = CQ_ENTRY_BYTES - 1;
const int CQ_ERROR_BYTE_OFFSET = 55;

#ifdef _RECYCLE_QUEUE_
#ifdef __HIP_ARCH_GFX900__
constexpr int MAX_NUM_CU = 64;
#endif
int rtn_num_cu = 64;
int rtn_num_qp_cu;
#endif

#define CHECK_HIP(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", \
              hipGetErrorString(error), error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
    }\
}

enum gpu_ib_stats {
    RING_SQ_DB = 0,
    UPDATE_WQE,
    POLL_CQ,
    NEXT_CQ,
    RTN_QUIET_COUNT,
    RTN_DB_COUNT,
    RTN_WQE_COUNT,
    MEM_WAIT,
    INIT,
    FINALIZE,
    GPU_IB_NUM_STATS
};

#ifdef PROFILE
typedef Stats<GPU_IB_NUM_STATS> GPUIBStats;
#else
typedef NullStats<GPU_IB_NUM_STATS> GPUIBStats;
#endif

extern "C" {
    int mlx5dv_init_obj(struct mlx5dv_obj *obj, uint64_t obj_type);
}

enum RTN_OPCODE_TYPE{
    NOP           = 0x0, //NOP - WQE with this opcode creates a completion, but
    SND_INV       = 0x1, // SND_INV - Send with Invalidate
    RDMA_WRITE    = 0x8, // RDMA_Write
    RDMA_WRITE_IM = 0x9, // RDMA_Write_with_Immediate
    SEND          = 0xA, // Send
    SEND_IM       = 0xB, // Send_with_Immediate
    LSO           = 0xE, // LSO - Large Send Offload
                         //    (used also for multi-packetsend WQE)
    WAIT          = 0xF, // WAIT
    RDMA_READ     = 0x10,// RDMA_Read
    ATOMIC_CAS    = 0x11,// Atomic_Compare_and_Swap
    ATOMIC_FAD    = 0x12,// Atomic_Fetch_and_Add
    ATOMIC_MCAS   = 0x14,// Atomic_Masked_Compare_and_Swap -
                         //    (ExtendedAtomic operation)
    ATOMIC_MFAD   = 0x15,// Atomic_Masked_Fetch_and_Add -
                         //    (Extended Atomic operation)
    RECEIVE_EN    = 0x16,// RECEIVE_EN
    SEND_EN       = 0x17,// SEND_EN
    SET_PSV       = 0x20,// SET_PSV
    DUMP          = 0x23,// DUMP
    UMR           = 0x25,// UMR
    RDMA_FENCE    = 0x30,// fence to complete. This is not IB opcode but
                         //   rather RTN_OPCODE
};

struct  rtn_cq {
    volatile uint32_t   *dbrec_cq;
    volatile uint32_t   *dbrec_cq_H;
    void                *cq_q;
    void                *cq_q_H;
    struct              ibv_cq *cq;
    uint16_t            cq_log_size;
    uint16_t            cq_size;
};

struct  rtn_qp {
    uint64_t            *db;
    void                *sq_base;
    volatile uint32_t   *dbrec_send;
    volatile uint32_t   *dbrec_send_H;
    uint64_t            *db_H;
    void                *sq_base_H;
    struct ibv_qp       *qp;
    struct rtn_cq       *rcq;
    uint16_t            max_nwqe;
};

typedef struct rtn_atomic_ret{
    void                *atomic_base_ptr;
    uint32_t            atomic_lkey;
    uint64_t            atomic_counter;
} rtn_atomic_ret_t;

typedef struct rtn_sq_post_dv {
    uint64_t            segments[8];
    uint32_t            current_sq;
    uint16_t            wqe_idx;
} rtn_sq_post_dv_t;

class RTNGlobalHandle {
  public:
    struct rtn_qp       *rqp;
    uint32_t            sq_counter;
    uint32_t            quiet_counter;
    ThreadImpl          threadImpl;
    uint32_t            cq_counter;
    uint64_t            *sq_current;
    uint8_t             *cq_current;
    int                 num_qps;
    int                 num_cqs;
    uint32_t            *hdp_rkey;
    uintptr_t           *hdp_address;
#ifdef _USE_GPU_UPDATE_SQ_
    uint32_t            ctrl_qp_sq;
    uint64_t            ctrl_sig;
    uint32_t            rkey;
    uint32_t            lkey;
#endif
    rtn_atomic_ret_t    *atomic_ret;
    hdp_reg_t           *hdp_regs;
#ifdef _RECYCLE_QUEUE_
    uint32_t            *softohw;
    uint32_t            **hwtoqueue;
    uint32_t            *queueTocken;
#endif
    rtn_sq_post_dv_t    *rtn_sq_post_dv;
#ifdef _USE_DC_
    uint32_t            *vec_dct_num;
    uint32_t            *vec_rkey;
    uint16_t            *vec_lids;
#endif
    bool                sq_overflow;
    GPUIBStats          profiler;
};

class QueuePair {
    /* TODO: Make most of these private */
  public:
    uint64_t            *db;
    uint64_t            *current_sq;
    uint8_t             *current_cq_q;
    volatile uint32_t   *dbrec_send;
    volatile uint32_t   *dbrec_cq;
    uint32_t            *hdp_rkey;
    uintptr_t           *hdp_address;
    hdp_reg_t           hdp_regs;
    rtn_atomic_ret_t    atomic_ret;
    ThreadImpl          threadImpl;
    uint32_t            sq_counter;
    uint32_t            quiet_counter;
    int                 num_cqs;
    uint32_t            cq_consumer_counter;
    uint16_t            cq_log_size;
    uint16_t            cq_size;
#ifdef _USE_GPU_UPDATE_SQ_
    uint32_t            ctrl_qp_sq;
    uint64_t            ctrl_sig;
    uint32_t            rkey;
    uint32_t            lkey;
#endif
    GPUIBStats          profiler;
#ifdef _USE_DC_
    uint32_t            *vec_dct_num;
    uint32_t            *vec_rkey;
    uint16_t            *vec_lids;
#endif
    uint16_t            max_nwqe;
    bool                sq_overflow;
    RTNGlobalHandle     *global_handle;

    __device__ QueuePair(RTNGlobalHandle* rtn_handle, int offset);

    __device__ ~QueuePair();

    __device__ void waitCQSpace(int num_msgs);

    __device__ void put_nbi(void *dest, const void *source, size_t nelems,
                            int pe, bool db_ring);

    __device__ void quiet_single();

    __device__ void fence(int pe);

    __device__ void get_nbi(void *dest, const void *source, size_t nelems,
                            int pe,bool db_ring);

    __device__ int64_t atomic_fetch(void *dest, int64_t value, int64_t cond,
                                    int pe, bool db_ring, uint8_t atomic_op);

    __device__ void atomic_nofetch(void *dest, int64_t value, int64_t cond,
                                   int pe, bool db_ring, uint8_t atomic_op);

  protected:
    __device__ void
    update_posted_wqe_generic(int pe, int32_t size, uintptr_t* laddr,
                              uintptr_t* raddr, uint8_t opcode,
                              int64_t atomic_data, int64_t atomic_cmp,
                              bool ring_db, uint64_t atomic_ret_pos);
    __device__ void
    update_cntrl_seg(uintptr_t *wqe, uint8_t opcode, uint16_t wqe_idx,
                     bool flag, uint32_t ctrl_qp_sq, uint64_t ctrl_sig);

    __device__ void
    update_atomic_data_seg(uintptr_t * base, uint8_t opcode,
                           int64_t atomic_data, int64_t atomic_cmp,
                           uint64_t atomic_ret_pos);
    __device__ void
    update_rdma_seg(uintptr_t * base, uintptr_t * raddr, uint32_t rkey);

    __device__ void
    update_data_seg(uintptr_t * base, uintptr_t * laddr, int32_t size,
                    bool opcode, uint32_t lkey);

    __device__ void quiet_internal();

    __device__ void compute_db_val_opcode(uint64_t *db_val, uint16_t dbrec_val,
                                          uint8_t opcode);

    __device__ void ring_doorbell(uint64_t db_val);

    __device__ bool is_cq_owner_sw(uint8_t *cq_entry);

    __device__ uint8_t get_cq_opcode(uint8_t *cq_entry);

    __device__ uint8_t get_cq_error_syndrome(uint8_t *cq_entry);

    friend SingleThreadImpl;
    friend MultiThreadImpl;
};

__device__ uint32_t lowerID ();
__device__ int wave_SZ();
#endif
