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
#ifndef QUEUE_PAIR_HPP
#define QUEUE_PAIR_HPP

#include "config.h"

#include "thread_policy.hpp"
#include "hdp_policy.hpp"
#include "connection_policy.hpp"
#include "stats.hpp"

#include <infiniband/mlx5dv.h>

class RTNGlobalHandle;
class GPUIBBackend;

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

const int max_nb_atomic = 4096;

struct rtn_atomic_ret_t {
    uint64_t *atomic_base_ptr;
    uint32_t atomic_lkey;
    uint64_t atomic_counter;
};

const int RTN_INLINE_THRESHOLD = 8;

/*
 * A single IB QueuePair (SQ and CQ) that the GPU can use to perform network
 * operations.  The majority of the important ROC_SHMEM operations are
 * performed by this class.
 */
class QueuePair {

    class SegmentBuilder
    {
        const int SEGMENTS_PER_WQE = 4;

        union mlx5_segment {
            mlx5_wqe_ctrl_seg ctrl_seg;
            mlx5_wqe_raddr_seg raddr_seg;
            mlx5_wqe_atomic_seg atomic_seg;
            mlx5_wqe_data_seg data_seg;
            mlx5_wqe_inl_data_seg inl_data_seg;
            mlx5_base_av base_av;
        };

        mlx5_segment *seg_ptr;

      public:
        __device__
        SegmentBuilder(uint64_t wqe_idx, void * base)
          : seg_ptr(&(static_cast<mlx5_segment*>(base))[SEGMENTS_PER_WQE * wqe_idx])
        { }

        __device__ void
        update_cntrl_seg(uint8_t opcode, uint16_t wqe_idx, uint32_t ctrl_qp_sq,
                         uint64_t ctrl_sig, ConnectionImpl &connection_policy);

        __device__ void
        update_connection_seg(int pe, ConnectionImpl &connection_policy);

        __device__ void
        update_atomic_data_seg(uint64_t atomic_data, uint64_t atomic_cmp);

        __device__ void update_rdma_seg(uintptr_t *raddr, uint32_t rkey);

        __device__ void update_inl_data_seg(uintptr_t * laddr, int32_t size);

        __device__ void
        update_data_seg(uintptr_t * laddr, int32_t size, uint32_t lkey);
    };

    /* TODO: Most of these should be private/protected */
  public:

    #ifdef PROFILE
    typedef Stats<GPU_IB_NUM_STATS> GPUIBStats;
    #else
    typedef NullStats<GPU_IB_NUM_STATS> GPUIBStats;
    #endif

    /*
     * Pointer to the hardware doorbell register for the QP.
     */
    uint64_t *db = nullptr;

    /*
     * Base pointer of this QP's SQ
     * TODO: Use the correct struct type for this.
     */
    uint64_t *current_sq = nullptr;

    /*
     * Base pointer of this QP's CQ
     */
    mlx5_cqe64 *current_cq_q = nullptr;

    /*
     * Pointer to the doorbell record for this SQ.
     */
    volatile uint32_t *dbrec_send = nullptr;

    /*
     * Pointer to the doorbell record for the CQ.
     */
    volatile uint32_t *dbrec_cq = nullptr;

    uint32_t *hdp_rkey = nullptr;

    uintptr_t *hdp_address = nullptr;

    HdpPolicy hdp_policy;

    rtn_atomic_ret_t atomic_ret;

    ThreadImpl threadImpl;

    ConnectionImpl connection_policy;

    /*
     * Current index into the SQ (non-modulo size).
     */
    uint32_t sq_counter = 0;

    /*
     * Number of outstanding messages on this QP that need to be completed
     * during a quiet operation.
     */
    uint32_t quiet_counter = 0;

    int num_cqs = 0;

    /*
     * Current index into the SQ (non-module size).
     */
    uint32_t cq_consumer_counter = 0;

    uint16_t cq_log_size = 0;

    uint16_t cq_size = 0;

    uint32_t ctrl_qp_sq = 0;

    uint64_t ctrl_sig = 0;

    uint32_t rkey = 0;

    uint32_t lkey = 0;

    GPUIBStats profiler;

    uint16_t max_nwqe = 0;

    bool sq_overflow = 0;

    /*
     * Pointer to the QP in global memory that this QP is copied from.  When
     * this QP is destroyed, the dynamic (indicies, stats, etc) in the
     * global_qp are updated.
     */
    QueuePair *global_qp = nullptr;

    explicit QueuePair(GPUIBBackend* backend);

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

    __device__ void quiet_internal();

    __device__ void compute_db_val_opcode(uint64_t *db_val, uint16_t dbrec_val,
                                          uint8_t opcode);

    __device__ void ring_doorbell(uint64_t db_val);

    __device__ bool is_cq_owner_sw(mlx5_cqe64 *cq_entry);

    __device__ uint8_t get_cq_error_syndrome(mlx5_cqe64 *cq_entry);

    template <typename T>
    __device__ static void swap_endian_store(T *dst, const T val);

    friend SingleThreadImpl;
    friend MultiThreadImpl;
    friend RCConnectionImpl;
    friend DCConnectionImpl;
};

#endif //QUEUE_PAIR_HPP
