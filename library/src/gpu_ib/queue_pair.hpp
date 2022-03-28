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

#ifndef ROCSHMEM_LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP
#define ROCSHMEM_LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP

/**
 * @file queue_pair.hpp
 *
 * @section DESCRIPTION
 * An IB QueuePair (SQ and CQ) that the device can use to perform network
 * operations. Most important ROC_SHMEM operations are performed by this
 * class.
 */

#include <infiniband/mlx5dv.h>

#include "../atomic_return.hpp"
#include "config.h"  // NOLINT(build/include_subdir)
#include "connection_policy.hpp"
#include "hdp_policy.hpp"
#include "stats.hpp"
#include "thread_policy.hpp"

namespace rocshmem {

class GPUIBBackend;

enum gpu_ib_stats {
    RING_SQ_DB = 0,
    UPDATE_WQE,
    POLL_CQ,
    NEXT_CQ,
    QUIET_COUNT,
    DB_COUNT,
    WQE_COUNT,
    MEM_WAIT,
    INIT,
    FINALIZE,
    GPU_IB_NUM_STATS
};

typedef union db_reg {
    uint64_t *ptr;
    uintptr_t uint;
} db_reg_t;

class QueuePair {
 public:
    /**
     * TODO(bpotter): document
     */
    explicit QueuePair(GPUIBBackend* backend);

    /**
     * TODO(bpotter): document
     */
    __device__ ~QueuePair();

    /**
     * TODO(bpotter): document
     */
    __device__ void
    waitCQSpace(int num_msgs);

    /**
     * TODO(bpotter): document
     */
    __device__ void
    waitSQSpace(int num_msgs);

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    put_nbi(void *dest,
            const void *source,
            size_t nelems,
            int pe,
            bool db_ring);

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    put_nbi_cqe(void *dest,
            const void *source,
            size_t nelems,
            int pe,
            bool db_ring);

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    quiet_single();

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    quiet_single_heavy(int pe);

    /**
     * TODO(bpotter): document
     */
    __device__ void
    fence(int pe);

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    get_nbi(void *dest,
            const void *source,
            size_t nelems,
            int pe,
            bool db_ring);

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    get_nbi_cqe(void *dest,
            const void *source,
            size_t nelems,
            int pe,
            bool db_ring);

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    zero_b_rd(int pe);

    /**
     * TODO(bpotter): document
     */
    __device__ int64_t
    atomic_fetch(void *dest,
                 int64_t value,
                 int64_t cond,
                 int pe,
                 bool db_ring,
                 uint8_t atomic_op);

    /**
     * TODO(bpotter): document
     */
    __device__ void
    atomic_nofetch(void *dest,
                   int64_t value,
                   int64_t cond,
                   int pe,
                   bool db_ring,
                   uint8_t atomic_op);

    /**
     * TODO(bpotter): document
     */
    void setDBval(uint64_t val);

 protected:
    /**
     * TODO(bpotter): document
     */
    template <class level, bool cqe>
    __device__ void
    update_posted_wqe_generic(int pe,
                              int32_t size,
                              uintptr_t *laddr,
                              uintptr_t *raddr,
                              uint8_t opcode,
                              int64_t atomic_data,
                              int64_t atomic_cmp,
                              bool ring_db,
                              uint64_t atomic_ret_pos,
                              bool zero_byte_rd = false);

    /**
     * TODO(bpotter): document
     */
    template <class level>
    __device__ void
    quiet_internal();

    /**
     * TODO(bpotter): document
     */
    __device__ void
    compute_db_val_opcode(uint64_t *db_val,
                          uint16_t dbrec_val,
                          uint8_t opcode);

    /**
     * TODO(bpotter): document
     */
    __device__ void
    set_completion_flag_on_wqe(int num_wqes);

    /**
     * TODO(bpotter): document
     */
    template<bool cqe>
    __device__ void update_wqe_ce_single(int num_wqes);

    /**
     * TODO(bpotter): document
     */
    template<bool cqe>
    __device__ void update_wqe_ce_thread(int num_wqes);

    /**
     * TODO(bpotter): document
     */
    __device__ void
    ring_doorbell(uint64_t db_val);

    /**
     * TODO(bpotter): document
     */
    __device__ bool
    is_cq_owner_sw(mlx5_cqe64 *cq_entry);

    /**
     * TODO(bpotter): document
     */
    __device__ uint8_t
    get_cq_error_syndrome(mlx5_cqe64 *cq_entry);

 private:
    const int inline_threshold = 8;

    /* TODO(bpotter): Most of these should be private/protected */
 public:
    #ifdef PROFILE
    typedef Stats<GPU_IB_NUM_STATS> GPUIBStats;
    #else
    typedef NullStats<GPU_IB_NUM_STATS> GPUIBStats;
    #endif

    /*
     * Pointer to the hardware doorbell register for the QP.
     */
    db_reg_t db;

    /*
     * Base pointer of this QP's SQ
     * TODO(bpotter): Use the correct struct type for this.
     */
    uint64_t *current_sq = nullptr;
    uint64_t *current_sq_H = nullptr;

    /*
     * Base pointer of this QP's CQ
     */
    mlx5_cqe64 *current_cq_q = nullptr;
    mlx5_cqe64 *current_cq_q_H = nullptr;

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

    atomic_ret_t atomic_ret;

    ThreadImpl threadImpl;

    ConnectionImpl connection_policy;

    char * const* base_heap {nullptr};
    /*
     * Current index into the SQ (non-modulo size).
     */
    uint32_t sq_counter = 0;
    uint32_t local_sq_cnt = 0;

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

    uint64_t db_val;
    /*
     * Pointer to the QP in global memory that this QP is copied from.  When
     * this QP is destroyed, the dynamic (indicies, stats, etc) in the
     * global_qp are updated.
     */
    QueuePair *global_qp = nullptr;

    friend SingleThreadImpl;
    friend MultiThreadImpl;
    friend THREAD;
    friend WG;
    friend WAVE;
    friend RCConnectionImpl;
    friend DCConnectionImpl;
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_GPU_IB_QUEUE_PAIR_HPP
