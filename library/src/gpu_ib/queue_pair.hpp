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
     * @brief Constructor.
     *
     * @param[in] backend Backend needed for member access.
     */
    explicit QueuePair(GPUIBBackend* backend);

    /**
     * @brief Destructor.
     */
    __device__ ~QueuePair();

    /**
     * @brief Inspect completion queue and possibly wait for free space.
     *
     * @param[in] num_msgs Number of entries needing space in completion queue.
     */
    __device__ void
    waitCQSpace(int num_msgs);

    /**
     * @brief Inspect send queue and possibly wait for free space.
     *
     * @param[in] num_msgs Number of entries needing space in send queue.
     */
    __device__ void
    waitSQSpace(int num_msgs);

    /**
     * @brief Create and enqueue a non-blocking put work queue entry (wqe).
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     *
     * @param[in] dest Destination address for data transmission.
     * @param[in] source Source address for data transmission.
     * @param[in] nelems Size in bytes of data transmission.
     * @param[in] pe Destination processing element of data transmission.
     * @param[in] db_ring Denotes whether send queue door bell should be rung.
     */
    template <class level>
    __device__ void
    put_nbi(void *dest,
            const void *source,
            size_t nelems,
            int pe,
            bool db_ring);

    /**
     * @brief Create and enqueue a non-blocking put work queue entry (wqe).
     *
     * @note This variant differs from put_nbi by requesting that a completion
     * queue entry is generated in the completion queue.
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     *
     * @param[in] dest Destination address for data transmission.
     * @param[in] source Source address for data transmission.
     * @param[in] nelems Size in bytes of data transmission.
     * @param[in] pe Destination processing element of data transmission.
     * @param[in] db_ring Denotes whether send queue door bell should be rung.
     */
    template <class level>
    __device__ void
    put_nbi_cqe(void *dest,
                const void *source,
                size_t nelems,
                int pe,
                bool db_ring);

    /**
     * @brief Consume a completion queue entry from this queue pair's
     * completion queue.
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     */
    template <class level>
    __device__ void
    quiet_single();

    /**
     * @brief Send a zero-byte read to enforce ordering and then consume
     * a completion queue entry from this queue pair's completion queue.
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     *
     * @param[in] pe Processing element id to send the zero_b_rd.
     */
    template <class level>
    __device__ void
    quiet_single_heavy(int pe);

    /**
     * @brief Create and enqueue a HDP flush work queue entry on the remote PE.
     *
     * @param[in] pe Processing element id to send the HDP flush operation.
     *
     * TODO(@khamidou): does this require a zero_b_rd to enforce write ordering
     * The HDP flush is itself a write. Could this write be reordered with
     * respect to other write on the network and arrive out-of-order?
     */
    __device__ void
    fence(int pe);

    /**
     * @brief Create and enqueue a non-blocking get work queue entry (wqe).
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     *
     * @param[in] dest Destination address for data transmission.
     * @param[in] source Source address for data transmission.
     * @param[in] nelems Size in bytes of data transmission.
     * @param[in] pe Destination processing element of data transmission.
     * @param[in] db_ring Denotes whether send queue door bell should be rung.
     */
    template <class level>
    __device__ void
    get_nbi(void *dest,
            const void *source,
            size_t nelems,
            int pe,
            bool db_ring);

    /**
     * @brief Create and enqueue a non-blocking get work queue entry (wqe).
     *
     * @note This variant differs from get_nbi by requesting that a completion
     * queue entry is generated in the completion queue.
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     *
     * @param[in] dest Destination address for data transmission.
     * @param[in] source Source address for data transmission.
     * @param[in] nelems Size in bytes of data transmission.
     * @param[in] pe Destination processing element of data transmission.
     * @param[in] db_ring Denotes whether send queue door bell should be rung.
     */
    template <class level>
    __device__ void
    get_nbi_cqe(void *dest,
                const void *source,
                size_t nelems,
                int pe,
                bool db_ring);

    /**
     * @brief Create and enqueue a zero-byte read to enforce write ordering.
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     *
     * @param[in] pe Processing element id to send the zero_b_rd.
     */
    template <class level>
    __device__ void
    zero_b_rd(int pe);

    /**
     * @brief Create and enqueue an atomic fetch work queue entry (wqe).
     *
     * @param[in] dest Destination address for data transmission.
     * @param[in] value Data value for the atomic operation.
     * @param[in] cond Used in atomic comparisons.
     * @param[in] pe Destination processing element of data transmission.
     * @param[in] db_ring Denotes whether send queue door bell should be rung.
     * @param[in] atomic_op The atomic operation to perform.
     *
     * @return An atomic value
     */
    __device__ int64_t
    atomic_fetch(void *dest,
                 int64_t value,
                 int64_t cond,
                 int pe,
                 bool db_ring,
                 uint8_t atomic_op);

    /**
     * @brief Create and enqueue an atomic fetch work queue entry (wqe).
     *
     * @param[in] dest Destination address for data transmission.
     * @param[in] value Data value for the atomic operation.
     * @param[in] cond Used in atomic comparisons.
     * @param[in] pe Destination processing element of data transmission.
     * @param[in] db_ring Denotes whether send queue door bell should be rung.
     * @param[in] atomic_op The atomic operation to perform.
     */
    __device__ void
    atomic_nofetch(void *dest,
                   int64_t value,
                   int64_t cond,
                   int pe,
                   bool db_ring,
                   uint8_t atomic_op);

    /**
     * @brief Helper method to set the doorbell's value.
     *
     * @param[in] val Desired value for the doorbell.
     */
    void setDBval(uint64_t val);

 protected:
    /**
     * @brief Helper method to build work requests for the send queue.
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     * @tparam cqe Flag to optionally generate cqes.
     *
     * @param[in] pe Destination processing element of data transmission.
     * @param[in] size Size in bytes of data transmission.
     * @param[in] laddr Local address.
     * @param[in] raddr Remote address.
     * @param[in] opcode Operation to be performed.
     * @param[in] atomic_data An atomic data value to be used.
     * @param[in] atomic_cmp An atomic comparison operation to be performed.
     * @param[in] ring_db Boolean denoting if doorbell should be rung.
     * @param[in] atomic_ret_pos Index into atomic return structure.
     * @param[in] zero_byte_rd Boolean if zero byte read should be used.
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
     * @brief Helper method to drain completion queue entries.
     *
     * @tparam level Implements specific behaviors for thread, warp, block access.
     *
     */
    template <class level>
    __device__ void
    quiet_internal();

    /**
     * @brief Helper method to compute doorbell value opcode which is used to
     * ring the doorbell.
     *
     * @param[in,out] db_val
     * @param[in] dbrec_val
     * @param[in] opcode
     */
    __device__ void
    compute_db_val_opcode(uint64_t *db_val,
                          uint16_t dbrec_val,
                          uint8_t opcode);

    /**
     * @brief Helper method that sets the field in a work queue entry to
     * generate a completion entry in the completion queue.
     *
     * @param num_wqes Number of work entries this completion entry represents.
     */
    __device__ void
    set_completion_flag_on_wqe(int num_wqes);

    /**
     * @brief Helper method to update fields for the work queue entry.
     *
     * @tparam cqe Flag to optionally generate cqes.
     *
     * @note Single variant is meant to be callable by a block leader.
     */
    template <bool cqe>
    __device__ void
    update_wqe_ce_single(int num_wqes);

    /**
     * @brief Helper method to update fields for the work queue entry.
     *
     * @tparam cqe Flag to optionally generate cqes.
     *
     * @note Thread variant is meant to be callable by multiple threads.
     */
    template <bool cqe>
    __device__ void
    update_wqe_ce_thread(int num_wqes);

    /**
     * @brief Helper method to ring the doorbell
     *
     * @param[in] db_val Doorbell value is written by method.
     */
    __device__ void
    ring_doorbell(uint64_t db_val);

    /**
     * @brief Helper method to extract syndrome field from cqe.
     *
     * @param[in] cq_entry Completion queue entry.
     */
    __device__ uint8_t
    get_cq_error_syndrome(mlx5_cqe64 *cq_entry);

 private:
    const int inline_threshold {8};

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
    db_reg_t db {};

    /*
     * Base pointer of this QP's SQ
     * TODO(bpotter): Use the correct struct type for this.
     */
    uint64_t *current_sq {nullptr};
    uint64_t *current_sq_H {nullptr};

    /*
     * Base pointer of this QP's CQ
     */
    mlx5_cqe64 *current_cq_q {nullptr};
    mlx5_cqe64 *current_cq_q_H {nullptr};

    /*
     * Pointer to the doorbell record for this SQ.
     */
    volatile uint32_t *dbrec_send {nullptr};

    /*
     * Pointer to the doorbell record for the CQ.
     */
    volatile uint32_t *dbrec_cq {nullptr};

    uint32_t *hdp_rkey {nullptr};

    uintptr_t *hdp_address {nullptr};

    HdpPolicy hdp_policy {};

    atomic_ret_t atomic_ret {};

    ThreadImpl threadImpl {};

    ConnectionImpl connection_policy;

    char * const* base_heap {nullptr};
    /*
     * Current index into the SQ (non-modulo size).
     */
    uint32_t sq_counter {0};
    uint32_t local_sq_cnt {0};

    /*
     * Number of outstanding messages on this QP that need to be completed
     * during a quiet operation.
     */
    uint32_t quiet_counter {0};

    int num_cqs {0};

    /*
     * Current index into the SQ (non-module size).
     */
    uint32_t cq_consumer_counter {0};

    uint16_t cq_log_size {0};

    uint16_t cq_size {0};

    uint32_t ctrl_qp_sq {0};

    uint64_t ctrl_sig {0};

    uint32_t rkey {0};

    uint32_t lkey {0};

    GPUIBStats profiler {};

    uint16_t max_nwqe {0};

    bool sq_overflow {0};

    uint64_t db_val {};
    /*
     * Pointer to the QP in global memory that this QP is copied from.  When
     * this QP is destroyed, the dynamic (indicies, stats, etc) in the
     * global_qp are updated.
     */
    QueuePair *global_qp {nullptr};

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
