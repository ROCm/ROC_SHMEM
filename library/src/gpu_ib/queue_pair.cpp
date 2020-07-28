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

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include <hip/hip_runtime.h>

#include <stdio.h>

#include "queue_pair.hpp"
#include "util.hpp"
#include "backend.hpp"

template <typename T> __device__ void
QueuePair::swap_endian_store(T *dst, const T val)
{
    union U {
        T val;
        uint8_t bytes[sizeof(T)];
    } src, dst_tmp;

    src.val = val;

    /*
     * This does not generate optimal bit twiddling on the GPU, so specialize
     * for the primitive types below to eek out a bit more perf while keeping
     * this for all other types we might throw at it.
     */
    std::reverse_copy(src.bytes, src.bytes + sizeof(T), dst_tmp.bytes);

    *dst = dst_tmp.val;
}

template <> __device__ void
QueuePair::swap_endian_store(uint64_t *dst, const uint64_t val)
{
    uint64_t new_val = ((val << 8) & 0xFF00FF00FF00FF00ULL) |
            ((val >> 8) & 0x00FF00FF00FF00FFULL);

    new_val = ((new_val << 16) & 0xFFFF0000FFFF0000ULL) |
            ((new_val >> 16) & 0x0000FFFF0000FFFFULL);

    *dst = (new_val << 32) | (new_val >> 32);
}

template <> __device__ void
QueuePair::swap_endian_store(uint32_t *dst, const uint32_t val)
{
    uint32_t new_val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF);
    *dst = (new_val << 16) | (new_val >> 16);
}

template <> __device__ void
QueuePair::swap_endian_store(uint16_t *dst, const uint16_t val)
{ *dst = ((val << 8) & 0xFF00 ) | ((val >> 8) & 0x00FF); }

template <> __device__ void
QueuePair::swap_endian_store(int64_t *dst, const int64_t val)
{ swap_endian_store((uint64_t *) dst, (const uint64_t) val); }

template <> __device__ void
QueuePair::swap_endian_store(int32_t *dst, const int32_t val)
{ swap_endian_store((uint32_t *) dst, (const uint32_t) val); }

template <> __device__ void
QueuePair::swap_endian_store(int16_t *dst, const int16_t val)
{ swap_endian_store((uint16_t * ) dst, (const uint16_t) val); }

__device__ bool
QueuePair::is_cq_owner_sw(mlx5_cqe64 *cqe_entry)
{
    return mlx5dv_get_cqe_owner(cqe_entry) ==
        ((cq_consumer_counter >> cq_log_size) & 1);
}

__device__ uint8_t
QueuePair::get_cq_error_syndrome(mlx5_cqe64 *cqe_entry)
{
    return cqe_entry->sop_qpn.sop;
}

__device__ void
QueuePair::ring_doorbell(uint64_t db_val)
{
    GPU_DPRINTF("*** Ringing DB @%p: dbrec %p db_val %llu counter %d\n",
                db, (uint32_t*) dbrec_send, db_val, sq_counter);

    swap_endian_store((uint32_t*) dbrec_send, (uint32_t) sq_counter);

    *db = db_val;
}

QueuePair::QueuePair(GPUIBBackend *backend)
    : hdp_policy(*backend->hdp_policy),
      connection_policy(*backend->connection_policy)
{
    hdp_rkey = backend->hdp_rkey;
    hdp_address = backend->hdp_address;

    atomic_ret.atomic_lkey = backend->atomic_ret->atomic_lkey;
    atomic_ret.atomic_counter = 0;
}

__device__
QueuePair::~QueuePair()
{
    uint64_t start = profiler.startTimer();

    global_qp->sq_counter = sq_counter;
    global_qp->cq_consumer_counter = cq_consumer_counter;
    global_qp->current_sq = current_sq;
    global_qp->current_cq_q = current_cq_q;
    global_qp->sq_overflow = sq_overflow;
    global_qp->quiet_counter = quiet_counter;
    profiler.endTimer(start, FINALIZE);

    global_qp->profiler.accumulateStats(profiler);

    __syncthreads();
}

__device__ void
QueuePair::compute_db_val_opcode(uint64_t *db_val, uint16_t dbrec_val,
                                 uint8_t opcode)
{
    uint64_t opcode64 = opcode;
    opcode64 = opcode64 << 24 & 0x000000FFFF000000;
    uint64_t dbrec = dbrec_val << 8;
    dbrec = dbrec & 0x0000000000FFFF00;
    uint64_t val = *db_val;
    val = val & 0xFFFFFFFFFF0000FF;
    *db_val = val | dbrec | opcode64;
}

__device__ void
QueuePair::SegmentBuilder::
update_cntrl_seg(uint8_t opcode, uint16_t wqe_idx, uint32_t ctrl_qp_sq,
                 uint64_t ctrl_sig, ConnectionImpl &connection_policy)
{
    mlx5_wqe_ctrl_seg * ctrl_seg = &seg_ptr->ctrl_seg;
    ctrl_seg->opmod_idx_opcode = (opcode << 24) | (wqe_idx << 8);

    uint32_t DS = (opcode == MLX5_OPCODE_RDMA_WRITE ||
                   opcode == MLX5_OPCODE_RDMA_READ) ? 3 : 4;

    DS += connection_policy.wqeCntrlOffset();
    ctrl_seg->qpn_ds = (DS << 24) | ctrl_qp_sq;
    ctrl_seg->signature = ctrl_sig;
    ctrl_seg->fm_ce_se = ctrl_sig >> 24;
    ctrl_seg->imm = ctrl_sig >> 32;
    seg_ptr++;
}

__device__ void
QueuePair::SegmentBuilder::
update_atomic_data_seg(uint64_t atomic_data, uint64_t atomic_cmp)
{
    mlx5_wqe_atomic_seg * atomic_seg = &seg_ptr->atomic_seg;
    swap_endian_store(&atomic_seg->swap_add, atomic_data);
    swap_endian_store(&atomic_seg->compare, atomic_cmp);
    seg_ptr++;
}

__device__ void
QueuePair::SegmentBuilder::update_rdma_seg(uintptr_t * raddr, uint32_t rkey)
{
    mlx5_wqe_raddr_seg * raddr_seg = &seg_ptr->raddr_seg;
    swap_endian_store(&raddr_seg->raddr, (uint64_t) raddr);
    raddr_seg->rkey = rkey;
    seg_ptr++;
}

__device__ void
QueuePair::SegmentBuilder::
update_data_seg(uintptr_t * laddr, int32_t size, uint32_t lkey)
{
    mlx5_wqe_data_seg * data_seg = &seg_ptr->data_seg;
    swap_endian_store(&data_seg->byte_count, size & 0x7FFFFFFFU);
    data_seg->lkey = lkey;
    swap_endian_store(&data_seg->addr, (uint64_t) laddr);
    seg_ptr++;
}

__device__ void
QueuePair::SegmentBuilder::update_inl_data_seg(uintptr_t * laddr, int32_t size)
{
    mlx5_wqe_inl_data_seg * inl_data_seg = &seg_ptr->inl_data_seg;
    swap_endian_store(&inl_data_seg->byte_count,
                      (size & 0x3FF) | 0x80000000);

    // Assume fence HDP flush
    // TODO: Rework fence interface to avoid this
    if (!laddr) {
        uint8_t flush_val = 1;
        memcpy(inl_data_seg + 1, &flush_val, sizeof(flush_val));
    } else {
        memcpy(inl_data_seg + 1, laddr, size);
    }

    seg_ptr++;
}

__device__ void
QueuePair::SegmentBuilder::
update_connection_seg(int pe, ConnectionImpl &connection_policy)
{
    if (connection_policy.updateConnectionSegmentImpl(&seg_ptr->base_av, pe))
        seg_ptr++;
}

__device__ void
QueuePair::quiet_internal()
{
    if (!quiet_counter) return;

    profiler.incStat(RTN_QUIET_COUNT);

    uint64_t start = profiler.startTimer();

    cq_consumer_counter = cq_consumer_counter + quiet_counter - 1;

    mlx5_cqe64 *cqe_entry = &current_cq_q[cq_consumer_counter % cq_size];

    while (!is_cq_owner_sw(cqe_entry) ||
            (mlx5dv_get_cqe_opcode(cqe_entry) == 0xF))
    { __roc_inv(); }

    uint8_t opcode = mlx5dv_get_cqe_opcode(cqe_entry);
    GPU_DPRINTF("*** inside quiet outstanding %d wait_index %d cq_size %d "
                "op_code %d\n", quiet_counter, cq_consumer_counter, cq_size,
                opcode);

    if (opcode != 0) {
        uint8_t syndrome = get_cq_error_syndrome(cqe_entry);
        printf("*** inside quiet ERROR signature %d\n", syndrome);
    }

    threadImpl.decQuietCounter(&quiet_counter, quiet_counter);
    profiler.endTimer(start, POLL_CQ);

    start = profiler.startTimer();
    cq_consumer_counter++;
    swap_endian_store((uint32_t*) dbrec_cq, cq_consumer_counter);
    profiler.endTimer(start, NEXT_CQ);
}

__device__ void
QueuePair::quiet_single()
{
    threadImpl.quiet(this);
}

__device__ void
QueuePair::update_posted_wqe_generic(int pe, int32_t size, uintptr_t* laddr,
                                     uintptr_t* raddr, uint8_t opcode,
                                     int64_t atomic_data, int64_t atomic_cmp,
                                     bool ring_db, uint64_t atomic_ret_pos)
{
    GPU_DPRINTF("Function: update_posted_wqe_generic\n"
                "*** pe %d size %d laddr %p raddr %p opcode %d "
                "atomic_data %lu atomic_cmp %lu rtn_gpu_handle %p, "
                "ring_db %d\n", pe, size, laddr, raddr, opcode, atomic_data,
                atomic_cmp, this, ring_db);

    uint64_t start = profiler.startTimer();

    threadImpl.postLock(this);
    uint32_t num_wqes = connection_policy.getNumWqes(opcode);

    // Get the index for my thread's put in the SQ.
    uint64_t my_sq_counter = threadImpl.threadAtomicAdd(&sq_counter, num_wqes);
    uint64_t my_sq_index = my_sq_counter % max_nwqe;

    // 16-bit little endian version of the SQ index needed to build the cntrl
    // segment in the WQE.
    // FIXME: Seems completely broken past 64K...
    uint16_t le_sq_counter;
    swap_endian_store(&le_sq_counter, (uint16_t) my_sq_counter);

    GPU_DPRINTF("Posting %d WQEs at index %llu le_sq_counter %d\n",
                num_wqes, my_sq_index, le_sq_counter);

    bool flag = sq_overflow;
    uint32_t lkey;
    uint32_t rkey = 0;
    uint32_t ctrl_qp_sq;
    uint64_t ctrl_sig;
    lkey = this->lkey;
    rkey = this->rkey;
    ctrl_qp_sq = this->ctrl_qp_sq;
    ctrl_sig = this->ctrl_sig;

    connection_policy.setRkey(rkey, pe);

    if (opcode == MLX5_OPCODE_RDMA_WRITE && !size) {
        rkey = hdp_rkey[pe];
        size = 4;
    }

    /*
     * Build out all the segments required for my WQE(s) based on the
     * operation, starting at my_sq_index into the SQ.  SegmentBuilder will
     * keep track of placing the segments in the correct location.
     */
    SegmentBuilder seg_build(my_sq_index, current_sq);
    seg_build.update_cntrl_seg(opcode, le_sq_counter, ctrl_qp_sq, ctrl_sig,
                               connection_policy);

    seg_build.update_connection_seg(pe, connection_policy);
    seg_build.update_rdma_seg(raddr, rkey);

    if (opcode == MLX5_OPCODE_ATOMIC_FA || opcode == MLX5_OPCODE_ATOMIC_CS) {
        seg_build.update_atomic_data_seg(atomic_data, atomic_cmp);
        size = 8;
        lkey = atomic_ret.atomic_lkey;
        laddr = &atomic_ret.atomic_base_ptr[atomic_ret_pos];
    }

    if (size <= RTN_INLINE_THRESHOLD && opcode == MLX5_OPCODE_RDMA_WRITE)
        seg_build.update_inl_data_seg(laddr, size);
    else
        seg_build.update_data_seg(laddr, size, lkey);

    profiler.incStat(RTN_WQE_COUNT);
    profiler.endTimer(start, UPDATE_WQE);
    start = profiler.startTimer();

    threadImpl.finishPost(this, ring_db, num_wqes, pe, le_sq_counter, opcode);

    profiler.incStat(RTN_DB_COUNT);
    profiler.endTimer(start, RING_SQ_DB);
}

//////////// SHMEM Interface
__device__ void
QueuePair::put_nbi(void *dest, const void *source, size_t nelems,
                   int pe, bool db_ring)
{
    GPU_DPRINTF("Function: rtn_put_nbi\n");
    update_posted_wqe_generic(pe, nelems, (uintptr_t*) source,
                              (uintptr_t*) dest, MLX5_OPCODE_RDMA_WRITE,
                              0, 0, db_ring, 0);
}

__device__ void
QueuePair::get_nbi(void *dest, const void *source, size_t nelems,
                   int pe, bool db_ring)
{
    GPU_DPRINTF("Function: rtn_get_nbi\n");

    update_posted_wqe_generic(pe, nelems, (uintptr_t*) source,
                              (uintptr_t*) dest, MLX5_OPCODE_RDMA_READ,0, 0,
                              db_ring, 0);
}

__device__ int64_t
QueuePair::atomic_fetch(void *dest, int64_t value, int64_t cond,
                        int pe, bool db_ring, uint8_t atomic_op)
{
    GPU_DPRINTF("Function: rtn_atomic_fetch <op> %d \n", atomic_op);
    uint64_t pos = threadImpl.threadAtomicAdd((unsigned long long *)
                                     &atomic_ret.atomic_counter);
    pos = pos % max_nb_atomic;
    update_posted_wqe_generic(pe, sizeof(int64_t), nullptr,
                              (uintptr_t*) dest, (uint8_t) atomic_op,
                              value, cond, db_ring, pos);

    quiet_single();

    int64_t ret = (int64_t)((int64_t*)atomic_ret.atomic_base_ptr)[pos];
    __threadfence();

    return ret;
}

__device__ void
QueuePair::atomic_nofetch(void *dest, int64_t value, int64_t cond,
                          int pe, bool db_ring, uint8_t atomic_op)
{
    GPU_DPRINTF("Function: rtn_atomic_nofetch <op> %d \n", atomic_op);

    uint64_t pos = threadImpl.threadAtomicAdd((unsigned long long *)
                                              &atomic_ret.atomic_counter);
    pos = pos % max_nb_atomic;

    update_posted_wqe_generic(pe, sizeof(int64_t), nullptr,
                              (uintptr_t*) dest, (uint8_t) atomic_op,
                              value, cond, db_ring, pos);

    quiet_single();

}

__device__ void
QueuePair::fence(int pe)
{
    uintptr_t remote_hdp = hdp_address[pe];
    update_posted_wqe_generic(pe, 0, nullptr, (uintptr_t*)remote_hdp,
                              MLX5_OPCODE_RDMA_WRITE, 0, 0, true, 0);
}

__device__ void
QueuePair::waitCQSpace(int num_msgs)
{
    // We cannot post more outstanding requests than the completion queue
    // size.  Force a quiet if we are out of space.
    if ((quiet_counter + num_msgs) >= cq_size) {
        GPU_DPRINTF("*** inside post_cq forcing flush: outstanding %d "
                    "adding %d cq_size %d\n", quiet_counter, num_msgs,
                    cq_size);

        // TODO: More targeted flush would be better here.
        quiet_internal();
    }
}
