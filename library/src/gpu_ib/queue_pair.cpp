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

#include "queue_pair.hpp"

#include <hip/hip_runtime.h>

#include "backend_ib.hpp"
#include "config.h"  // NOLINT(build/include_subdir)
#include "endian.hpp"
#include "segment_builder.hpp"
#include "util.hpp"

namespace rocshmem {

QueuePair::QueuePair(GPUIBBackend *backend)
    : hdp_policy(*backend->hdp_policy),
      connection_policy(*backend->networkImpl.connection_policy) {
    hdp_rkey = backend->networkImpl.hdp_rkey;
    hdp_address = backend->networkImpl.hdp_address;

    atomic_ret.atomic_lkey = backend->networkImpl.atomic_ret->atomic_lkey;
    atomic_ret.atomic_counter = 0;
}

__device__
QueuePair::~QueuePair() {
    uint64_t start = profiler.startTimer();

    global_qp->sq_counter = sq_counter;
    global_qp->local_sq_cnt = local_sq_cnt;
    global_qp->cq_consumer_counter = cq_consumer_counter;
    global_qp->current_sq = current_sq;
    global_qp->current_cq_q = current_cq_q;
    global_qp->sq_overflow = sq_overflow;
    global_qp->quiet_counter = quiet_counter;
    profiler.endTimer(start, FINALIZE);

    global_qp->profiler.accumulateStats(profiler);

    __syncthreads();
}

__device__ bool
QueuePair::is_cq_owner_sw(mlx5_cqe64 *cqe_entry) {
    // static MLX5DV_ALWAYS_INLINE
    // uint8_t mlx5dv_get_cqe_owner(struct mlx5_cqe64 *cqe)
    // {
    //     return cqe->op_own & 0x1;
    // }
    // return mlx5dv_get_cqe_owner(cqe_entry) ==
    //     ((cq_consumer_counter >> cq_log_size) & 1);

    return (__builtin_nontemporal_load(&cqe_entry->op_own) & 0x1) ==
           ((cq_consumer_counter >> cq_log_size) & 1);
}

__device__ uint8_t
QueuePair::get_cq_error_syndrome(mlx5_cqe64 *cqe_entry) {
    mlx5_err_cqe * cqe_err = reinterpret_cast<mlx5_err_cqe*>(cqe_entry);
    return cqe_err->syndrome;
}

__device__ void
QueuePair::ring_doorbell(uint64_t db_val) {
    swap_endian_store(const_cast<uint32_t*>(dbrec_send),
                      reinterpret_cast<uint32_t>(sq_counter));
    STORE(db.ptr, db_val);
    db.uint  ^= 256;
}

__device__ void
QueuePair::set_completion_flag_on_wqe(int num_wqes){
    uint64_t *wqe = &current_sq[8 * ((sq_counter - num_wqes) % max_nwqe)];
    uint8_t *wqe_ce = reinterpret_cast<uint8_t*>(wqe) + 11;
    *wqe_ce = 8;
}

template<>
__device__ void
QueuePair::update_wqe_ce_single<false>(int num_wqes) {
    if(sq_counter % max_nwqe == (max_nwqe-2)){
        set_completion_flag_on_wqe(num_wqes);
        quiet_counter++;
    }
}

template<>
__device__ void
QueuePair::update_wqe_ce_single<true>(int num_wqes) {
    set_completion_flag_on_wqe(num_wqes);
    quiet_counter++;
}

template<>
__device__ void
QueuePair::update_wqe_ce_thread<false>(int num_wqes) {
}

template<>
__device__ void
QueuePair::update_wqe_ce_thread<true>(int num_wqes) {
    set_completion_flag_on_wqe(num_wqes);
    atomicAdd(&quiet_counter, 1);
}


__device__ void
QueuePair::compute_db_val_opcode(uint64_t *db_val,
                                 uint16_t dbrec_val,
                                 uint8_t opcode) {
    uint64_t opcode64 = opcode;
    opcode64 = opcode64 << 24 & 0x000000FFFF000000;

    uint64_t dbrec = dbrec_val << 8;
    dbrec = dbrec & 0x0000000000FFFF00;

    uint64_t val = *db_val;
    val = val & 0xFFFFFFFFFF0000FF;

    *db_val = val | dbrec | opcode64;
}

template <class level>
__device__ void
QueuePair::quiet_internal() {
    uint32_t quiet_val = quiet_counter;
    if (!quiet_val) {
        return;
    }

    level L;

    profiler.incStat(QUIET_COUNT);

    uint64_t start = profiler.startTimer();

    cq_consumer_counter = cq_consumer_counter + quiet_val - 1;
    uint32_t indx = (cq_consumer_counter % cq_size);

    mlx5_cqe64 *cqe_entry = &current_cq_q[indx];

    int val_ld = uncached_load_ubyte(&(cqe_entry->op_own));
    uint8_t val_op_own = val_ld;

     while (!((val_op_own & 0x1) ==
             ((cq_consumer_counter >> cq_log_size) & 1))
            || ((val_op_own) >> 4) == 0xF) {
        val_ld = uncached_load_ubyte(&(cqe_entry->op_own));
        val_op_own = val_ld;
    }

    uint8_t opcode = val_op_own >> 4;
    if (opcode != 0) {
        uint8_t syndrome = get_cq_error_syndrome(cqe_entry);
        mlx5_err_cqe * cqe_err = reinterpret_cast<mlx5_err_cqe*>(cqe_entry);
        gpu_dprintf("QUIET ERROR: signature %d opcode_qpn %llx wqe_cnt %llx \n",
                    syndrome, cqe_err->s_wqe_opcode_qpn, cqe_err->wqe_counter);
    }

    L.decQuietCounter(&quiet_counter, quiet_val);
    profiler.endTimer(start, POLL_CQ);

    start = profiler.startTimer();
    cq_consumer_counter++;
    swap_endian_store(const_cast<uint32_t*>(dbrec_cq),
                      cq_consumer_counter);
    profiler.endTimer(start, NEXT_CQ);
}

template<class level>
__device__ void
QueuePair::quiet_single() {
    level L;
    L.quiet(this);
}

template<class level>
__device__ void
QueuePair::quiet_single_heavy(int pe) {
    level L;
    L.quiet_heavy(this, pe);
}

template<class level, bool cqe>
__device__ void
QueuePair::update_posted_wqe_generic(int pe,
                                     int32_t size,
                                     uintptr_t *laddr,
                                     uintptr_t *raddr,
                                     uint8_t opcode,
                                     int64_t atomic_data,
                                     int64_t atomic_cmp,
                                     bool ring_db,
                                     uint64_t atomic_ret_pos,
                                     bool zero_byte_rd) {
    uint64_t start = profiler.startTimer();

    level L;
    L.postLock(this, pe);
    uint32_t num_wqes = connection_policy.getNumWqes(opcode);

    // Get the index for my thread's put in the SQ.
    uint64_t my_sq_counter = L.threadAtomicAdd(&sq_counter, num_wqes);
    uint64_t my_sq_index = my_sq_counter % max_nwqe;

    // 16-bit little endian version of the SQ index needed to build the cntrl
    // segment in the WQE.
    uint16_t le_sq_counter;
    uint16_t sq_counter_u16 = my_sq_counter;
    swap_endian_store(&le_sq_counter, sq_counter_u16);


    bool flag = sq_overflow;
    uint32_t lkey_in_stack_frame = lkey;
    uint32_t rkey_in_stack_frame = rkey;
    uint32_t ctrl_qp_sq_in_stack_frame = ctrl_qp_sq;
    uint64_t ctrl_sig_in_stack_frame = ctrl_sig;

    connection_policy.setRkey(&rkey_in_stack_frame, pe);

    if (opcode == MLX5_OPCODE_RDMA_WRITE && !size) {
        rkey_in_stack_frame = hdp_rkey[pe];
        size = 4;
    }

    /*
     * Build out all the segments required for my WQE(s) based on the
     * operation, starting at my_sq_index into the SQ. SegmentBuilder will
     * keep track of placing the segments in the correct location.
     */
    SegmentBuilder seg_build(my_sq_index, current_sq);
    seg_build.update_cntrl_seg(opcode,
                               le_sq_counter,
                               ctrl_qp_sq_in_stack_frame,
                               ctrl_sig_in_stack_frame,
                               &connection_policy,
                               zero_byte_rd);
    seg_build.update_connection_seg(pe, &connection_policy);
    seg_build.update_rdma_seg(raddr, rkey_in_stack_frame);

    if (opcode == MLX5_OPCODE_ATOMIC_FA || opcode == MLX5_OPCODE_ATOMIC_CS) {
        seg_build.update_atomic_data_seg(atomic_data, atomic_cmp);
        size = 8;
        lkey_in_stack_frame = atomic_ret.atomic_lkey;
        laddr = &atomic_ret.atomic_base_ptr[atomic_ret_pos];
    }

    if (size <= inline_threshold && opcode == MLX5_OPCODE_RDMA_WRITE) {
        seg_build.update_inl_data_seg(laddr, size);
    } else {
        seg_build.update_data_seg(laddr, size, lkey_in_stack_frame);
    }

    profiler.incStat(WQE_COUNT);
    profiler.endTimer(start, UPDATE_WQE);
    start = profiler.startTimer();

    L.template finishPost<cqe>(this,
                               ring_db,
                               num_wqes,
                               pe,
                               le_sq_counter,
                               opcode);

    profiler.incStat(DB_COUNT);
    profiler.endTimer(start, RING_SQ_DB);
}

/******************************************************************************
 ****************************** SHMEM INTERFACE *******************************
 *****************************************************************************/
template<class level>
__device__ void
QueuePair::put_nbi(void *dest,
                   const void *source,
                   size_t nelems,
                   int pe,
                   bool db_ring) {
    uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
    uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);

    update_posted_wqe_generic<level, false>(pe,
                                            nelems,
                                            src,
                                            dst,
                                            MLX5_OPCODE_RDMA_WRITE,
                                            0,
                                            0,
                                            db_ring,
                                            0);
}

template<class level>
__device__ void
QueuePair::put_nbi_cqe(void *dest,
                       const void *source,
                       size_t nelems,
                       int pe,
                       bool db_ring) {
    uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
    uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);

    update_posted_wqe_generic<level, true>(pe,
                                           nelems,
                                           src,
                                           dst,
                                           MLX5_OPCODE_RDMA_WRITE,
                                           0,
                                           0,
                                           db_ring,
                                           0);
}

template <class level>
__device__ void
QueuePair::get_nbi(void *dest,
                   const void *source,
                   size_t nelems,
                   int pe,
                   bool db_ring) {
    uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
    uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);

    update_posted_wqe_generic<level, false>(pe,
                                            nelems,
                                            src,
                                            dst,
                                            MLX5_OPCODE_RDMA_READ,
                                            0,
                                            0,
                                            db_ring,
                                            0);
}

template <class level>
__device__ void
QueuePair::get_nbi_cqe(void *dest,
                       const void *source,
                       size_t nelems,
                       int pe,
                       bool db_ring) {
    uintptr_t *src = reinterpret_cast<uintptr_t*>(const_cast<void*>(source));
    uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);

    update_posted_wqe_generic<level, true>(pe,
                                           nelems,
                                           src,
                                           dst,
                                           MLX5_OPCODE_RDMA_READ,
                                           0,
                                           0,
                                           db_ring,
                                           0);
}

template <class level>
__device__ void
QueuePair::zero_b_rd(int pe) {
    uintptr_t *dst = reinterpret_cast<uintptr_t*>(base_heap[pe]);

    update_posted_wqe_generic<level, true>(pe,
                                           0,
                                           nullptr,
                                           dst,
                                           MLX5_OPCODE_RDMA_READ,
                                           0,
                                           0,
                                           true,
                                           0,
                                           true);  // enable 0_bye read op
}

__device__ int64_t
QueuePair::atomic_fetch(void *dest,
                        int64_t value,
                        int64_t cond,
                        int pe,
                        bool db_ring,
                        uint8_t atomic_op) {
    THREAD TH;
    uint64_t pos = TH.threadAtomicAdd(
        reinterpret_cast<unsigned long long*>( /* NOLINT(runtime/int) */
            &atomic_ret.atomic_counter));

    pos = pos % max_nb_atomic;

    int64_t *atomic_base_ptr =
        reinterpret_cast<int64_t*>(atomic_ret.atomic_base_ptr);

    int64_t *load_address = &atomic_base_ptr[pos];

    *load_address = -100;

    uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);

    update_posted_wqe_generic<THREAD, true>(pe,
                                            sizeof(int64_t),
                                            nullptr,
                                            dst,
                                            atomic_op,
                                            value,
                                            cond,
                                            db_ring,
                                            pos);
    quiet_single<THREAD>();

    while (uncached_load(load_address) == -100) {
    }

    int64_t ret = *load_address;

    __threadfence();

    return ret;
}

__device__ void
QueuePair::atomic_nofetch(void *dest,
                          int64_t value,
                          int64_t cond,
                          int pe,
                          bool db_ring,
                          uint8_t atomic_op) {
    THREAD TH;
    uint64_t pos = TH.threadAtomicAdd(
        reinterpret_cast<unsigned long long*>( /* NOLINT(runtime/int) */
            &atomic_ret.atomic_counter));
    pos = pos % max_nb_atomic;
    uintptr_t *dst = reinterpret_cast<uintptr_t*>(dest);

    update_posted_wqe_generic<THREAD, true>(pe,
                                            sizeof(int64_t),
                                            nullptr,
                                            dst,
                                            atomic_op,
                                            value,
                                            cond,
                                            db_ring,
                                            pos);

    quiet_single<THREAD>();
}

__device__ void
QueuePair::fence(int pe) {
    auto remote_hdp_uncast = hdp_address[pe];
    uintptr_t *remote_hdp = reinterpret_cast<uintptr_t*>(remote_hdp_uncast);
    update_posted_wqe_generic<THREAD, false>(pe,
                                             0,
                                             nullptr,
                                             remote_hdp,
                                             MLX5_OPCODE_RDMA_WRITE,
                                             0,
                                             0,
                                             true,
                                             0);
}

__device__ void
QueuePair::waitCQSpace(int num_msgs) {
    // We cannot post more outstanding requests than the completion queue
    // size.  Force a quiet if we are out of space.
    if ((quiet_counter + num_msgs) >= cq_size) {
        GPU_DPRINTF("*** inside post_cq forcing flush: outstanding %d "
                    "adding %d cq_size %d\n", quiet_counter, num_msgs,
                    cq_size);

        // TODO(khamidou): More targeted flush would be better here.
        quiet_single<THREAD>();
    }
}

__device__ void
QueuePair::waitSQSpace(int num_msgs) {
    // We cannot post more outstanding requests than the Send queue
    // size.  Force a quiet if we are out of space.
    local_sq_cnt += num_msgs;
    int div = local_sq_cnt /max_nwqe;

    if (div >0) {
        GPU_DPRINTF("*** inside waitSQSpace forcing flush to overrun the SQ"
                    " sq_counter %d  adding %d quiet_conter %d \n", sq_counter,
                    num_msgs, max_nwqe, quiet_counter);

        quiet_single<THREAD>();
        local_sq_cnt = local_sq_cnt% max_nwqe;

    }
}


void
QueuePair::setDBval(uint64_t val) {
    db_val = val;
}

#define THREAD_LEVEL_GEN(T)                                                    \
    template                                                                   \
    __device__ void                                                            \
    QueuePair::put_nbi<T>(void *dest,                                          \
                          const void *source,                                  \
                          size_t nelems,                                       \
                          int pe,                                              \
                          bool db_ring);                                       \
    template                                                                   \
    __device__ void                                                            \
    QueuePair::put_nbi_cqe<T>(void *dest,                                      \
                              const void *source,                              \
                              size_t nelems,                                   \
                              int pe,                                          \
                              bool db_ring);                                   \
    template                                                                   \
    __device__ void                                                            \
    QueuePair::get_nbi<T>(void *dest,                                          \
                          const void *source,                                  \
                          size_t nelems,                                       \
                          int pe,                                              \
                          bool db_ring);                                       \
    template                                                                   \
    __device__ void                                                            \
    QueuePair::get_nbi_cqe<T>(void *dest,                                      \
                              const void *source,                              \
                              size_t nelems,                                   \
                              int pe,                                          \
                              bool db_ring);                                   \
     template                                                                  \
    __device__ void                                                            \
    QueuePair::zero_b_rd<T>(int pe);                                           \
    template                                                                   \
    __device__ void                                                            \
    QueuePair::quiet_single<T>();                                              \
     template                                                                  \
    __device__ void                                                            \
    QueuePair::quiet_single_heavy<T>(int pe);                                  \
    template                                                                   \
    __device__ void                                                            \
    QueuePair::quiet_internal<T>();

THREAD_LEVEL_GEN(THREAD)
THREAD_LEVEL_GEN(WG)
THREAD_LEVEL_GEN(WAVE)

}  // namespace rocshmem
