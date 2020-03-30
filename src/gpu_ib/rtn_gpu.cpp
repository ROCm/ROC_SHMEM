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

#include "hip/hip_runtime.h"

#include <stdio.h>

#include "rtn.hpp"
#include "rtn_internal.hpp"
#include "util.hpp"

__device__ void  __rtn_waitcnt() { asm volatile ("s_waitcnt vmcnt(0);"); }

__device__ void inline __store_short (uint16_t val, uint16_t* dst)
{
        asm  volatile("flat_store_short %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}

__device__ void inline __store_dword (uint32_t val, uint32_t* dst)
{
        asm  volatile("flat_store_dword %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}

__device__ void inline __store_dwordx2 (uint64_t val, uint64_t* dst)
{
        asm  volatile("flat_store_dwordx2 %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}

__device__ void inline __store_dwordx2 (int2 val, uint64_t* dst)
{
        asm  volatile("flat_store_dwordx2 %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}

__device__ void inline __store_dwordx4 (ulong2 val, uint64_t* dst)
{
        asm  volatile("flat_store_dwordx4 %0 %1 glc slc"
                      : : "X"(dst), "X" (val));
}

__device__  void inline __atomic_storex2 (uint64_t val, uint64_t* dst)
{
        asm  volatile("flat_atomic_swap_x2 %0 %1 slc \ns_waitcnt  lgkmcnt(0)"
                      : : "v"(dst), "v" (val));
}

__device__ uint32_t lowerID ()
{
    //uint64_t mask = __activemask();
    uint64_t mask = __ballot(1);
    return  (__ffsll((unsigned long long int)mask) -1);
}

__device__ int wave_SZ()
{
    uint64_t mask = __ballot(1);
    return  (__popcll((unsigned long long int)mask));
}

__device__ uint64_t inline swap_uint64(uint64_t val)
{
    val = ((val << 8) & 0xFF00FF00FF00FF00ULL) |
            ((val >> 8) & 0x00FF00FF00FF00FFULL);
    val = ((val << 16) & 0xFFFF0000FFFF0000ULL) |
            ((val >> 16) & 0x0000FFFF0000FFFFULL);
    return (val << 32) | (val >> 32);
}

__device__ uint32_t inline swap_uint32(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00 ) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

__device__ uint16_t inline swap_uint16(uint16_t val)
{
    val = ((val << 8) & 0xFF00 ) | ((val >> 8) & 0x00FF);
    return (val);
}

__device__ bool
QueuePair::is_cq_owner_sw(uint8_t *cqe_entry)
{
    bool owner_val = cqe_entry[CQ_HEADER_BYTE_OFFSET] & 1;
    bool expected_owner_val = (cq_consumer_counter >> cq_log_size) & 1;
    return owner_val == expected_owner_val;
}

__device__ uint8_t
QueuePair::get_cq_opcode(uint8_t *cqe_entry)
{
    return cqe_entry[CQ_HEADER_BYTE_OFFSET] >> 4;
}

__device__ uint8_t
QueuePair::get_cq_error_syndrome(uint8_t *cqe_entry)
{
    return cqe_entry[CQ_ERROR_BYTE_OFFSET];
}

__device__ void
QueuePair::ring_doorbell(uint64_t db_val)
{
    GPU_DPRINTF("*** Ringing DB @%p: dbrec %p db_val %llu counter %d\n",
                db, (uint32_t*) dbrec_send, db_val, sq_counter);

    uint32_t l_val = sq_counter;
    l_val = swap_uint32(l_val);

    __store_dword ((uint32_t) l_val, (uint32_t*) dbrec_send);

    __rtn_waitcnt();
    __store_dwordx2(db_val, db);
}

__device__
QueuePair::QueuePair(RTNGlobalHandle *rtn_handle, int offset)
{
    uint64_t start = profiler.startTimer();

    GPU_DPRINTF("Setting handle offset %d\n", offset);

    db = rtn_handle[offset].rqp->db;
    current_sq = rtn_handle[offset].sq_current;
    current_cq_q = rtn_handle[offset].cq_current;
    dbrec_send = rtn_handle[offset].rqp->dbrec_send;
    dbrec_cq = rtn_handle[offset].rqp->rcq->dbrec_cq;
    sq_overflow = false;
    cq_log_size = rtn_handle[offset].rqp->rcq->cq_log_size;
    cq_size = rtn_handle[offset].rqp->rcq->cq_size;
    max_nwqe = rtn_handle[offset].rqp->max_nwqe;
    num_cqs = rtn_handle[offset].num_cqs;
    threadImpl = rtn_handle[offset].threadImpl;
    sq_counter = rtn_handle[offset].sq_counter;
    cq_consumer_counter = rtn_handle[offset].cq_counter;
    sq_overflow = rtn_handle[offset].sq_overflow;
    quiet_counter = rtn_handle[offset].quiet_counter;
    global_handle = &rtn_handle[offset];

    // TODO: All these rtn_handle references that directly access rtn_handle
    // [0] need to be moved to GPUIBContext.
    hdp_rkey = rtn_handle[0].hdp_rkey;
    hdp_address = rtn_handle[0].hdp_address;
    threadImpl.setDBval(*current_sq);

#ifdef _USE_GPU_UPDATE_SQ_
    lkey = rtn_handle[offset].lkey;
    rkey = rtn_handle[offset].rkey;
    ctrl_qp_sq = rtn_handle[offset].ctrl_qp_sq;
    ctrl_sig = rtn_handle[offset].ctrl_sig;
#endif
    hdp_copy(&hdp_regs, rtn_handle[0].hdp_regs);

    atomic_ret.atomic_base_ptr =
        &(((uint64_t*)(rtn_handle[0].atomic_ret->atomic_base_ptr))
        [max_nb_atomic * get_flat_block_id()]);

    atomic_ret.atomic_lkey = rtn_handle[0].atomic_ret->atomic_lkey;
    atomic_ret.atomic_counter = 0;
#ifdef _USE_DC_
    vec_dct_num = rtn_handle[0].vec_dct_num;
    vec_lids = rtn_handle[0].vec_lids;
    vec_rkey = rtn_handle[0].vec_rkey;
#endif
    profiler.endTimer(start, INIT);
}

__device__
QueuePair::~QueuePair()
{
    uint64_t start = profiler.startTimer();

    global_handle->threadImpl = threadImpl;
    global_handle->sq_counter = sq_counter;
    global_handle->cq_counter = cq_consumer_counter;
    global_handle->sq_current = current_sq ;
    global_handle->cq_current = current_cq_q;
    global_handle->sq_overflow = sq_overflow;
    global_handle->quiet_counter = quiet_counter;

    profiler.endTimer(start, FINALIZE);
    global_handle->profiler.accumulateStats(profiler);

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

__device__ void inline rtn_update_dc_seg (uintptr_t *wqe,
                                          uint32_t *vec_dct_num,
                                          uint16_t *vec_lids, int pe)
{
    uintptr_t *dc_seg = wqe + 2;

    int8_t num_dcts = 1;
    uint32_t dct = vec_dct_num[pe * num_dcts];

    uint32_t *ptr = (uint32_t*) dc_seg + 2;
    *ptr = dct;//val | dct;

    uint16_t lid = (uint16_t)vec_lids[pe];
    int16_t *ptr_lid = (int16_t *) dc_seg + 7;
    *ptr_lid = lid;

}

__device__ void
QueuePair::update_cntrl_seg(uintptr_t *wqe, uint8_t opcode,
                            uint16_t wqe_idx, bool flag,
                            uint32_t ctrl_qp_sq, uint64_t ctrl_sig)
{
#ifdef _USE_GPU_UPDATE_SQ_
    uint32_t DS;
#ifdef _USE_DC_
    if(opcode == RDMA_WRITE  || opcode == RDMA_READ) {
        DS = 4;
    } else {
        DS = 5;
    }
#else
    if (opcode == RDMA_WRITE  || opcode == RDMA_READ) {
        DS = 3;
    } else {
        DS = 4;
    }
#endif

    DS = swap_uint32(DS);
    ctrl_qp_sq = DS | ctrl_qp_sq;
    uint32_t ctrl_idx_opcode =
        (((uint32_t)wqe_idx) << 8 ) | ((uint32_t)opcode) << 24;
    int2 ctrl1 = make_int2(ctrl_idx_opcode, ctrl_qp_sq);
    __store_dwordx2 (ctrl1, (uint64_t*) wqe);

    __store_dwordx2 (ctrl_sig, (uint64_t*) (wqe + 1));

#else

    if (flag == true) {
        uint8_t  *ptr_wqe_idx = (uint8_t*) wqe + 1;
        __store_short ((uint16_t) wqe_idx, (uint16_t*)ptr_wqe_idx);
    }
    if((opcode != RDMA_WRITE)){
        uint8_t *ptr = (uint8_t*) wqe + 3;
        *ptr = opcode;
        if(opcode == ATOMIC_FAD || opcode == ATOMIC_CAS) {
            ptr = (uint8_t*)wqe + 7;
            *ptr = ((uint8_t) *ptr + 0x01);
        }
    }
#endif
}

__device__ void
QueuePair::update_atomic_data_seg(uintptr_t * base, uint8_t opcode,
                                  int64_t atomic_data, int64_t atomic_cmp,
                                  uint64_t atomic_ret_pos)
{
#ifndef _USE_DC_
    uintptr_t *atomic = base + 4;
#else
    uintptr_t *atomic = base + 6;
#endif

    *((int64_t*)atomic) = (int64_t)(swap_uint64(atomic_data));
    atomic = atomic + 1;
    *((int64_t*)atomic) = (int64_t)(swap_uint64(atomic_cmp));
#ifndef _USE_DC_
    uintptr_t *data = base + 6;
#else
    uintptr_t *data = base + 8;
#endif

    uint32_t sz_no_inline = ((uint32_t)8 & 0x0FFFFFFF);

    *((uint32_t*)data) = swap_uint32((uint32_t)(sz_no_inline));
    uint32_t* lkey = (uint32_t*) data + 1;
    *((uint32_t*)lkey) = atomic_ret.atomic_lkey;

    uint64_t laddr = swap_uint64((uint64_t)
        (((uintptr_t*)atomic_ret.atomic_base_ptr) +
         atomic_ret_pos ));

    uintptr_t * ret_atomic_addr = (uintptr_t*)data + 1;

    *((uint64_t*)ret_atomic_addr) = (uint64_t) laddr;
}

__device__ void
QueuePair::update_rdma_seg(uintptr_t * base, uintptr_t * raddr, uint32_t rkey)
{
    uint64_t new_raddr = swap_uint64((uint64_t)raddr);

#ifdef _USE_GPU_UPDATE_SQ_
    uint64_t *rdma;
#ifndef _USE_DC_
    rdma = base + 2;
#else
    rdma = base + 4;
#endif
     __store_dwordx2((uint64_t)new_raddr, (uint64_t*) rdma);
     __store_dword((uint32_t)rkey, (uint32_t*) rdma + 2);


#else
#ifndef _USE_DC_
    uintptr_t *rdma = base + 2;
    if (rkey != 0) {
        uintptr_t *rkey_ptr = base + 3;
        *((uint32_t*)rkey_ptr) = rkey;
    }
#else
    uintptr_t *rdma = base + 4;
    uintptr_t *rkey_ptr = base + 5;
    *((uint32_t*)rkey_ptr) = rkey;
#endif

     __store_dwordx2 ((uint64_t)new_raddr, (uint64_t*) rdma);
#endif

}

__device__ void
QueuePair::update_data_seg(uintptr_t * base, uintptr_t * laddr,
                           int32_t size, bool opcode, uint32_t lkey)
{
#ifndef _USE_DC_
    uintptr_t *data_sz = base + 4;
#else
    uintptr_t *data_sz = base + 6;
#endif
    if ((size <= RTN_INLINE_THRESHOLD) && opcode) {

        *data_sz = swap_uint32((size & 0x000003FF) | 0x80000000);

        /*
         * TODO: uintptr_t is the wrong data type for basically everything.
         * IB operates using DWORDS, so to reference the next DWORD we
         * have to cast to smaller type.  Needs to be addressed throughout
         * the code.
         */
        if (laddr) {
            memcpy(((uint32_t *) data_sz) + 1, laddr, size);
        } else {
            // Assume fence HDP flush
            // TODO: Rework fence interface to avoid this
            *(((uint32_t *) data_sz) + 1) = 1;
        }

    } else {
        uint64_t new_laddr = swap_uint64((uint64_t)laddr);
        uint32_t sz_no_inline = ((uint32_t)size & 0x7FFFFFFF);

        uintptr_t *data = data_sz + 1;
#ifndef _USE_GPU_UPDATE_SQ_
        sz_no_inline =  swap_uint32((uint32_t)(sz_no_inline));

        __store_dword ((uint32_t)sz_no_inline, (uint32_t*) data_sz);
        __store_dwordx2 ((uint64_t)new_laddr, (uint64_t*) data);
#else
        sz_no_inline = swap_uint32(sz_no_inline);
        int2 no_inl_val = make_int2(sz_no_inline, lkey);

        __store_dwordx2 (no_inl_val, (uint64_t*) data_sz);
        __store_dwordx2 ((uint64_t)new_laddr, (uint64_t*) data);
#endif
    }
}

__device__ void
QueuePair::quiet_internal()
{
    if (!quiet_counter) return;

    profiler.incStat(RTN_QUIET_COUNT);

    uint64_t start = profiler.startTimer();

    cq_consumer_counter = cq_consumer_counter + quiet_counter - 1;

    uint8_t *cqe_entry =
        &current_cq_q[CQ_ENTRY_BYTES * (cq_consumer_counter % cq_size)];

    while (!is_cq_owner_sw(cqe_entry) || (get_cq_opcode(cqe_entry) == 0xF))
    { __roc_inv(); }

    uint8_t opcode = get_cq_opcode(cqe_entry);
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
    *dbrec_cq = swap_uint32(cq_consumer_counter);
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
    uint32_t num_wqes = 1;

#ifdef _USE_DC_
    // FIXME: we assume all threads in wave are performing ATOMIC ops
    // while this might be common, we do no thave such restriction
    // so need to be fixed
    if (opcode == ATOMIC_FAD || opcode == ATOMIC_CAS)
        num_wqes = 2;
#endif

    // Get the index for my thread's put in the SQ.
    uint64_t my_sq_counter = threadImpl.threadAtomicAdd(&sq_counter, num_wqes);
    uint64_t my_sq_index = my_sq_counter % max_nwqe;

#ifndef _USE_GPU_UPDATE_SQ_
    // FIXME: seems broken for multi-wqe ops where we might skip 0.
    if (my_sq_index == 0) sq_overflow = true;
#endif

    // 16-bit little endian version of the SQ index needed to build the cntrl
    // segment in the WQE.
    // FIXME: Seems completely broken past 64K...
    uint16_t le_sq_counter = swap_uint16(my_sq_counter);

    // Pointer to my spot in the SQ.
    uintptr_t *base = current_sq + (8 * my_sq_index);

    GPU_DPRINTF("Posting %d WQEs at index %llu addr %p le_sq_counter %d\n",
                num_wqes, my_sq_index, base, le_sq_counter);

    bool flag = sq_overflow;
    uint32_t lkey;
    uint32_t rkey = 0;
    uint32_t ctrl_qp_sq;
    uint64_t ctrl_sig;
#ifdef _USE_GPU_UPDATE_SQ_
    lkey = this->lkey;
    rkey = this->rkey;
    ctrl_qp_sq = this->ctrl_qp_sq;
    ctrl_sig = this->ctrl_sig;
#endif

#ifdef _USE_DC_
    rkey = vec_rkey[pe];
#endif

    if (opcode == RDMA_FENCE) {
        opcode = RDMA_WRITE;
        rkey = hdp_rkey[pe];
        laddr = NULL;
        size = 4;
    }

    update_cntrl_seg(base, opcode, le_sq_counter, flag, ctrl_qp_sq, ctrl_sig);
#ifdef _USE_DC_
    rtn_update_dc_seg(base, vec_dct_num, vec_lids, pe);
#endif
    update_rdma_seg(base, raddr, rkey);

    switch (opcode) {
        case RDMA_WRITE:
            update_data_seg(base, laddr, size, true, lkey);
            break;
        case RDMA_READ:
            update_data_seg(base, laddr, size, false, lkey);
            break;
        case ATOMIC_FAD:
            update_atomic_data_seg(base, opcode, atomic_data, atomic_cmp,
                                   atomic_ret_pos);
            break;
        case ATOMIC_CAS:
            update_atomic_data_seg(base, opcode, atomic_data, atomic_cmp,
                                   atomic_ret_pos);
            break;
        default:
            assert("UNKNOWN OPCODE");
    }

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
                              (uintptr_t*) dest, (uint8_t) RDMA_WRITE,
                              0, 0, db_ring, 0);
}

__device__ void
QueuePair::get_nbi(void *dest, const void *source, size_t nelems,
                   int pe, bool db_ring)
{
    GPU_DPRINTF("Function: rtn_get_nbi\n");

    update_posted_wqe_generic(pe, nelems, (uintptr_t*) source,
                              (uintptr_t*) dest, (uint8_t) RDMA_READ,0, 0,
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
    update_posted_wqe_generic(pe, sizeof(int64_t), NULL,
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

    update_posted_wqe_generic(pe, sizeof(int64_t), NULL,
                              (uintptr_t*) dest, (uint8_t) atomic_op,
                              value, cond, db_ring, pos);

    quiet_single();

}

__device__ void
QueuePair::fence(int pe)
{
    uintptr_t remote_hdp = hdp_address[pe];
    update_posted_wqe_generic(pe, 4, NULL, (uintptr_t*)remote_hdp,
                             (uint8_t) RDMA_FENCE, 0, 0, true, 0);
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
