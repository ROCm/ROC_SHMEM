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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_INTERNAL_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_INTERNAL_HPP

#include "config.h"

#include <cstdint>
#include "hdp_policy.hpp"
#include "ipc_policy.hpp"
#include "stats.hpp"
#include "util.hpp"
#include "symmetric_heap.hpp"
#include "atomic_return.hpp"

namespace rocshmem {

#define DEFAULT_QUEUE_SIZE 64

#define SFENCE()   asm volatile("sfence" ::: "memory")

enum ro_net_cmds {
    RO_NET_PUT,
    RO_NET_P,
    RO_NET_GET,
    RO_NET_PUT_NBI,
    RO_NET_GET_NBI,
    RO_NET_AMO_FOP,
    RO_NET_AMO_FCAS,
    RO_NET_FENCE,
    RO_NET_QUIET,
    RO_NET_FINALIZE,
    RO_NET_TO_ALL,
    RO_NET_TEAM_TO_ALL,
    RO_NET_SYNC,
    RO_NET_BARRIER_ALL,
    RO_NET_BROADCAST,
    RO_NET_TEAM_BROADCAST,
    RO_NET_ALLTOALL,
    RO_NET_FCOLLECT,
};

enum ro_net_types {
    RO_NET_FLOAT,
    RO_NET_CHAR,
    RO_NET_DOUBLE,
    RO_NET_INT,
    RO_NET_LONG,
    RO_NET_LONG_LONG,
    RO_NET_SHORT,
    RO_NET_LONG_DOUBLE
};

/*
 * PVAR counters available in ro_net GPU side
 * RO_NET_WAIT_SLOT : reports the time waiting for a cmd queue slot
 * RO_NET_PACK      : reports the time to pack a a request from GPU
 * RO_NET_FENCE1    : reports the time for the first memory fence
 * RO_NET_FENCE2    : reports the time for the second memory fence
 * RO_NET_WAIT_HOST : reports the time GPU is waiting on CPU for blocking
 *                      calls
 * RO_NET_WAIT      : reports the time spent in ro_net_wait polling on
 *                      memory
 */
enum ro_net_pvar_t{
    RO_NET_WAIT_SLOT,
    RO_NET_FENCE1,
    RO_NET_FENCE2,
    RO_NET_PACK,
    RO_NET_WAIT_HOST,
    RO_NET_WAIT,
};

enum RO_NET_Op {
    RO_NET_SUM,
};

typedef struct queue_element {
    // Polled by the CPU to determine when a command is ready.  Set by the GPU
    // once a queue element has been completely filled out.  This is padded
    // from the actual data to prevent thrashing on an APU when the GPU is
    // trying to fill out a packet and the CPU is reading the valid bit.
    volatile char valid;
    char padding[63];
    // All fields written by the GPU and read by the CPU
    ro_net_cmds  type;
    int     PE;
    int     size;
    void*   src;
    void*   dst;
    int     threadId;
    // For collectives
    int logPE_stride;
    int PE_size;
    void*  pWrk;
    long*  pSync;
    int  op;
    int  datatype;
    int PE_root;
    MPI_Comm team_comm;
} __attribute__((__aligned__(64))) queue_element_t;

typedef struct queue_desc {
    // Read index for the queue.  Rarely read by the GPU when it thinks the
    // queue might be full, but the GPU normally uses a local copy that lags
    // behind the true read_idx.
    uint64_t read_idx;
    char padding1[56];
    // Write index for the queue.  Never accessed by CPU, since it uses the
    // valid bit in the packet itself to determine whether there is data to
    // consume.  The GPU has a local copy of the write_idx that it uses, but it
    // does write the local index to this location when the kernel completes
    // in case the queue needs to be reused without reseting all the pointers
    // to zero.
    uint64_t write_idx;
    char padding2[56];
    // This bit is used by the GPU to wait on a blocking operation. The initial
    // value is 0.  When a GPU enqueues a blocking operation, it waits for this
    // value to resolve to 1, which is set by the CPU when the blocking
    // operation completes.  The GPU then resets status back to zero.  There is
    // a seperate status variable for each work-item in a work-group
    char *status;
    char padding3[63];
} __attribute__((__aligned__(64))) queue_desc_t;


enum ro_net_stats {
    WAITING_ON_SLOT = 0,
    THREAD_FENCE_1,
    THREAD_FENCE_2,
    WAITING_ON_HOST,
    PACK_QUEUE,
    SHMEM_WAIT,
    RO_NUM_STATS
};

#ifdef PROFILE
typedef Stats<RO_NUM_STATS> ROStats;
#else
typedef NullStats<RO_NUM_STATS> ROStats;
#endif

/* Meant for local allocation on the GPU */
struct ro_net_wg_handle {
    queue_element_t *queue;
    ROStats profiler;
    unsigned int *barrier_ptr;
    uint64_t read_idx;
    uint64_t write_idx;
    uint64_t *host_read_idx;
    uint64_t queue_size;
    char *status;
    char *g_ret;
    atomic_ret_t atomic_ret;
    IpcImpl ipcImpl;
    HdpPolicy hdp_policy;
};

/* Device-side internal functions */
__device__ void inline
__ro_inv() {
    asm volatile ("buffer_wbinvl1;");
}

__device__ bool
isFull(uint64_t read_idx,
       uint64_t write_idx,
       int queue_size);

__device__ void
build_queue_element(ro_net_cmds type,
                    void *dst,
                    void *src,
                    size_t size,
                    int pe,
                    int logPE_stride,
                    int PE_size,
                    int PE_root,
                    void *pWrk,
                    long *pSync,
                    MPI_Comm team_comm,
                    int ro_net_win_id,
                    struct ro_net_wg_handle *handle,
                    bool blocking,
                    ROC_SHMEM_OP op = ROC_SHMEM_SUM,
                    ro_net_types datatype = RO_NET_INT);

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_INTERNAL_HPP
