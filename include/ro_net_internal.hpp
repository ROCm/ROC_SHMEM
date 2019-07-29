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

#ifndef RO_NET_INTERNAL_H
#define RO_NET_INTERNAL_H

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#define  __HIP_PLATFORM_HCC__

#include "hip/hip_runtime.h"
#include "hdp_helper.hpp"
#include "ro_net.hpp"

#define hipCheck(cmd, msg) \
{\
    if (cmd != hipSuccess) {\
        fprintf(stderr, "Unrecoverable HIP error: %s\n", msg);\
        exit(-1);\
    }\
}

#define hipGetDevice_assert(dev)\
{ hipCheck(hipGetDevice(dev), "cannot get device"); }

#define hipMalloc_assert(ptr, size) \
{ hipCheck(hipMalloc(ptr, size), "cannot allocate device memory"); }

#define hipExtMallocWithFlags_assert(ptr, size, flags) \
{ hipCheck(hipExtMallocWithFlags(ptr, size, flags), \
           "cannot allocate uncacheable device memory"); }

#define hipHostMalloc_assert(ptr, size) \
{ hipCheck(hipHostMalloc(ptr, size), "cannot allocate host memory"); }

#define hipFree_assert(ptr) \
{ hipCheck(hipFree(ptr), "cannot free device memory"); }

#define hipHostFree_assert(ptr) \
{ hipCheck(hipHostFree(ptr), "cannot free host memory"); }

#define hipHostRegister_assert(ptr, size, flags) \
{ hipCheck(hipHostRegister(ptr, size, flags), "cannot register host memory"); }

#define hipHostUnregister_assert(ptr) \
{ hipCheck(hipHostUnregister(ptr), "cannot unregister host memory"); }

#define DEFAULT_QUEUE_SIZE 64

#define SFENCE()   asm volatile("sfence" ::: "memory")

#ifdef PROFILE
#define PVAR_START() \
if (handle->profiler.enabled) { \
    start = __read_clock(); \
}
#define PVAR_END(x) \
if (handle->profiler.enabled) {\
    atomicAdd((unsigned long long *) &x, __read_clock() - start); \
}
#else
#define PVAR_START()
#define PVAR_END(x)
#endif /* PROFILE */

extern bool RO_NET_DEBUG;

#ifdef DEBUG
# define DPRINTF(x) if (RO_NET_DEBUG) printf x
#else
# define DPRINTF(x) do {} while (0)
#endif

enum ro_net_cmds {
    RO_NET_PUT,
    RO_NET_GET,
    RO_NET_PUT_NBI,
    RO_NET_GET_NBI,
    RO_NET_FENCE,
    RO_NET_QUIET,
    RO_NET_FINALIZE,
    RO_NET_FLOAT_SUM_TO_ALL,
    RO_NET_BARRIER_ALL,
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
} __attribute__((__aligned__(64))) queue_element_t;

typedef struct host_stats {
    uint64_t numGet;
    uint64_t numGetNbi;
    uint64_t numPut;
    uint64_t numPutNbi;
    uint64_t numQuiet;
    uint64_t numFinalize;
} host_stats_t;

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
    host_stats_t host_stats;
} __attribute__((__aligned__(64))) queue_desc_t;


typedef struct profiler {
    uint64_t waitingOnSlot;
    uint64_t threadFence1;
    uint64_t threadFence2;
    uint64_t waitingOnHost;
    uint64_t packQueue;
    uint64_t shmem_wait;
    int enabled;
} profiler_t;

struct ro_net_handle {
    queue_element_t **queues;
    queue_desc_t *queue_descs;
    profiler_t   *profiler;
    int num_wgs;
    int num_queues;
    int num_threads;
    pthread_t *worker_threads;
    bool done_flag;
    // 1 if available, 0 if in use
    unsigned int *queueTokens;
    unsigned int *barrier_ptr;
    int num_pes;
    int my_pe;
    bool *needs_quiet;
    bool *needs_blocking;
    uint64_t queue_size;
    hsa_amd_hdp_flush_t *hdp_regs;
};

/* Meant for local allocation on the GPU */
struct ro_net_wg_handle {
    queue_element_t *queue;
    profiler_t      profiler;
    unsigned int *queueTokens;
    unsigned int *barrier_ptr;
    uint64_t read_idx;
    uint64_t write_idx;
    uint64_t *host_read_idx;
    uint64_t queue_size;
    char *status;
    int queueTokenIndex;
    int num_queues;
    int num_pes;
    int my_pe;
    volatile unsigned int *hdp_flush;
};

typedef struct pthread_args {
    int thread_id;
    int num_threads;
    struct ro_net_handle *ro_net_gpu_handle;
} pthread_args_t;

void load_elmt (__m256i* next_element, char* reg);

/* Host-side internal functions */
ro_net_status_t ro_net_free_runtime(
    struct ro_net_handle * ro_net_gpu_handle);

bool ro_net_process_queue(int queue_idx,
                            struct ro_net_handle * ro_net_gpu_handle,
                            bool *finalize);

void *ro_net_poll(void* args);

inline void ro_net_progress(int wg_id,
                              struct ro_net_handle *ronet_gpu_handle);

void ro_net_device_uc_malloc(void **ptr, size_t size);

/* Device-side internal functions */
__device__ void inline  __ro_inv() { asm volatile ("buffer_wbinvl1_vol;"); }
__device__ uint64_t inline  __read_clock() {
    uint64_t clock;
    asm volatile ("s_memrealtime %0\n\t"
                  "s_waitcnt lgkmcnt(0)\n\t"
                    : "=s" (clock));
    return clock;
}

__device__ bool isFull(uint64_t read_idx, uint64_t write_idx, int queue_size);

__device__ void build_queue_element(ro_net_cmds type, void* dst, void * src,
                                    size_t size, int pe, int logPE_stride,
                                    int PE_size, void* pWrk,
                                    long* pSync,
                                    struct ro_net_wg_handle *handle,
                                    bool blocking);

/* Currently unused asm that might be useful in the future.

__device__ void gws_barrier_init(unsigned int bar_num, unsigned int bar_val,
                                 unsigned int *bar_inited);

__device__ void gws_barrier_wait(unsigned int bar_num, unsigned int reset_val);


__device__  int1  inline __load_dword (volatile int1* src)
{
        int1 val;
        asm  volatile("flat_load_dword %0 %1 glc slc \ns_waitcnt vmcnt(0)"
                      : "=v" (val): "v"(src));
        return val;
}

__device__  void inline __atomic_store (int1 val, volatile int1* dst)
{
        asm  volatile("flat_atomic_swap %0 %1 slc \ns_waitcnt  lgkmcnt(0)"
                      : : "v"(dst), "v" (val));
}

__device__  int1 inline __atomic_load (volatile int1* src)
{
        int1 val;
        uint1 zero=0;
        asm  volatile("flat_atomic_or %0 %1 %2 glc slc" : "=v" (val)
                      : "v"(src), "v" (zero));
        return val;
}

__device__ void inline __store_byte (char1 val, volatile char1* dst)
{
        asm  volatile("flat_store_byte %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}
__device__ void inline __store_short (short1 val, volatile short1* dst)
{
        asm  volatile("flat_store_short %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}
__device__  void inline __store_dword (int1 val, volatile int1* dst)
{
        asm  volatile("flat_store_dword %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}

__device__ void inline __store_dwordx2 (int2 val, volatile int2* dst)
{
        asm  volatile("flat_store_dwordx2 %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}
__device__ void inline __store_dwordx3 (int3 val, volatile int3* dst)
{
        asm  volatile("flat_store_dwordx3 %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}
__device__ void inline __store_dwordx4 (int4 val, volatile int4* dst)
{
        asm  volatile("flat_store_dwordx4 %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}

__device__ void inline __store_long_dwordx2 (long1 val, volatile long1* dst)
{
        asm  volatile("flat_store_dwordx2 %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}

__device__ void inline __store_long_dwordx4 (long2 val, volatile long2* dst)
{
        asm  volatile("flat_store_dwordx4 %0 %1 glc slc"
                      : : "v"(dst), "v" (val));
}
*/

#endif
