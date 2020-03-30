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

#ifndef BACKEND_H
#define BACKEND_H

#include <roc_shmem.hpp>
#include <stats.hpp>
#include "config.h"

/**
 * @file backend.hpp
 * @brief Interface for Reverse Offload or GPU Native backends.  Unfortunately,
 * polymorphism is not implemented in the shader compiler, so we've got to
 * keep the device side function separate from the classes.
 */

enum BackendType {
    RO_BACKEND,
    GPU_IB_BACKEND
};

/*
 * Each roc_shmem_handle_t or roc_shmem_ctx_t is an opaque pointer to these
 * internal types.  These guys hold information generic to all backends, as
 * well as a pointer to a backend specific handle.  A ctx is per work-group
 * and the handle is for the entire GPU.
 *
 * The idea is to populate the ctx version at ctx creation with any information
 * it needs from the global handle, which will ideally be placed in shared
 * memory.
 *
 * WARNING: These are GPU-accessible structures, so alignment is absolutely
 * critical!
 *
 * TODO: It is expected that the handle and ctx will contain different
 * information eventually, so keep them separate types for now.
 *
 */

enum roc_shmem_stats {
    NUM_PUT = 0,
    NUM_PUT_NBI,
    NUM_P,
    NUM_GET,
    NUM_G,
    NUM_GET_NBI,
    NUM_FENCE,
    NUM_QUIET,
    NUM_TO_ALL,
    NUM_BARRIER_ALL,
    NUM_WAIT_UNTIL,
    NUM_FINALIZE,
    NUM_MSG_COAL,
    NUM_ATOMIC_FADD,
    NUM_ATOMIC_FCSWAP,
    NUM_ATOMIC_FINC,
    NUM_ATOMIC_FETCH,
    NUM_ATOMIC_ADD,
    NUM_ATOMIC_CSWAP,
    NUM_ATOMIC_INC,
    NUM_TEST,
    NUM_STATS
};


const int WF_SIZE = 64;

#ifdef PROFILE
typedef Stats<NUM_STATS> ROCStats;
#else
typedef NullStats<NUM_STATS> ROCStats;
#endif

class Backend;

class Context
{
  public:
    int num_pes = 0;
    int my_pe = -1;
    BackendType type = GPU_IB_BACKEND;

    ROCStats ctxStats;

  protected:

    // Ptr to current base of dynamic shared region for this work-group.
    char *dynamicPtr = nullptr;

  public:

    /*
     * Allocate memory from dynamic shared segment.  Must be called as a
     * work-group collective.
     */
    __device__ char *allocateDynamicShared(size_t size);

    /*
     * Attempts to coalesce basic RMA-style messages accross threads in the
     * same wavefront. If false, the calling thread was absorbed into one
     * of its peers and can simply return.  If true, the calling thread
     * must continue sending the message with the potentially updated
     * 'size' field.
     */
    __device__ bool wavefrontNetCoalescer(int pe, const void *source,
                                          const void *dest, size_t &size);

    __device__ Context(const Backend &handle);

    template <typename T>
    __device__ void wait_until(T *ptr, roc_shmem_cmps cmp, T val);

    template <typename T>
    __device__ int test(T *ptr, roc_shmem_cmps cmp, T val);

    /*
     * Dispatch functions to get runtime polymorphism without 'virtual' or
     * function pointers. Each one of these guys will use 'type' to static_cast
     * themsleves and dispatch to the appropriate derived class.  It's
     * basically doing part of what the 'virtual' keyword does, so when we get
     * that working in ROCm it will be super easy to adapt to it by just
     * removing the dispatch implementations.
     *
     * No comments for these guys since its basically the same as in the
     * roc_shmem.hpp public header.
     */
    __device__ void threadfence_system();

    __device__ void ctx_destroy();

    __device__ void
    putmem(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    getmem(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    putmem_nbi(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    getmem_nbi(void *dest, const void *source, size_t size, int pe);

    __device__ void fence();

    __device__ void quiet();

    __device__ void barrier_all();

    __device__ int64_t
    amo_fetch(void *dst, int64_t value, int64_t cond, int pe,
              uint8_t atomic_op);

    __device__ void
    amo(void *dst, int64_t value, int64_t cond, int pe, uint8_t atomic_op);

    template <typename T> __device__ void p(T *dest, T value, int pe);

    template <typename T> __device__ T g(T *source, int pe);

    template <typename T, ROC_SHMEM_OP Op> __device__ void
    to_all(T *dest, const T *source, int nreduce, int PE_start,
           int logPE_stride, int PE_size, T *pWrk, long *pSync);

    template <typename T> __device__ void
    put(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    put_nbi(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    get(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    get_nbi(T *dest, const T *source, size_t nelems, int pe);
};

struct ro_net_wg_handle;

class ROContext : public Context
{
    ro_net_wg_handle *backend_ctx = nullptr;

  public:

    __device__ ROContext(const Backend &b);

    __device__ void ctx_create(long option);

    /*
     * Implementations of Context dispatch functions.
    */
    __device__ void threadfence_system();

    __device__ void ctx_destroy();

    __device__ void
    putmem(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    getmem(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    putmem_nbi(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    getmem_nbi(void *dest, const void *source, size_t size, int pe);

    __device__ void fence();

    __device__ void quiet();

    __device__ void barrier_all();

    __device__ int64_t
    amo_fetch(void *dst, int64_t value, int64_t cond, int pe,
              uint8_t atomic_op);

    __device__ void
    amo(void *dst, int64_t value, int64_t cond, int pe, uint8_t atomic_op);

    template <typename T> __device__ void p(T *dest, T value, int pe);

    template <typename T> __device__ T g(T *source, int pe);

    template <typename T, ROC_SHMEM_OP Op> __device__ void
    to_all(T *dest, const T *source, int nreduce, int PE_start,
           int logPE_stride, int PE_size, T *pWrk, long *pSync);

    template <typename T> __device__ void
    put(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    put_nbi(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    get(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    get_nbi(T *dest, const T *source, size_t nelems, int pe);
};

class QueuePair;

class GPUIBContext : public Context
{
    QueuePair *rtn_gpu_handle = nullptr;
    char **base_heap = nullptr;
    int64_t *barrier_sync = nullptr;
    size_t current_heap_offset = 0;
    char *g_ret = nullptr;
#ifdef _USE_IPC_
    char **ipc_bases = nullptr;
    uint8_t shm_size = 0;
#endif
    uint32_t queue_id = 0;

    template <typename T, ROC_SHMEM_OP Op> __device__ void
    internal_direct_allreduce(T *dst, const T *src, int nelems, int PE_start,
                              int logPE_stride, int PE_size, T *pWrk,
                              long *pSync);
    __device__ void
    internal_direct_barrier(int pe, int n_pes, int64_t *pSync);

    __device__ void
    internal_atomic_barrier(int pe, int n_pes, int64_t *pSync);

    __device__ void quiet_single(int cq_num);

    __device__ QueuePair* getQueuePair(int pe);

    __device__ int getNumQueuePairs();

  public:

    __device__ GPUIBContext(const Backend &b);

    __device__ void ctx_create(long option);

    /*
     * Implementations of Context dispatch functions.
    */
    __device__ void threadfence_system();

    __device__ void ctx_destroy();

    __device__ void
    putmem(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    getmem(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    putmem_nbi(void *dest, const void *source, size_t nelems, int pe);

    __device__ void
    getmem_nbi(void *dest, const void *source, size_t size, int pe);

    __device__ void fence();

    __device__ void quiet();

    __device__ void barrier_all();

    __device__ int64_t
    amo_fetch(void *dst, int64_t value, int64_t cond, int pe,
              uint8_t atomic_op);

    __device__ void
    amo(void *dst, int64_t value, int64_t cond, int pe, uint8_t atomic_op);

    template <typename T> __device__ void p(T *dest, T value, int pe);

    template <typename T> __device__ T g(T *source, int pe);

    template <typename T, ROC_SHMEM_OP Op> __device__ void
    to_all(T *dest, const T *source, int nreduce, int PE_start,
           int logPE_stride, int PE_size, T *pWrk, long *pSync);

    template <typename T> __device__ void
    put(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    put_nbi(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    get(T *dest, const T *source, size_t nelems, int pe);

    template <typename T> __device__ void
    get_nbi(T *dest, const T *source, size_t nelems, int pe);
};

extern __constant__ Backend *gpu_handle;

class Backend
{
   /*
    * TODO: Duplication of functionality still exists in the derived classes.
    * Need to push more into base class.
    */
  public:

    virtual roc_shmem_status_t pre_init() = 0;
    virtual roc_shmem_status_t init(int num_queues) = 0;
    virtual roc_shmem_status_t finalize() = 0;
    virtual roc_shmem_status_t net_malloc(void **ptr, size_t size) = 0;
    virtual roc_shmem_status_t net_free(void *ptr) = 0;
    virtual roc_shmem_status_t dynamic_shared(size_t *shared_bytes) = 0;

    __host__ __device__ int getMyPE() const { return my_pe; }
    __host__ __device__ int getNumPEs() const { return num_pes; }

    roc_shmem_status_t dump_stats();
    roc_shmem_status_t reset_stats();

    Backend();

    virtual ~Backend();

    // TODO: Push backendend_handle into derived classes with the correct type
    // to avoid a billion casts.
    void *backend_handle = nullptr;
    int *print_lock = nullptr;
    BackendType type = GPU_IB_BACKEND;
    ROCStats globalStats;

  protected:
    virtual roc_shmem_status_t dump_backend_stats() = 0;
    virtual roc_shmem_status_t reset_backend_stats() = 0;

    int num_pes = 0;
    int my_pe = -1;
};

class Transport;
struct ro_net_handle;
struct roc_shmem;

/*
 * ROBackend (Revere Offload Transports)forwards GPU Requests to the host.
 */
class ROBackend : public Backend
{
  public:
    roc_shmem_status_t pre_init() override;
    roc_shmem_status_t init(int num_queues) override;
    roc_shmem_status_t finalize() override;
    roc_shmem_status_t net_malloc(void **ptr, size_t size) override;
    roc_shmem_status_t net_free(void *ptr) override;
    roc_shmem_status_t dynamic_shared(size_t *shared_bytes) override;

    ROBackend() { }

    virtual ~ROBackend() { }

    roc_shmem_status_t ro_net_free_runtime(ro_net_handle *handle);
    bool ro_net_process_queue(int queue_idx,
                              struct ro_net_handle *ro_net_gpu_handle,
                              bool *finalized);
    void ro_net_device_uc_malloc(void **ptr, size_t size);

  protected:
    roc_shmem_status_t dump_backend_stats() override;
    roc_shmem_status_t reset_backend_stats() override;
    void ro_net_poll(int thread_id, int num_threads);

    char *elt = nullptr;
    Transport *transport = nullptr;
    std::vector<std::thread> worker_threads;
};

/*
 * GPU IB Backend talks to IB adaptors directly from the GPU.
 */

class  GPUIBBackend : public Backend
{
  public:
    roc_shmem_status_t pre_init() override;
    roc_shmem_status_t init(int num_queues) override;
    roc_shmem_status_t finalize() override;
    roc_shmem_status_t net_malloc(void **ptr, size_t size) override;
    roc_shmem_status_t net_free(void *ptr) override;
    roc_shmem_status_t dynamic_shared(size_t *shared_bytes) override;

    GPUIBBackend() { }

    virtual ~GPUIBBackend() { }

  protected:
    roc_shmem_status_t dump_backend_stats() override;
    roc_shmem_status_t reset_backend_stats() override;
    void roc_shmem_collective_init();
    void roc_shmem_g_init();
    void thread_cpu_post_wqes();
    void thread_func(int sleep_time);

    volatile bool first_time = true;
    std::thread *worker_thread = nullptr;
};

#define DISPATCH(Func) \
    switch (type) { \
        case RO_BACKEND: static_cast<ROContext*>(this)->Func; break; \
        case GPU_IB_BACKEND: static_cast<GPUIBContext*>(this)->Func; break; \
        default: break; \
    }

#define DISPATCH_RET(Func) \
    switch (type) { \
        case RO_BACKEND: return static_cast<ROContext*>(this)->Func; \
            break; \
        case GPU_IB_BACKEND: return static_cast<GPUIBContext*>(this)->Func; \
            break; \
        default: break; \
    }

/*
 * Context dispatch implementations for the template functions.  Needs to
 * be in a header and not cpp because it is a template.
 */
template <typename T> __device__ void
Context::p(T *dest, T value, int pe) { DISPATCH(p(dest, value, pe)); }

template <typename T> __device__ T
Context::g(T *source, int pe) { DISPATCH_RET(g(source, pe)); }

// The only way to get multi-arg templates to feed into a macro
#define PAIR(A, B) A, B
template <typename T, ROC_SHMEM_OP Op> __device__ void
Context::to_all(T *dest, const T *source, int nreduce, int PE_start,
                      int logPE_stride, int PE_size, T *pWrk, long *pSync)
{ DISPATCH(to_all<PAIR(T, Op)>(dest, source, nreduce, PE_start, logPE_stride,
                               PE_size, pWrk, pSync)); }

template <typename T> __device__ void
Context::put(T *dest, const T *source, size_t nelems, int pe)
{ DISPATCH(put(dest, source, nelems, pe)); }

template <typename T> __device__ void
Context::put_nbi(T *dest, const T *source, size_t nelems, int pe)
{ DISPATCH(put_nbi(dest, source, nelems, pe)); }

template <typename T> __device__ void
Context::get(T *dest, const T *source, size_t nelems, int pe)
{ DISPATCH(get(dest, source, nelems, pe)); }

template <typename T> __device__ void
Context::get_nbi(T *dest, const T *source, size_t nelems, int pe)
{ DISPATCH(get_nbi(dest, source, nelems, pe)); }

#endif //BACKEND_H
