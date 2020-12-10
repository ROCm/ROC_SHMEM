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

#ifndef CONTEXT_H
#define CONTEXT_H

#include "util.hpp"
#include "stats.hpp"
#include "wf_coal_policy.hpp"

#include "gpu_ib/ipc_policy.hpp"

class Backend;

/**
 * Context class corresponds directly to the concept of an OpenSHMEM context.
 *
 * GPUs perform networking operations on a context that is created by the
 * application programmer or a "default context" managed by the runtime.
 *
 * Contexts can be allocated in shared memory, in which case they are private
 * to the creating workgroup, or they can be allocated in global memory, in
 * which case they are shareable across workgroups.
 *
 * This is an 'abstract' class, as much as there is such a thing on a GPU.
 * It uses 'type' to dispatch to a derived class for most of the interesting
 * behavior.
 */
class Context
{
  public:
    int num_pes = 0;

    int my_pe = -1;

    /**
     * Used only to dispatch to the correct derived type. This is used to
     * get around the fact that there are no virtual functions for device code.
     * See the 'DISPATCH' macro and usage for more details.
     */
    BackendType type = BackendType::GPU_IB_BACKEND;

    /**
     * Stats common to all types of Contexts.
     */
    ROCStats ctxStats;

  protected:
    /**
     * Context can be shared between different workgroups.
     *
     * TODO: Might consider refactor into its own class so we don't have to
     * instantiate these guys for unshareable contexts. Most contexts
     * 'shareable' behavior will be selected by the user at runtime, so static
     * policy selection won't work here.
     */
    bool shareable = false;

    /**
     * Shareable context lock.
     */
    int ctx_lock = 0;

    /**
     * Shareable context owner.
     */
    volatile int wg_owner = -1;

    /**
     * Num threads in the owning workgroup inside of locked OpenSHMEM calls.
     */
    volatile int num_threads_in_lock = 0;

    __device__ void lock();
    __device__ void unlock();

    /**
     * Coalesce policy for 'multi' configuration builds
     */
    WavefrontCoalescer wf_coal;

  public:
     /*
     * if the context is created with option SHMEM_CTX_NOSTORE means do not flush
     * LD/ST operations (ie do not do __threadfence()), by default it is OFF
    */
    bool flush_stores = true;

    __device__ void flushStores(){if (flush_stores) __threadfence();}

    __host__ Context(const Backend &handle, bool shareable);

    __device__ Context(const Backend &handle, bool shareable);

    /**
     * Dispatch functions to get runtime polymorphism without 'virtual' or
     * function pointers. Each one of these guys will use 'type' to static_cast
     * themselves and dispatch to the appropriate derived class. It's
     * basically doing part of what the 'virtual' keyword does, so when we get
     * that working in ROCm it will be super easy to adapt to it by just
     * removing the dispatch implementations.
     *
     * No comments for these guys since its basically the same as in the
     * roc_shmem.hpp public header.
     */
    template <typename T>
    __device__ void wait_until(T *ptr, roc_shmem_cmps cmp, T val);

    template <typename T>
    __device__ int test(T *ptr, roc_shmem_cmps cmp, T val);

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

    __device__ void* shmem_ptr(const void* dest, int pe);

    __device__ void barrier_all();

    __device__ int64_t
    amo_fetch(void *dst, int64_t value, int64_t cond, int pe,
              uint8_t atomic_op);

    __device__ void sync_all();

    __device__ void
    amo_add(void *dst, int64_t value, int64_t cond, int pe);

    __device__ void
    amo_cas(void *dst, int64_t value, int64_t cond, int pe);

    __device__ int64_t
    amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe);

    __device__ int64_t
    amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe);

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

    template <typename T>
    __device__ void
    broadcast(T *dest,
              const T *source,
              int nelems,
              int pe_root,
              int pe_start,
              int log_pe_stride,
              int pe_size,
              long *p_sync);
};

struct ro_net_wg_handle;

class ROContext : public Context
{
    ro_net_wg_handle *backend_ctx = nullptr;

  public:

    __host__ ROContext(const Backend &b, long options);

    __device__ ROContext(const Backend &b, long options);

    /**
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

    __device__ void* shmem_ptr(const void* dest, int pe);

    __device__ void barrier_all();

    __device__ void sync_all();

    __device__ void
    amo_add(void *dst, int64_t value, int64_t cond, int pe);

    __device__ void
    amo_cas(void *dst, int64_t value, int64_t cond, int pe);

    __device__ int64_t
    amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe);

    __device__ int64_t
    amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe);

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

    template <typename T>
    __device__ void
    broadcast(T *dest,
              const T *source,
              int nelems,
              int pe_root,
              int pe_start,
              int log_pe_stride,
              int pe_size,
              long *p_sync);
};

class QueuePair;
class GPUIBBackend;

class GPUIBContext : public Context
{
public:
    /**
     * Collection of queue pairs that are currently checked out by this
     * context from GPUIBBackend.
     */
    //TODO: keep it private and destroy in destructor for better encapsulation.
    QueuePair *rtn_gpu_handle = nullptr;
private:
    /**
     * Array of char * pointers corresponding to the heap base pointers VA for
     * each PE that we can communicate with.
     */
    char **base_heap = nullptr;

    /**
     * Temporary scratchpad memory used by internal barrier algorithms.
     */
    int64_t *barrier_sync = nullptr;

    size_t current_heap_offset = 0;

    /**
     * Buffer used to store the results of a *_g operation. These ops do not
     * provide a destination buffer, so the runtime must manage one.
     */
    char *g_ret = nullptr;

    template <typename T, ROC_SHMEM_OP Op> __device__ void
    internal_direct_allreduce(T *dst, const T *src, int nelems, int PE_start,
                              int logPE_stride, int PE_size, T *pWrk,
                              long *pSync);

    template <typename T, ROC_SHMEM_OP Op> __device__ void
    internal_ring_allreduce(T *dst, const T *src, int nelems,
                            int PE_start, int logPE_stride,
                            int PE_size, T *pWrk, long *pSync,
                            int n_seg, int seg_size, int chunk_size);

    template <typename T> __device__ void
    internal_put_broadcast(T *dst, const T *src, int nelems, int pe_root,
                           int PE_start, int logPE_stride, int PE_size,
                           long *pSync);

    template <typename T> __device__ void
    internal_get_broadcast(T *dst, const T *src, int nelems, int pe_root,
                           long *pSync);


    __device__ void
    internal_direct_barrier(int pe, int n_pes, int64_t *pSync);

    __device__ void
    internal_atomic_barrier(int pe, int n_pes, int64_t *pSync);

    __device__ void quiet_single(int cq_num);

    __device__ __host__ QueuePair* getQueuePair(int pe);

    __device__ __host__ int getNumQueuePairs();

  public:

    IpcImpl ipcImpl;

    __device__ GPUIBContext(const Backend &b, long options);

    __host__ GPUIBContext(const Backend &b, long options);

    /**
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

    __device__ void* shmem_ptr(const void* dest, int pe);

    __device__ void barrier_all();

    __device__ void sync_all();

    __device__ void
    amo_add(void *dst, int64_t value, int64_t cond, int pe);

    __device__ void
    amo_cas(void *dst, int64_t value, int64_t cond, int pe);

    __device__ int64_t
    amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe);

    __device__ int64_t
    amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe);

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

    template <typename T>
    __device__ void
    broadcast(T *dest,
              const T *source,
              int nelems,
              int pe_root,
              int pe_start,
              int log_pe_stride,
              int pe_size,
              long *p_sync);
};

#define DISPATCH(Func) \
    lock(); \
    switch (type) { \
        case BackendType::RO_BACKEND: static_cast<ROContext*>(this)->Func; break; \
        case BackendType::GPU_IB_BACKEND: static_cast<GPUIBContext*>(this)->Func; break; \
        default: break; \
    } \
    unlock();

#define DISPATCH_RET(Func) \
    lock(); \
    auto ret_val = 0; \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            ret_val = static_cast<ROContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            ret_val = static_cast<GPUIBContext*>(this)->Func; \
            break; \
        default: break; \
    } \
    unlock(); \
    return ret_val;

#define DISPATCH_RET_PTR(Func) \
    lock(); \
    void *ret_val = NULL; \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            ret_val = static_cast<ROContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            ret_val = static_cast<GPUIBContext*>(this)->Func; \
            break; \
        default: break; \
    } \
    unlock(); \
    return ret_val;

/**
 * Context dispatch implementations for the template functions. Needs to
 * be in a header and not cpp because it is a template.
 */
template <typename T> __device__ void
Context::p(T *dest, T value, int pe)
{
    ctxStats.incStat(NUM_P);

    /**
     * TODO: Need to handle _p a bit differently for coalescing, since the
     * owner of a coalesced message needs val from all absorbed messages.
     */
     DISPATCH(p(dest, value, pe));
}

template <typename T> __device__ T
Context::g(T *source, int pe)
{
    ctxStats.incStat(NUM_G);

    /**
     * TODO: Need to handle _g a bit differently for coalescing, since the
     * owner of a coalesced message needs val from all absorbed messages.
     */
    DISPATCH_RET(g(source, pe));
}

// The only way to get multi-arg templates to feed into a macro
#define PAIR(A, B) A, B
template <typename T, ROC_SHMEM_OP Op> __device__ void
Context::to_all(T *dest, const T *source, int nreduce, int PE_start,
                      int logPE_stride, int PE_size, T *pWrk, long *pSync)
{
    if (nreduce == 0)
        return;

    if (is_thread_zero_in_block())
        ctxStats.incStat(NUM_TO_ALL);

    DISPATCH(to_all<PAIR(T, Op)>(dest, source, nreduce, PE_start, logPE_stride,
                                PE_size, pWrk, pSync));
}

template <typename T> __device__ void
Context::put(T *dest, const T *source, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctxStats.incStat(NUM_PUT);

    DISPATCH(put(dest, source, nelems, pe));
}

template <typename T> __device__ void
Context::put_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctxStats.incStat(NUM_PUT_NBI);

    DISPATCH(put_nbi(dest, source, nelems, pe));
}

template <typename T> __device__ void
Context::get(T *dest, const T *source, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctxStats.incStat(NUM_GET);

    DISPATCH(get(dest, source, nelems, pe));
}

template <typename T> __device__ void
Context::get_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    if (nelems == 0)
        return;

    ctxStats.incStat(NUM_GET_NBI);

    DISPATCH(get_nbi(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::broadcast(T *dest,
                   const T *source,
                   int nelems,
                   int pe_root,
                   int pe_start,
                   int log_pe_stride,
                   int pe_size,
                   long *p_sync)
{
    if (nelems == 0) {
        return;
    }

    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_BROADCAST);
    }

    DISPATCH(broadcast<T>(dest,
                          source,
                          nelems,
                          pe_root,
                          pe_start,
                          log_pe_stride,
                          pe_size,
                          p_sync));
}

template <typename T> __device__ void
Context::wait_until(T *ptr, roc_shmem_cmps cmp, T val)
{
    while (!test(ptr, cmp, val)) __roc_inv();
}

template <typename T> __device__ int
Context::test(T *ptr, roc_shmem_cmps cmp, T val)
{
    int ret = 0;
    volatile T * vol_ptr = (T*) ptr;
    __roc_inv();
    switch (cmp) {
        case ROC_SHMEM_CMP_EQ:
            if (*vol_ptr == val) ret = 1;
            break;
        case ROC_SHMEM_CMP_NE:
            if (*vol_ptr != val) ret = 1;
            break;
        case ROC_SHMEM_CMP_GT:
            if (*vol_ptr > val) ret = 1;
            break;
        case ROC_SHMEM_CMP_GE:
            if (*vol_ptr >= val) ret = 1;
            break;
        case ROC_SHMEM_CMP_LT:
            if (*vol_ptr < val) ret = 1;
            break;
        case ROC_SHMEM_CMP_LE:
            if (*vol_ptr <= val) ret = 1;
            break;
        default:
            break;
    }
    return ret;
}

#endif // CONTEXT_H
