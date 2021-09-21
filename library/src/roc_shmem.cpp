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

/**
 * @file roc_shmem.cpp
 * @brief Public header for ROC_SHMEM device and host libraries.
 *
 * This is the implementation for the public roc_shmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 *
 * The device-side delegation is nasty because we can't use polymorphism with
 * our current shader compiler stack.  Maybe one day.....
 *
 * TODO: Could probably autogenerate many of these functions from macros.
 *
 * TODO: Support runtime backend detection.
 *
 */

#include <cstdlib>
#include <roc_shmem.hpp>

#include "context.hpp"
#include "backend.hpp"
#include "util.hpp"
#include "templates_host.hpp"

#include "gpu_ib/gpu_ib_host_templates.hpp"
#include "reverse_offload/ro_net_host_templates.hpp"

#define VERIFY_BACKEND() {                                                   \
        if (!backend) {                                                      \
            fprintf(stderr, "ROC_SHMEM_ERROR: %s in file '%s' in line %d\n", \
                            "Call 'roc_shmem_init'", __FILE__, __LINE__);\
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

Backend *backend = nullptr;

bool ROC_SHMEM_DEBUG = false;

Context *ROC_SHMEM_HOST_CTX_DEFAULT;

/**
 * Begin Host Code
 **/

__host__ Status inline
library_init(unsigned num_wgs)
{
    assert(!backend);

    if (getenv("ROC_SHMEM_DEBUG") != nullptr)
        ROC_SHMEM_DEBUG = true;

    rocm_init();
    if (getenv("ROC_SHMEM_RO") != nullptr) {
        hipHostMalloc(&backend, sizeof(ROBackend));
        backend = new (backend) ROBackend(num_wgs);
    } else {
        hipHostMalloc(&backend, sizeof(GPUIBBackend));
        backend = new (backend) GPUIBBackend(num_wgs);
    }

    if (!backend)
        return Status::ROC_SHMEM_OOM_ERROR;

    return Status::ROC_SHMEM_SUCCESS;
}

[[maybe_unused]]
__host__ Status
roc_shmem_init(unsigned num_wgs)
{
    return library_init(num_wgs);
}

[[maybe_unused]]
__host__ int
roc_shmem_init_thread(int required, int *provided, unsigned num_wgs)
{
    int ret;
    Status status;

    status = library_init(num_wgs);

    if (status == Status::ROC_SHMEM_SUCCESS) {
        roc_shmem_query_thread(provided);
        ret = 0;        /* successful */
    } else {
        ret = -1;       /* erroneous non-zero value */
    }

    return ret;
}

[[maybe_unused]]
__host__ int
roc_shmem_my_pe()
{
    VERIFY_BACKEND();
    return backend->getMyPE();
}

[[maybe_unused]]
__host__ int
roc_shmem_n_pes()
{
    VERIFY_BACKEND();
    return backend->getNumPEs();
}

[[maybe_unused]]
__host__ void *
roc_shmem_malloc(size_t size)
{
    VERIFY_BACKEND();

    void *ptr;
    backend->net_malloc(&ptr, size);
    return ptr;
}

[[maybe_unused]]
__host__ Status
roc_shmem_free(void *ptr)
{
    VERIFY_BACKEND();
    return backend->net_free(ptr);
}

[[maybe_unused]]
__host__ Status
roc_shmem_reset_stats()
{
    VERIFY_BACKEND();
    return backend->reset_stats();
}

[[maybe_unused]]
__host__ Status
roc_shmem_dump_stats()
{
    /** TODO: Many stats are backend independent! **/
    VERIFY_BACKEND();
    return backend->dump_stats();
}

[[maybe_unused]]
__host__ Status
roc_shmem_finalize()
{
    VERIFY_BACKEND();

    backend->~Backend();
    hipHostFree(backend);

    return Status::ROC_SHMEM_SUCCESS;
}

[[maybe_unused]]
__host__ Status
roc_shmem_dynamic_shared(size_t *shared_bytes)
{
    VERIFY_BACKEND();
    backend->dynamic_shared(shared_bytes);
    return Status::ROC_SHMEM_SUCCESS;
}

__host__ void
roc_shmem_query_thread(int *provided)
{
    /*
     * Host-facing functions always support full
     * thread flexibility i.e. THREAD_MULTIPLE.
     */
    *provided = ROC_SHMEM_THREAD_MULTIPLE;
}

__host__ void
roc_shmem_global_exit(int status)
{
    VERIFY_BACKEND();
    backend->global_exit(status);
}


/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

template <typename T>
__host__ void
roc_shmem_put(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_put((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_putmem(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_putmem((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ void
roc_shmem_p(T *dest, T value, int pe)
{
    roc_shmem_p((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ void
roc_shmem_get(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_get((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_getmem(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_getmem((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ T
roc_shmem_g(const T *source, int pe)
{
    return roc_shmem_g((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, source, pe);
}

template <typename T>
__host__ void
roc_shmem_put_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_put_nbi((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_putmem_nbi((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ void
roc_shmem_get_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_get_nbi((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_getmem_nbi((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_fetch_add(T *dest, T val, int pe)
{
    return roc_shmem_atomic_fetch_add((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_compare_swap(T *dest, T cond, T val, int pe)
{
    return roc_shmem_atomic_compare_swap((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT,
                                        dest, cond, val, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_fetch_inc(T *dest, int pe)
{
    return roc_shmem_atomic_fetch_inc((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_fetch(T *dest, int pe)
{
    return roc_shmem_atomic_fetch((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, pe);
}

template <typename T>
__host__ void
roc_shmem_atomic_add(T *dest, T val, int pe)
{
    roc_shmem_atomic_add((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ void
roc_shmem_atomic_inc(T *dest, int pe)
{
    roc_shmem_atomic_inc((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT, dest, pe);
}

__host__ void
roc_shmem_fence()
{
    roc_shmem_ctx_fence((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT);
}

__host__ void
roc_shmem_quiet()
{
    roc_shmem_ctx_quiet((roc_shmem_ctx_t) ROC_SHMEM_HOST_CTX_DEFAULT);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

template <typename T> __host__ void
roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    DPRINTF(("Host function: roc_shmem_put\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->put(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_putmem(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Host function: roc_shmem_ctx_putmem\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->putmem(dest, source, nelems, pe);
}

template <typename T> __host__ void
roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe)
{
    DPRINTF(("Host function: roc_shmem_p\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->p(dest, value, pe);
}

template <typename T> __host__ void
roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems,
              int pe)
{
    DPRINTF(("Host function: roc_shmem_get\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->get(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_getmem(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Host function: roc_shmem_ctx_getmem\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->getmem(dest, source, nelems, pe);
}

template <typename T> __host__ T
roc_shmem_g(roc_shmem_ctx_t ctx, const T *source, int pe)
{
    DPRINTF(("Host function: roc_shmem_g\n"));

    return ROC_SHMEM_HOST_CTX_DEFAULT->g(source, pe);
}

template <typename T> __host__ void
roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    DPRINTF(("Host function: roc_shmem_put_nbi\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->put_nbi(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_putmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Host function: roc_shmem_ctx_putmem_nbi\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->putmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__host__ void
roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    DPRINTF(("Host function: roc_shmem_get_nbi\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->get_nbi(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_getmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Host function: roc_shmem_ctx_getmem_nbi\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->getmem_nbi(dest, source, nelems, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    DPRINTF(("Host function: roc_shmem_atomic_fetch_add\n"));

    return ROC_SHMEM_HOST_CTX_DEFAULT->amo_fetch_add(dest, val, 0, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest, T cond, T val,
                             int pe)
{
    DPRINTF(("Host function: roc_shmem_atomic_compare_swap\n"));

    return ROC_SHMEM_HOST_CTX_DEFAULT->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    DPRINTF(("Host function: roc_shmem_atomic_fetch_inc\n"));

    return ROC_SHMEM_HOST_CTX_DEFAULT->amo_fetch_add(dest, 1, 0, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    DPRINTF(("Host function: roc_shmem_atomic_fetch\n"));

    return ROC_SHMEM_HOST_CTX_DEFAULT->amo_fetch_add(dest, 0, 0, pe);
}

template <typename T> __host__ void
roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    DPRINTF(("Host function: roc_shmem_atomic_add\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->amo_add((void*)dest, val, 0, pe);
}

template <typename T>
__host__ void
roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    DPRINTF(("Host function: roc_shmem_atomic_inc\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->amo_add(dest, 1, 0, pe);
}

__host__ void
roc_shmem_ctx_fence(roc_shmem_ctx_t ctx)
{
    DPRINTF(("Host function: roc_shmem_ctx_fence\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->fence();
}

__host__ void
roc_shmem_ctx_quiet(roc_shmem_ctx_t ctx)
{
    DPRINTF(("Host function: roc_shmem_ctx_quiet\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->quiet();
}

__host__ void
roc_shmem_barrier_all()
{
    DPRINTF(("Host function: roc_shmem_barrier_all\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->barrier_all();
}

__host__ void
roc_shmem_sync_all()
{
    DPRINTF(("Host function: roc_shmem_sync_all\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->sync_all();
}

template <typename T>
__host__ void
roc_shmem_broadcast(roc_shmem_ctx_t ctx,
                    T *dest,
                    const T *source,
                    int nelem,
                    int pe_root,
                    int pe_start,
                    int log_pe_stride,
                    int pe_size,
                    long *p_sync)
{
    DPRINTF(("Host function: roc_shmem_broadcast\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->broadcast<T>(dest,
                                             source,
                                             nelem,
                                             pe_root,
                                             pe_start,
                                             log_pe_stride,
                                             pe_size,
                                             p_sync);
}

template <typename T, ROC_SHMEM_OP Op> __host__ void
roc_shmem_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source,
                 int nreduce, int PE_start, int logPE_stride,
                 int PE_size, T *pWrk, long *pSync)
{
    DPRINTF(("Host function: roc_shmem_to_all\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->to_all<T, Op>(dest, source, nreduce, PE_start,
                                         logPE_stride, PE_size, pWrk, pSync);
}

template <typename T>
__host__ void
roc_shmem_wait_until(T *ptr, roc_shmem_cmps cmp, T val)
{
    DPRINTF(("Host function: roc_shmem_wait_until\n"));

    ROC_SHMEM_HOST_CTX_DEFAULT->wait_until(ptr, cmp, val);
}

template <typename T>
__host__ int
roc_shmem_test(T *ptr, roc_shmem_cmps cmp, T val)
{
    DPRINTF(("Host function: roc_shmem_testl\n"));

    return ROC_SHMEM_HOST_CTX_DEFAULT->test(ptr, cmp, val);
}

/**
 * Template generator for reductions
 **/
#define REDUCTION_GEN(T, Op) \
    template __host__ void \
    roc_shmem_to_all<T, Op>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                            int nreduce, int PE_start, int logPE_stride, \
                            int PE_size, T *pWrk, long *pSync);

#define ARITH_REDUCTION_GEN(T) \
    REDUCTION_GEN(T, ROC_SHMEM_SUM) \
    REDUCTION_GEN(T, ROC_SHMEM_MIN) \
    REDUCTION_GEN(T, ROC_SHMEM_MAX) \
    REDUCTION_GEN(T, ROC_SHMEM_PROD)

#define BITWISE_REDUCTION_GEN(T) \
    REDUCTION_GEN(T, ROC_SHMEM_OR) \
    REDUCTION_GEN(T, ROC_SHMEM_AND) \
    REDUCTION_GEN(T, ROC_SHMEM_XOR)

#define INT_REDUCTION_GEN(T) \
    ARITH_REDUCTION_GEN(T) \
    BITWISE_REDUCTION_GEN(T)

#define FLOAT_REDUCTION_GEN(T) \
    ARITH_REDUCTION_GEN(T)

/**
 * Declare templates for the required datatypes (for the compiler)
 **/
#define RMA_GEN(T) \
    template __host__ void \
    roc_shmem_put<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                     size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_put_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_p<T>(roc_shmem_ctx_t ctx, T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_get<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                     size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_get_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe); \
    template __host__ T \
    roc_shmem_g<T>(roc_shmem_ctx_t ctx, const T *source, int pe); \
    template __host__ void \
    roc_shmem_put<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_put_nbi<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_p<T>(T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_get<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_get_nbi<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ T \
    roc_shmem_g<T>(const T *source, int pe); \
    template __host__ void \
    roc_shmem_broadcast<T>(roc_shmem_ctx_t ctx, \
                           T *dest, \
                           const T *source, \
                           int nelem, \
                           int pe_root, \
                           int pe_start, \
                           int log_pe_stride, \
                           int pe_size, \
                           long *p_sync);

#define AMO_GEN(T) \
    template __host__ T \
    roc_shmem_atomic_fetch_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, \
                                  int pe); \
    template __host__ T \
    roc_shmem_atomic_compare_swap<T>(roc_shmem_ctx_t ctx,  T *dest, T cond, \
                                    T value, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __host__ void \
    roc_shmem_atomic_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_atomic_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch_add<T>(T *dest, T value, int pe); \
    template __host__ T \
    roc_shmem_atomic_compare_swap<T>(T *dest, T cond, T value, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch_inc<T>(T *dest, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch<T>(T *dest, int pe); \
    template __host__ void \
    roc_shmem_atomic_add<T>(T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_atomic_inc<T>(T *dest, int pe);

#define WAIT_GEN(T) \
    template __host__ void \
    roc_shmem_wait_until<T>(T *ptr, roc_shmem_cmps cmp, T val); \
    template __host__ int \
    roc_shmem_test<T>(T *ptr, roc_shmem_cmps cmp, T val);


/**
* Define APIs to call the template functions
**/

#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op) \
    __host__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                              int nreduce, int PE_start, int logPE_stride, \
                                              int PE_size, T *pWrk, long *pSync) \
    { \
        roc_shmem_to_all<T, Op>(ctx, dest, source, nreduce, PE_start, \
                                logPE_stride, PE_size, pWrk, pSync); \
    }

#define ARITH_REDUCTION_DEF_GEN(T, TNAME) \
    REDUCTION_DEF_GEN(T, TNAME, sum, ROC_SHMEM_SUM) \
    REDUCTION_DEF_GEN(T, TNAME, min, ROC_SHMEM_MIN) \
    REDUCTION_DEF_GEN(T, TNAME, max, ROC_SHMEM_MAX) \
    REDUCTION_DEF_GEN(T, TNAME, prod, ROC_SHMEM_PROD)

#define BITWISE_REDUCTION_DEF_GEN(T, TNAME) \
    REDUCTION_DEF_GEN(T, TNAME, or, ROC_SHMEM_OR) \
    REDUCTION_DEF_GEN(T, TNAME, and, ROC_SHMEM_AND) \
    REDUCTION_DEF_GEN(T, TNAME, xor, ROC_SHMEM_XOR)

#define INT_REDUCTION_DEF_GEN(T, TNAME) \
    ARITH_REDUCTION_DEF_GEN(T, TNAME) \
    BITWISE_REDUCTION_DEF_GEN(T, TNAME)

#define FLOAT_REDUCTION_DEF_GEN(T, TNAME) \
    ARITH_REDUCTION_DEF_GEN(T, TNAME)

#define RMA_DEF_GEN(T, TNAME) \
    __host__ void \
    roc_shmem_ctx_##TNAME##_put(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                size_t nelems, int pe) \
    { \
        roc_shmem_put<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                    size_t nelems, int pe) \
    { \
        roc_shmem_put_nbi<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe) \
    { \
        roc_shmem_p<T>(ctx, dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_get(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                size_t nelems, int pe) \
    { \
        roc_shmem_get<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                    size_t nelems, int pe) \
    { \
        roc_shmem_get_nbi<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_g(roc_shmem_ctx_t ctx, const T *source, int pe) \
    { \
        return roc_shmem_g<T>(ctx, source, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_put(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_put<T>(dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_put_nbi(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_put_nbi<T>(dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_p(T *dest, T value, int pe) \
    { \
        roc_shmem_p<T>(dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_get(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_get<T>(dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_get_nbi(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_get_nbi<T>(dest, source, nelems, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_g(const T *source, int pe) \
    { \
        return roc_shmem_g<T>(source, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_broadcast(roc_shmem_ctx_t ctx, \
                                      T *dest, \
                                      const T *source, \
                                      int nelem, \
                                      int pe_root, \
                                      int pe_start, \
                                      int log_pe_stride, \
                                      int pe_size, \
                                      long *p_sync) \
    { \
        roc_shmem_broadcast<T>(ctx, dest, source, nelem, pe_root, pe_start, \
                               log_pe_stride, pe_size, p_sync); \
    }

#define AMO_DEF_GEN(T, TNAME) \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T value, \
                                             int pe) \
    { \
        return roc_shmem_atomic_fetch_add<T>(ctx, dest, value, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest, T cond, \
                                               T value, int pe) \
    { \
        return roc_shmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch_inc<T>(ctx, dest, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch<T>(ctx, dest, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_atomic_add(roc_shmem_ctx_t ctx, T *dest, T value, int pe) \
    { \
        roc_shmem_atomic_add<T>(ctx, dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        roc_shmem_atomic_inc<T>(ctx, dest, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch_add(T *dest, T value, int pe) \
    { \
        return roc_shmem_atomic_fetch_add<T>(dest, value, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_compare_swap(T *dest, T cond, T value, int pe) \
    { \
        return roc_shmem_atomic_compare_swap<T>(dest, cond, value, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch_inc(T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch_inc<T>(dest, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch(T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch<T>(dest, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_atomic_add(T *dest, T value, int pe) \
    { \
        roc_shmem_atomic_add<T>(dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_atomic_inc(T *dest, int pe) \
    { \
        roc_shmem_atomic_inc<T>(dest, pe); \
    }

#define WAIT_DEF_GEN(T, TNAME) \
    __host__ void \
    roc_shmem_##TNAME##_wait_until(T *ptr, roc_shmem_cmps cmp, T val) \
    { \
        roc_shmem_wait_until<T>(ptr, cmp, val); \
    } \
    __host__ int \
    roc_shmem_##TNAME##_test(T *ptr, roc_shmem_cmps cmp, T val) \
    { \
        return roc_shmem_test<T>(ptr, cmp, val); \
    }

/******************************************************************************
 ************************* Macro Invocation Per Type **************************
 *****************************************************************************/

INT_REDUCTION_GEN(int)
INT_REDUCTION_GEN(short)
INT_REDUCTION_GEN(long)
INT_REDUCTION_GEN(long long)
FLOAT_REDUCTION_GEN(float)
FLOAT_REDUCTION_GEN(double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
//FLOAT_REDUCTION_GEN(long double)

/* Supported RMA types */
RMA_GEN(float) RMA_GEN(double) RMA_GEN(char) // RMA_GEN(long double)
RMA_GEN(signed char) RMA_GEN(short) RMA_GEN(int) RMA_GEN(long)
RMA_GEN(long long) RMA_GEN(unsigned char) RMA_GEN(unsigned short)
RMA_GEN(unsigned int) RMA_GEN(unsigned long) RMA_GEN(unsigned long long)

/* Supported AMO types for now (Only 64-bits)  */
AMO_GEN(int64_t)
AMO_GEN(uint64_t)
//AMO_GEN(long long)
//AMO_GEN(unsigned long long)
//AMO_GEN(size_t)
//AMO_GEN(ptrdiff_t)

/* Supported synchronization types */
WAIT_GEN(float) WAIT_GEN(double) WAIT_GEN(char) //WAIT_GEN(long double)
WAIT_GEN(signed char) WAIT_GEN(short) WAIT_GEN(int) WAIT_GEN(long)
WAIT_GEN(long long) WAIT_GEN(unsigned char) WAIT_GEN(unsigned short)
WAIT_GEN(unsigned int) WAIT_GEN(unsigned long) WAIT_GEN(unsigned long long)

INT_REDUCTION_DEF_GEN(int, int)
INT_REDUCTION_DEF_GEN(short, short)
INT_REDUCTION_DEF_GEN(long, long)
INT_REDUCTION_DEF_GEN(long long, longlong)
FLOAT_REDUCTION_DEF_GEN(float, float)
FLOAT_REDUCTION_DEF_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
//FLOAT_REDUCTION_DEF_GEN(long double, longdouble)

RMA_DEF_GEN(float, float)
RMA_DEF_GEN(double, double)
RMA_DEF_GEN(char, char)
//RMA_DEF_GEN(long double, longdouble)
RMA_DEF_GEN(signed char, schar)
RMA_DEF_GEN(short, short)
RMA_DEF_GEN(int, int)
RMA_DEF_GEN(long, long)
RMA_DEF_GEN(long long, longlong)
RMA_DEF_GEN(unsigned char, uchar)
RMA_DEF_GEN(unsigned short, ushort)
RMA_DEF_GEN(unsigned int, uint)
RMA_DEF_GEN(unsigned long, ulong)
RMA_DEF_GEN(unsigned long long, ulonglong)

AMO_DEF_GEN(int64_t, int64)
AMO_DEF_GEN(uint64_t, uint64)
//AMO_DEF_GEN(long long, longlong)
//AMO_DEF_GEN(unsigned long long, ulonglong)
//AMO_DEF_GEN(size_t, size)
//AMO_DEF_GEN(ptrdiff_t, ptrdiff)

WAIT_DEF_GEN(float, float)
WAIT_DEF_GEN(double, double)
WAIT_DEF_GEN(char, char)
//WAIT_DEF_GEN(long double, longdouble)
WAIT_DEF_GEN(signed char, schar)
WAIT_DEF_GEN(short, short)
WAIT_DEF_GEN(int, int)
WAIT_DEF_GEN(long, long)
WAIT_DEF_GEN(long long, longlong)
WAIT_DEF_GEN(unsigned char, uchar)
WAIT_DEF_GEN(unsigned short, ushort)
WAIT_DEF_GEN(unsigned int, uint)
WAIT_DEF_GEN(unsigned long, ulong)
WAIT_DEF_GEN(unsigned long long, ulonglong)
