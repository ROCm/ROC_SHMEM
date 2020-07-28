/******************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROC_SHMEM_H
#define ROC_SHMEM_H

#define  __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>

/**
 * @file roc_shmem.hpp
 * @brief Public header for ROC_SHMEM device and host libraries.
 *
 * This file contains all the callable functions and data structures for both
 * the device-side runtime and host-side runtime.
 *
 * The comments on these functions are sparse, but the semantics are the same
 * as those implemented in OpenSHMEM unless otherwise documented. Please see
 * the OpenSHMEM 1.4 standards documentation for more details:
 *
 * http://openshmem.org/site/sites/default/site_files/OpenSHMEM-1.4.pdf
 */

/**
 * @brief Status codes for user-facing ROC_SHMEM calls.
 */
enum roc_shmem_status_t {
    ROC_SHMEM_UNKNOWN_ERROR,
    ROC_SHMEM_INVALID_ARGUMENTS,
    ROC_SHMEM_OOM_ERROR,
    ROC_SHMEM_SUCCESS,
};

/**
 * @brief Types defined for roc_shmem_wait() operations.
 */
enum roc_shmem_cmps {
    ROC_SHMEM_CMP_EQ,
    ROC_SHMEM_CMP_NE,
    ROC_SHMEM_CMP_GT,
    ROC_SHMEM_CMP_GE,
    ROC_SHMEM_CMP_LT,
    ROC_SHMEM_CMP_LE,
};

enum ROC_SHMEM_OP {
    ROC_SHMEM_SUM,
    ROC_SHMEM_MAX,
    ROC_SHMEM_MIN,
    ROC_SHMEM_PROD,
    ROC_SHMEM_AND,
    ROC_SHMEM_OR,
    ROC_SHMEM_XOR
};

enum roc_shmem_thread_ops {
    SHMEM_THREAD_SINGLE,
    SHMEM_THREAD_FUNNELED,
    SHMEM_THREAD_WG_FUNNELED,
    SHMEM_THREAD_SERIALIZED,
    SHMEM_THREAD_MULTIPLE
};

constexpr size_t SHMEM_REDUCE_MIN_WRKDATA_SIZE = 1024;
constexpr size_t SHMEM_BARRIER_SYNC_SIZE = 256;
constexpr size_t SHMEM_REDUCE_SYNC_SIZE = 256;
constexpr size_t SHMEM_BCAST_SYNC_SIZE = 256;
constexpr size_t SHMEM_SYNC_VALUE = 0;

const int SHMEM_CTX_SERIALIZED = 1;
const int SHMEM_CTX_PRIVATE = 2;
const int SHMEM_CTX_NOSTORE = 4;
const int SHMEM_CTX_WG_PRIVATE = 8;

/**
 * @brief GPU side OpenSHMEM context created from each work-groups'
 * roc_shmem_wg_handle_t
 */
typedef uint64_t* roc_shmem_ctx_t;

/**
 * Shmem default context.
 */
extern __constant__ roc_shmem_ctx_t SHMEM_CTX_DEFAULT;



/******************************************************************************
 **************************** HOST INTERFACE **********************************
 *****************************************************************************/

/**
 * @brief Initialize the ROC_SHMEM runtime and underlying transport layer.
 *        Allocate GPU/CPU queues and optionally spawn progress threads.
 *
 * @param[in] num_wgs (Optional) Communicate to ROC_SHMEM that launched kernels
 *                     will never exceed num_wgs number of work-groups in
 *                     a single grid launch. ROC_SHMEM can use this to reduce
 *                     memory utilization in some cases. If no argument is
 *                     provided, ROC_SHMEM will size resources based on worst-
 *                     case analysis of the target hardware.
 *
 * @return Status of the initialization.
 *
 */
roc_shmem_status_t roc_shmem_init(unsigned num_wgs = 0);

/**
 * @brief Function that dumps internal stats to stdout.
 *
 * @return Status of operation.
 *
 */
roc_shmem_status_t roc_shmem_dump_stats();

/**
 * @brief Reset all internal stats.
 *
 * @return Status of operation.
 *
 */
roc_shmem_status_t roc_shmem_reset_stats();

/**
 * @brief Finalize the ROC_SHMEM runtime.
 *
 * @return Status of finalization.
 *
 */
roc_shmem_status_t roc_shmem_finalize();

/**
 * @brief Allocate memory of \p size bytes from the symmetric heap. This is a
 * collective operation and must be called by all PEs.
 *
 * @param[in] size Memory allocation size in bytes.
 *
 * @return A pointer to the allocated memory on the symmetric heap.
 *
 * @todo Return error code instead of ptr.
 *
 */
void* roc_shmem_malloc(size_t size);

/**
 * @brief Free a memory allocation from the symmetric heap. This is a
 * collective operation and must be called by all PEs.
 *
 * @param[in] ptr Pointer to previously allocated memory on the symmetric heap.
 *
 * @return Status of the operation.
 *
 */
roc_shmem_status_t roc_shmem_free(void* ptr);

/**
 * @brief Query for the number of PEs.
 *
 * @return Number of PEs.
 *
 */
int roc_shmem_n_pes();

/**
 * @brief Query the PE ID of the caller.
 *
 * @return PE ID of the caller.
 *
 */
int roc_shmem_my_pe();

/**
 * @brief ROC_SHMEM makes extensive use of dynamic shared memory inside of
 * its runtime. This function returns the amount of dynamic shared memory
 * required by a ROC_SHMEM-enabled kernel. The user must add this allocation
 * to any other dynamic shared memory used by the kernel and provide it as a
 * launch parameter to the kernel.
 *
 * @param[in] shared_bytes Amount of dynamic shared memory ROC_SHMEM kernels
 * require in bytes.
 *
 * @return Status of the operation.
 *
 */
roc_shmem_status_t roc_shmem_dynamic_shared(size_t *shared_bytes);



/******************************************************************************
 **************************** DEVICE INTERFACE ********************************
 *****************************************************************************/

/**
 * @brief Initializes device-side ROC_SHMEM resources. Must be called before
 * any threads in this work-group invoke other ROC_SHMEM functions.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_wg_init();

/**
 * @brief Finalizes device-side ROC_SHMEM resources. Must be called before
 * work-group completion if the work-group also called roc_shmem_wg_init().
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_wg_finalize();

/**
 * @brief Initializes device-side ROC_SHMEM resources. Must be called before
 * any threads in this work-group invoke other ROC_SHMEM functions. This is
 * a variant of roc_shmem_wg_init that allows the caller to request a
 * threading mode.
 *
 * @param[in] requested Requested thread mode from roc_shmem_thread_ops.
 * @param[out] provided Thread mode selected by the runtime. May not be equal
 *                      to requested thread mode.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_wg_init_thread(int requested, int *provided);

/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[out] provided Thread mode the runtime is operating in.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_query_thread(int *provided);

/**
 * @brief Creates an OpenSHMEM context. By design, the context is private
 * to the calling work-group.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @param[in] options Options for context creation. Ignored in current design.
 * @param[out] ctx    Context handle.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_wg_ctx_create(long options, roc_shmem_ctx_t *ctx);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @param[in] The context to destroy.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_wg_ctx_destroy(roc_shmem_ctx_t ctx);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into roc_shmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_putmem(roc_shmem_ctx_t ctx, void *dest,
                                 const void *source, size_t nelems, int pe);

__device__ void roc_shmem_putmem(void *dest,
                                 const void *source, size_t nelems, int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into roc_shmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe);

template <typename T>
__device__ void roc_shmem_put(T *dest, const T *source, size_t nelems, int pe);

/**
 * @brief Writes a single value to \p dest at \p pe PE to \p dst at \p pe.
 * The caller must call into roc_shmem_quiet() if remote completion is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] value  Value to write to dest at \p pe.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe);

template <typename T>
__device__ void roc_shmem_p(T *dest, T value, int pe);

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_getmem(roc_shmem_ctx_t ctx, void *dest,
                                 const void *source, size_t nelems, int pe);

__device__ void roc_shmem_getmem(void *dest,
                                 const void *source, size_t nelems, int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe);

template <typename T>
__device__ void roc_shmem_get(T *dest, const T *source, size_t nelems, int pe);

/**
 * @brief reads and returns single value from \p source at \p pe.
 * The calling work-group/thread will block until the operation completes.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] source sourcen address. Must be an address on the symmetric
 *                   heap.
 * @param[in] pe     PE of the remote process.
 *
 * @return the value read from remote \p source at \p pe.
 *
 */
template <typename T>
__device__ T roc_shmem_g(roc_shmem_ctx_t ctx, T *source, int pe);

template <typename T>
__device__ T roc_shmem_g(T *source, int pe);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_putmem_nbi(roc_shmem_ctx_t ctx, void *dest,
                                     const void *source, size_t nelems,
                                     int pe);

__device__ void roc_shmem_putmem_nbi(void *dest,
                                     const void *source, size_t nelems,
                                     int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *src,
                                  size_t nelems, int pe);

template <typename T>
__device__ void roc_shmem_put_nbi(T *dest, const T *src,
                                  size_t nelems, int pe);

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_getmem_nbi(roc_shmem_ctx_t ctx, void *dest,
                                     const void *source, size_t nelems,
                                     int pe);

__device__ void roc_shmem_getmem_nbi(void *dest,
                                     const void *source, size_t nelems,
                                     int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest,
                                  const T *source, size_t nelems, int pe);

template <typename T>
__device__ void roc_shmem_get_nbi(T *dest,
                                  const T *source, size_t nelems, int pe);

/**
 * @brief Atomically add the value \p val to \p dest on \p pe. The operation
 * returns the older value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest before the \p val was added.
 *
 */
template <typename T>
__device__ T roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest,
                                        T val, int pe);

template <typename T>
__device__ T roc_shmem_atomic_fetch_add(T *dest, T val, int pe);

/**
 * @brief Atomically compares if the value in \p dest with \p cond is equal
 * then put \p val in \p dest. The operation returns the older value of \p dest
 * to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] cond    The value to be compare with.
 * @param[in] val     The value to be atomically swapped.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest.
 *
 */
template <typename T>
__device__ T roc_shmem_atomic_fetch_cswap(roc_shmem_ctx_t ctx, T *dest,
                                          T cond, T val, int pe);

template <typename T>
__device__ T roc_shmem_atomic_fetch_cswap(T *dest, T cond, T val, int pe);

/**
 * @brief Atomically add 1 to \p dest on \p pe. The operation
 * returns the older value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The old value of \p dest before it was incremented by 1.
 *
 */
template <typename T>
__device__ T roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__device__ T roc_shmem_atomic_fetch_inc(T *dest, int pe);

/**
 * @brief Atomically return the value of \p dest to the calling PE.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return            The value of \p dest.
 *
 */
template <typename T>
__device__ T roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__device__ T roc_shmem_atomic_fetch(T *dest, int pe);

/**
 * @brief Atomically add the value \p val to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] val     The value to be atomically added.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 *
 */
template <typename T>
__device__ void roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest,
                                     T val, int pe);

template <typename T>
__device__ void roc_shmem_atomic_add(T *dest, T val, int pe);

/**
 * @brief Atomically compares if the value in \p dest with \p cond is equal
 * then put \p val in \p dest.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] cond    The value to be compare with.
 * @param[in] val     The value to be atomically swapped.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 *
 */
template <typename T>
__device__ void  roc_shmem_atomic_cswap(roc_shmem_ctx_t ctx, T *dest,
                                        T cond, T val, int pe);

template <typename T>
__device__ void  roc_shmem_atomic_cswap(T *dest, T cond, T val, int pe);

/**
 * @brief Atomically add 1 to \p dest on \p pe.
 *
 * The operation is blocking.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] pe      PE of the remote process.
 *
 * @return void
 *
 */
template <typename T>
__device__ void roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__device__ void roc_shmem_atomic_inc(T *dest, int pe);

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_fence(roc_shmem_ctx_t ctx);

__device__ void roc_shmem_fence();

/**
 * @brief Completes all previous operations posted to this context.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 *
 */
__device__ void roc_shmem_quiet(roc_shmem_ctx_t ctx);

__device__ void roc_shmem_quiet();

/**
 * @brief Query the total number of PEs.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] wg_handle GPU side handle.
 *
 * @return Total number of PEs.
 *
 */
__device__ int roc_shmem_n_pes(roc_shmem_ctx_t ctx);

/**
 * @brief Query the PE ID of the caller.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] wg_handle GPU side handle
 *
 * @return PE ID of the caller.
 *
 */
__device__ int roc_shmem_my_pe(roc_shmem_ctx_t ctx);

/**
 * @brief Perform an allreduce between PEs in the active set. The caller
 * is blocked until the reduction completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nreduce      Size of the buffer to participate in the reduction.
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pWrk         Temporary work buffer provided to ROC_SHMEM. Must
 *                         be of size at least max(size/2 + 1,
                           SHMEM_REDUCE_MIN_WRKDATA_SIZE).
 * @param[in] pSync        Temporary sync buffer provided to ROC_SHMEM. Must
                           be of size at least SHMEM_REDUCE_SYNC_SIZE.
 * @param[in] handle       GPU side handle.
 *
 * @return void
 *
 */
template<typename T, ROC_SHMEM_OP Op>
__device__ void roc_shmem_wg_to_all(roc_shmem_ctx_t ctx, T *dest,
                                    const T *source, int nreduce, int PE_start,
                                    int logPE_stride, int PE_size, T *pWrk,
                                    long *pSync);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 *
 */
__device__ void roc_shmem_wg_barrier_all(roc_shmem_ctx_t ctx);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 *
 */
__device__ void roc_shmem_wg_sync_all(roc_shmem_ctx_t ctx);

/**
 * @brief Block the caller until the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 * @param[in] ptr Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return void
 *
 */
template <typename T>
__device__ void roc_shmem_wait_until(roc_shmem_ctx_t ctx, T *ptr,
                                     roc_shmem_cmps cmp, T val);

/**
 * @brief test if the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ctx Context with which to perform this operation.
 * @param[in] ptr Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return 1 if the evaluation is true else 0
 *
 */
template <typename T>
__device__ int roc_shmem_test(roc_shmem_ctx_t ctx, T *ptr,
                              roc_shmem_cmps cmp, T val);

/**
 * @brief Query the current time. Similar to gettimeofday() on the CPU. To use
 * this function, ROC_SHMEM must be configured with profiling support
 * (--enable-profile).
 *
 * Can be called per thread with no performance penalty.
 *
 * @return Time in micro-seconds.
 *
 */
__device__ uint64_t roc_shmem_timer();

/**
 * @brief Enable the timers and profilers at runtime.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] wg_handle GPU-side handle.
 *
 * @return void
 *
 */
__device__ void profiler_enable(roc_shmem_ctx_t ctx);

/**
 * @brief Set SKIP to \p status. This is useful for warmup iterations.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] wg_handle GPU-side handle.
 * @param[in] status    Status of skip.
 *
 * @return void
 *
 */
__device__ void profiler_skip(roc_shmem_ctx_t ctx, bool status);

/**
 * @brief Make all uncacheable GPU data visible to other agents in the sytem.
 *
 * This only works for data that was explicitly allocated uncacheable on the
 * GPU!
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] GPU-side handle.
 *
 * @return void
 *
 */
__device__ void roc_shmem_threadfence_system(roc_shmem_ctx_t ctx);

#endif
