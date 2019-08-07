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

#ifndef RO_NET_H
#define RO_NET_H

#define  __HIP_PLATFORM_HCC__
#include "hip/hip_runtime.h"

/**
 * @file ro_net.hpp
 * @brief Public header for RO_NET device and host libraries.
 *
 * This file contains all the callable functions and data structures for both
 * the device- and host-side runtimes.  The comments on these functions are
 * rather sparse, but the semanitcs are the same as those implemented in
 * OpenSHMEM unless otherwise documented.  Please see the OpenSHMEM 1.4
 * standards documentation for more details:
 *
 * http://openshmem.org/site/sites/default/site_files/OpenSHMEM-1.4.pdf
 */

/**
 * @brief Status codes for user-facing RO_NET calls.
 */
enum ro_net_status_t {
    RO_NET_UNKNOWN_ERROR,
    RO_NET_INVALID_ARGUMENTS,
    RO_NET_OOM_ERROR,
    RO_NET_SUCCESS,
};

/**
 * @brief Types defined for ro_net_wait() operations.
 */
enum ro_net_cmps {
    RO_NET_CMP_EQ,
    RO_NET_CMP_NE,
};

/**
 * @brief CPU side handle required for all CPU RO_NET calls.
 */
typedef uint64_t  ro_net_handle_t;

/**
 * @brief GPU side handle. Each work-group will have it own copy.
 */
typedef uint64_t  ro_net_wg_handle_t;

/**
 * @brief GPU side OpenSHMEM context created from each work-groups'
 * ro_net_wg_handle_t
 */
typedef uint64_t* ro_net_ctx_t;


/* Host-side interface */

/**
 * @brief Initialize the RO_NET runtime and underlying transport layer.
 *
 * @param[out] ro_net_gpu_handle CPU side handle.
 *
 * @return Status of the initialization.
 *
 */
ro_net_status_t ro_net_pre_init(ro_net_handle_t **ro_net_gpu_handle);

/**
 * @brief Allocate GPU/CPU queues and optionally spawn progress threads.
 *
 * @param[out] ro_net_gpu_handle CPU side handle.
 * @param[in] num_threads        Number of CPU helper threads per CPU
 *                               OpenSHMEM PE.
 * @param[in] num_queues         Force RO_NET to use this many queues.  By
 *                               default RO_NET will allocate worst-case for
 *                               the hardware assuming the minimum WG size,
 *                               but this option can be used to reduce queue
 *                               memory if the programmer knows better.
 * @return Status of the initialization.
 *
 */
ro_net_status_t ro_net_init(ro_net_handle_t **ro_net_gpu_handle,
                            int num_threads, int num_queues = 0);

/**
 * @brief User visible progress function.  This function should be used only if
 * the number of helper threads is 0.  Each PE must then call into this
 * function after launching a RO_NET enabled kernel in order to progress
 * messages.  The function will return when all GPU threads have called
 * ro_net_finalize().
 *
 * @param[in] ro_net_ gpu_handle CPU side handle.
 * @param[in] num_wgs Number of work-groups that must call finalize before
 *                    ro_net_forward() will return.
 *
 * @return Status of operation.
 *
 */
ro_net_status_t ro_net_forward(ro_net_handle_t * ro_net_gpu_handle,
                               int num_wgs);

/**
 * @brief Function that dumps internal stats to stdout.
 *
 * @param[in] ro_net_gpu_handle CPU side handle.
 *
 * @return Status of operation.
 *
 */
ro_net_status_t ro_net_dump_stats(ro_net_handle_t * ro_net_gpu_handle);

/**
 * @brief Reset all internal stats.
 *
 * @param[in] ro_net_gpu_handle CPU side handle.
 *
 * @return Status of operation.
 *
 */
ro_net_status_t ro_net_reset_stats(ro_net_handle_t * ro_net_gpu_handle);

/**
 * @brief Finalize the RO_NET runtime.
 *
 * @param[in] ro_net_gpu_handle CPU side handle.
 *
 * @return Status of finalization.
 *
 */
ro_net_status_t ro_net_finalize(ro_net_handle_t * ro_net_gpu_handle);


/**
 * @brief Allocate memory of \p size bytes from the symmetric heap.  This is a
 * collective operation and must be called by all PEs.
 *
 * @param[in] size Memory allocation size in bytes.
 *
 * @return A pointer to the allocated memory on the symmetric heap.
 *
 * @todo Return error code instead of ptr.
 *
 */
void* ro_net_malloc(size_t size);


/**
 * @brief Free a memory allocation from the symmetric heap.  This is a
 * collective operation and must be called by all PEs.
 *
 * @param[in] ptr Pointer to previously allocated memory on the symmetric heap.
 *
 * @return Status of the operation.
 *
 */
ro_net_status_t ro_net_free(void* ptr);


/**
 * @brief Query for the number of PEs.
 *
 * @return Number of PEs.
 *
 */
int ro_net_n_pes();


/**
 * @brief Query the PE ID of the caller.
 *
 * @return PE ID of the caller.
 *
 */
int ro_net_my_pe();

/**
 * Device-side interface.
 *
 * @todo Return error codes.
 */

/**
 * @brief Creates an OpenSHMEM context. By design, the context is private
 * to the calling work-group.
 *
 * Only one thread per work-group is allowed to call into this function.
 *
 * @param[in] options Options for context creation.  Ignored in current design.
 * @param[out] ctx    Context handle.
 * @param[in] handle  Per-work-group ro_net handle.
 *
 * @return void.
 *
 */
__device__ void ro_net_ctx_create(long options, ro_net_ctx_t *ctx,
                                  ro_net_wg_handle_t * handle);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * Only one thread per work-group is allowed to call into this function.
 *
 * @param[in] The context to destroy.
 *
 * @return void.
 *
 */
__device__ void ro_net_ctx_destroy(ro_net_ctx_t ctx);

/**
 * @brief Writes contiguous data of \p size bytes from \p src on the calling
 * PE to \p dst at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p src).  The caller must
 * call into ro_net_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] ctx  Context with which to perform this operation.
 * @param[in] dst  Destination address. Must be an address on the symmetric
 *                 heap.
 * @param[in] src  Source address. Must be an address on the symmetric heap.
 * @param[in] size Size of the transfer in bytes.
 * @param[in] pe   PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void ro_net_putmem(ro_net_ctx_t ctx, void *dst, void *src,
                              int size, int pe);

/**
 * @brief Reads contiguous data of \p size bytes from \p src on \p pe 
 * to \p dst on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dst).
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] ctx  Context with which to perform this operation.
 * @param[in] dst  Destination Address. Must be an address on the symmetric
 *                 heap.
 * @param[in] src  Source address. Must be an address on the symmetric heap.
 * @param[in] size Size of the transfer in bytes.
 * @param[in] pe   PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void ro_net_getmem(ro_net_ctx_t ctx, void *dst, void *src,
                                int size, int pe);

/**
 * @brief Writes contiguous data of \p size bytes from \p src on the calling
 * PE to \p dst on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * ro_net_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] ctx  Context with which to perform this operation.
 * @param[in] dst  Destination address. Must be an address on the symmetric
                   heap.
 * @param[in] src  Source address. Must be an address on the symmetric heap.
 * @param[in] size Size of the transfer in bytes.
 * @param[in] pe   PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void ro_net_putmem_nbi(ro_net_ctx_t ctx, void *dst, void *src,
                                  int size, int pe);

/**
 * @brief Reads contiguous data of \p size bytes from \p src on \p pe
 * to \p dst on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted.  The caller must call
 * ro_net_quiet() on the same context if completion notification is required.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] ctx  Context with which to perform this operation.
 * @param[in] dst  Destination address. Must be an address on the symmetric
 *                 heap.
 * @param[in] src  Source address. Must be an address on the symmetric heap.
 * @param[in] size Size of the transfer in bytes.
 * @param[in] pe   PE of the remote process.
 *
 * @return void.
 *
 */
__device__ void ro_net_getmem_nbi(ro_net_ctx_t ctx, void *dst, void *src,
                                  int size, int pe);

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 *
 */
__device__ void ro_net_fence(ro_net_ctx_t ctx);

/**
 * @brief Completes all previous operations posted to this context.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] ctx Context with which to perform this operation.
 *
 * @return void.
 *
 */
__device__ void ro_net_quiet(ro_net_ctx_t ctx);

/**
 * @brief Initializes the GPU-side runtime and creates a GPU handle for each
 * work-group.
 *
 * Only one thread per work-group is allowed to call into this function.
 *
 * @param[in] handle     CPU side handle.
 * @param[out] wg_handle GPU side handle.
 *
 * @return void.
 *
 */
__device__ void ro_net_init(ro_net_handle_t * handle,
                            ro_net_wg_handle_t **wg_handle);

/**
 * @brief Finalizes the GPU-side runtime.
 *
 * Only one thread per work-group is allowed to call into this function.
 *
 * @param[in] handle    GPU-side handle.
 * @param[in] wg_handle GPU-side handle.
 *
 * @return void.
 *
 */
__device__ void ro_net_finalize(ro_net_handle_t * handle,
                                  ro_net_wg_handle_t * wg_handle);

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
__device__ int ro_net_n_pes(ro_net_wg_handle_t *wg_handle);


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
__device__ int ro_net_my_pe(ro_net_wg_handle_t *wg_handle);

/**
 * @brief Perform an allreduce with SUM operation between PEs in the active
 * set.  The caller is blocked until the reduction completes.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] dst          Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] src          Source address. Must be an address on the symmetric
                           heap.
 * @param[in] size         Size of the buffer to participate in the reduction.
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pWrk         Temporary work buffer provided to RO_NET.
 * @param[in] pSync        Temporary work buffer provided to RO_NET.
 * @param[in] handle       GPU side handle.
 *
 * @return void
 *
 */
__device__ void ro_net_float_sum_to_all(float *dst, float *src, int size,
                                        int PE_start, int logPE_stride,
                                        int PE_size, float *pWrk,
                                        long *pSync,
                                        ro_net_wg_handle_t * handle);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 *
 */
__device__ void ro_net_barrier_all(ro_net_wg_handle_t * handle);

/**
 * @brief Block the caller until the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity.  However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * RO_NET function.
 *
 * @param[in] ctx Context with which to perform this operation.
 * @param[in] ptr Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return void
 *
 */
__device__ void ro_net_wait_until(ro_net_ctx_t ctx, void *ptr,
                                  ro_net_cmps cmp, int val);

/**
 * @brief Query the current time. Similar to gettimeofday() on the CPU. To use
 * this function, RO_NET must be configured with profiling support
 * (--enable-profile).
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] wg_handle GPU-side handle.
 *
 * @return Time in micro-seconds.
 *
 */
__device__ uint64_t ro_net_timer(ro_net_wg_handle_t * wg_handle);

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
__device__ void profiler_enable(ro_net_wg_handle_t * wg_handle);

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
__device__ void profiler_skip(ro_net_wg_handle_t * wg_handle, bool status);


/**
 * @brief Make all uncacheable GPU data visible to other agents in the sytem.
 * This only works for data that was explicitly allocated unacheable on the
 * GPU!
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] GPU-side handle.
 *
 * @return void
 *
 */
__device__ void ro_net_threadfence_system(ro_net_wg_handle_t * wg_handle);

#endif
