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
 * @brief Status codes for user-facing RO_NET calls
 */
enum ro_net_status_t {
    RO_NET_UNKNOWN_ERROR,
    RO_NET_INVALID_ARGUMENTS,
    RO_NET_OOM_ERROR,
    RO_NET_SUCCESS,
};

/**
 * @brief Type of ro_net_wait() operation
 */
enum ro_net_cmps {
    RO_NET_CMP_EQ,
    RO_NET_CMP_NE,
};

/**
 * @brief CPU side handle
 */
typedef uint64_t  ro_net_handle_t;

/**
 * @brief GPU side handle. Each work-group will have it own copy
 */
typedef uint64_t  ro_net_wg_handle_t;

/**
 * @brief GPU side OpenSHMEM context
 */
typedef uint64_t* ro_net_ctx_t;


/* Host-side interface */

/**
 * @brief initialize the RO_NET runtime and underlying transport layer.
 *
 * @param[out] CPU side handle.
 *
 * @retval : status of the initialization.
 *
 */
ro_net_status_t ro_net_pre_init(ro_net_handle_t **ro_net_gpu_handle);

/**
 * @brief allocate GPU/CPU queues and optionally spawn progress threads
 *
 * @param[out] CPU side handle.
 *
 * @param[in] number of Workgroups in the persisitent kernel
 *
 * @param[in] number of CPU helper threads per CPU OpenSHMEM PE
 *
 * @param[in] number of producer/consumer queues for the runtime to use
 *
 * @retval : status of the initialization.
 *
 */
ro_net_status_t ro_net_init(ro_net_handle_t **ro_net_gpu_handle,
                                int num_wgs,  int num_threads, int num_queues);


/**
 * @brief This function should be used only if the number of helper threads is 0.
 * the PE calls this function after launching the kernel.
 *
 * @param[in] CPU side handle.
 *
 * @retval : status of operation.
 *
 */
ro_net_status_t ro_net_forward(ro_net_handle_t * ro_net_gpu_handle);

ro_net_status_t ro_net_dump_stats(ro_net_handle_t * ro_net_gpu_handle);
ro_net_status_t ro_net_reset_stats(ro_net_handle_t * ro_net_gpu_handle);

/**
 * @brief finalize the RO_NET runtime.
 *
 * @param[in] CPU side handle.
 *
 * @retval : status of the initialization.
 *
 */
ro_net_status_t ro_net_finalize(ro_net_handle_t * ro_net_gpu_handle);


/**
 * @brief Allocate a memory of size @size from the  symmetric heap.
 *
 * @param[in] size of the requestion allocation in bytes.
 *
 * @retval : a pointer to the allocated memory on the symmetric heap.
 *
 */
void* ro_net_malloc(size_t size);


/**
 * @brief free a memory spaces and return it to the symmetric heap.
 *
 * @param[in] pointer to the memory to free. @ptr should be part
 * of the symmetric heap
 *
 * @retval : status of the operation.
 *
 */
ro_net_status_t ro_net_free(void* ptr);


/**
 * @brief returns the number of PEs.
 *
 * @retval : number of PEs.
 *
 */
int ro_net_n_pes();


/**
 * @brief returns the ID (rank) o fthe calling PE.
 *
 * @retval : returns the rank.
 *
 */
int ro_net_my_pe();

/* Device-side interface */

/**
 * @brief creates an OpenSHMEM context. By design, the context is private
 * to he calling workgroup.
 *
 * @param[in] options of the context, must be 0
 *
 * @param[out] the context handle
 *
 * @param[in] GPU side ro_net handle
 *
 * @retval : void.
 *
 */
__device__ void ro_net_ctx_create(long options, ro_net_ctx_t *ctx,
                                    ro_net_wg_handle_t * handle);

/**
 * @brief destroys an OpenSHMEM context.
 *
 * @param[in] teh context to destroy
 *
 * @retval : void.
 *
 */
__device__ void ro_net_ctx_destroy(ro_net_ctx_t ctx);

/**
 * @brief writes a contigous data of size @size bytes from @src
 * on the calling PE to the @dst on PE @pe. The caling WG will block until the
 * operation completes locally (it is safe to reuse @src buffer).
 *
 * @param[in] context to perform this operation
 *
 * @param[in] destination buffer. Must be an address on the symmetric heap
 *
 * @param[in] source address. Must be an address on the symmetric heap
 *
 * @param[in] size of the transfer in bytes
 *
 * @param[in] PE of the remote process
 *
 * @retval : void.
 *
 */
__device__ void ro_net_putmem(ro_net_ctx_t ctx, void *dst, void *src,
                                int size, int pe);

/**
 * @brief reads a contigous data of size @size bytes from @src
 * on the remote PE to the @dst on the calling PE. The caling WG will block
 * until the operation completes (it is safe to access @dst buffer).
 *
 * @param[in] context to perform this operation
 *
 * @param[in] destination buffer. Must be an address on the symmetric heap
 *
 * @param[in] source address. Must be an address on the symmetric heap
 *
 * @param[in] size of the transfer in bytes
 *
 * @param[in] PE of the remote process
 *
 * @retval : void.
 *
 */
__device__ void ro_net_getmem(ro_net_ctx_t ctx, void *dst, void *src,
                                int size, int pe);

/**
 * @brief writes a contigous data of size @size bytes from @src
 * on the calling PE to the @dst on PE @pe. The operation is not blocking.
 * The calling WG will return as soon as the request is posted.
 * The operation will complete by calling ro_net_quiet on teh same ctx
 *
 * @param[in] context to perform this operation
 *
 * @param[in] destination buffer. Must be an address on the symmetric heap
 *
 * @param[in] source address. Must be an address on the symmetric heap
 *
 * @param[in] size of the transfer in bytes
 *
 * @param[in] PE of the remote process
 *
 * @retval : void.
 *
 */
__device__ void ro_net_putmem_nbi(ro_net_ctx_t ctx, void *dst, void *src,
                                    int size, int pe);

/**
 * @brief reads a contigous data of size @size bytes from @src
 * on the remote PE to the @dst on the calling PE. The operation is not
 * blocking. The calling WG will return as soon as the request is posted.
 * The operation will complete by calling ro_net_quiet on teh same ctx
 *
 * @param[in] context to perform this operation
 *
 * @param[in] destination buffer. Must be an address on the symmetric heap
 *
 * @param[in] source address. Must be an address on the symmetric heap
 *
 * @param[in] size of the transfer in bytes
 *
 * @param[in] PE of the remote process
 *
 * @retval : void.
 *
 */
__device__ void ro_net_getmem_nbi(ro_net_ctx_t ctx, void *dst, void *src,
                                    int size, int pe);

/**
 * @brief guarantees the order with OpenSHMEM semantics.
 *
 * @param[in] context to perform this operation
 *
 * @retval : void.
 *
 */
__device__ void ro_net_fence(ro_net_ctx_t ctx);

/**
 * @brief completes all previous operations posted to this context @ctx
 *
 * @param[in] context to perform this operation
 *
 * @retval : void.
 *
 */
__device__ void ro_net_quiet(ro_net_ctx_t ctx);

/**
 * @brief initializes the GPU side of the runtime and creates a GPU handle for
 * each WG
 *
 * @param[in] CPU side handle
 *
 * @param[out] GPU side handle
 *
 * @retval : void.
 *
 */
__device__ void ro_net_init(ro_net_handle_t * handle,
                              ro_net_wg_handle_t **wg_handle);

/**
 * @brief finalizes the GPU side of the runtime
 *
 * @param[in] GPU side handle
 *
 * @param[in] context handle
 *
 * @retval : void.
 *
 */
__device__ void ro_net_finalize(ro_net_handle_t * handle,
                                  ro_net_wg_handle_t * wg_handle);

/**
 * @brief return number of PEs
 *
 * @param[in] GPU side handle
 *
 * @retval : number of PEs
 *
 */
__device__ int ro_net_n_pes(ro_net_wg_handle_t *wg_handle);


/**
 * @brief return rank of the PE
 *
 * @param[in] GPU side handle
 *
 * @retval : rank of PE
 *
 */
__device__ int ro_net_my_pe(ro_net_wg_handle_t *wg_handle);

/**
 * @brief perform an allreduce with SUM operation between PEs in the active
 * set.
 *
 * @param[in] GPU side handle
 *
 * @retval : void
 *
 */
__device__ void ro_net_float_sum_to_all(float *dst, float *src, int size,
                                          int PE_start, int logPE_stride,
                                          int PE_size, float *pWrk,
                                          long *pSync,
                                          ro_net_wg_handle_t * handle);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * Only the calling WG is blocked.
 *
 * @param[in] GPU side handle
 *
 * @retval : void
 *
 */
__device__ void ro_net_barrier_all(ro_net_wg_handle_t * handle);


/**
 * @brief block the calling WG (busy polling) until the condition
 *  (*@ptr @ro_net_cmps @val) is true.
 *
 * @param[in] context
 *
 * @param[in] pointer to amemory on the symmetric heap
 *
 * @param[in] value to compare to
 *
 * @retval : void
 *
 */
__device__ void ro_net_wait_until(ro_net_ctx_t ctx, void *ptr,
                                    ro_net_cmps, int val);

/**
 * @brief return the current time. Similar to gettimeofday(). To use this function
 * RO_NET must be configured with profile support (--enable-profile)
 *
 * @param[in] GPU side handle
 *
 * @retval : time in micro-seconds
 *
 */
__device__ uint64_t ro_net_timer(ro_net_wg_handle_t * wi_handle);

/**
 * @brief enable the timers and profilers (PVARs) at runtime.
 *
 * @param[in] GPU side handle
 *
 * @retval : void
 *
 */
__device__ void profiler_enable(ro_net_wg_handle_t * wi_handle);

/**
 * @brief set SKIP to true/flase (@status). This is useful for warmup iterations.
 *
 * @param[in] GPU side handle
 *
 * @param[in] status of skip
 *
 * @retval : void
 *
 */
__device__ void profiler_skip(ro_net_wg_handle_t * wg_handle, bool status);


/**
 * @brief Make all uncacheable GPU data visible to other agents in the sytem.
 *
 * @param[in] GPU side handle
 *
 * @retval : void
 *
 */
__device__ void ro_net_threadfence_system(ro_net_wg_handle_t * wg_handle);

#endif
