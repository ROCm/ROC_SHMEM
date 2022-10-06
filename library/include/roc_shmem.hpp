/******************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCSHMEM_LIBRARY_INCLUDE_ROC_SHMEM_HPP
#define ROCSHMEM_LIBRARY_INCLUDE_ROC_SHMEM_HPP

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

namespace rocshmem {

enum ROC_SHMEM_OP {
    ROC_SHMEM_SUM,
    ROC_SHMEM_MAX,
    ROC_SHMEM_MIN,
    ROC_SHMEM_PROD,
    ROC_SHMEM_AND,
    ROC_SHMEM_OR,
    ROC_SHMEM_XOR
};

/**
 * @brief Status codes for user-facing ROC_SHMEM calls.
 */
enum class Status {
    ROC_SHMEM_UNKNOWN_ERROR,
    ROC_SHMEM_INVALID_ARGUMENTS,
    ROC_SHMEM_OOM_ERROR,
    ROC_SHMEM_TOO_MANY_TEAMS_ERROR,
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

enum roc_shmem_thread_ops {
    ROC_SHMEM_THREAD_SINGLE,
    ROC_SHMEM_THREAD_FUNNELED,
    ROC_SHMEM_THREAD_WG_FUNNELED,
    ROC_SHMEM_THREAD_SERIALIZED,
    ROC_SHMEM_THREAD_MULTIPLE
};

/**
 * @brief Bitwise flags to mask configuration parameters.
 */
enum roc_shmem_team_configs {
    ROC_SHMEM_TEAM_DEFAULT_CONFIGS,
    ROC_SHMEM_TEAM_NUM_CONTEXTS
};

typedef struct {
    int num_contexts;
} roc_shmem_team_config_t;

constexpr size_t ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE = 1024;
constexpr size_t ROC_SHMEM_ATA_MAX_WRKDATA_SIZE = (4 * 1024 * 1024);
constexpr size_t ROC_SHMEM_BARRIER_SYNC_SIZE  = 256;
constexpr size_t ROC_SHMEM_REDUCE_SYNC_SIZE   = 256;
// Internally calls sync function, which matches barrier implementation
constexpr size_t ROC_SHMEM_BCAST_SYNC_SIZE    = ROC_SHMEM_BARRIER_SYNC_SIZE;
constexpr size_t ROC_SHMEM_ALLTOALL_SYNC_SIZE = ROC_SHMEM_BARRIER_SYNC_SIZE + 1;
constexpr size_t ROC_SHMEM_FCOLLECT_SYNC_SIZE = ROC_SHMEM_ALLTOALL_SYNC_SIZE;
constexpr size_t ROC_SHMEM_SYNC_VALUE = 0;

const int ROC_SHMEM_CTX_ZERO = 0;
const int ROC_SHMEM_CTX_SERIALIZED = 1;
const int ROC_SHMEM_CTX_PRIVATE = 2;
const int ROC_SHMEM_CTX_NOSTORE = 4;
const int ROC_SHMEM_CTX_WG_PRIVATE = 8;

/**
 * @brief GPU side OpenSHMEM context created from each work-groups'
 * roc_shmem_wg_handle_t
 */
typedef struct {
    void* ctx_opaque;
    void* team_opaque;
} roc_shmem_ctx_t;

/**
 * Shmem default context.
 */
extern __constant__ roc_shmem_ctx_t ROC_SHMEM_CTX_DEFAULT;

typedef uint64_t* roc_shmem_team_t;
extern roc_shmem_team_t ROC_SHMEM_TEAM_WORLD;

const roc_shmem_team_t ROC_SHMEM_TEAM_INVALID = nullptr;

/******************************************************************************
 **************************** HOST INTERFACE **********************************
 *****************************************************************************/
__host__ Status
roc_shmem_init(unsigned num_wgs = 0);

/**
 * @brief Initialize the ROC_SHMEM runtime and underlying transport layer
 *        with an attempt to enable the requested thread support.
 *        Allocate GPU/CPU queues and optionally spawn progress threads.
 *
 * @param[in] requested Requested thread mode (from roc_shmem_thread_ops)
 *                      for host-facing functions.
 * @param[out] provided Thread mode selected by the runtime. May not be equal
 *                      to requested thread mode.
 * @param[in] num_wgs   (Optional) Communicate to ROC_SHMEM that launched
 *                      kernels will never exceed num_wgs number of work-groups
 *                      in a single grid launch. ROC_SHMEM can use this to
 *                      reduce memory utilization in some cases. If no argument
 *                      is provided, ROC_SHMEM will size resources based on
 *                      worst-case analysis of the target hardware.
 *
 * @return Status of the operation -- 0 upon success, non-zero otherwise
 */
__host__ int
roc_shmem_init_thread(int requested,
                      int *provided,
                      unsigned num_wgs = 0);

/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[out] provided Thread mode the runtime is operating in.
 *
 * @return void.
 */
__host__ void
roc_shmem_query_thread(int *provided);

/**
 * @brief Function that dumps internal stats to stdout.
 *
 * @return Status of operation.
 */
__host__ Status
roc_shmem_dump_stats();

/**
 * @brief Reset all internal stats.
 *
 * @return Status of operation.
 */
__host__ Status
roc_shmem_reset_stats();

/**
 * @brief Finalize the ROC_SHMEM runtime.
 *
 * @return Status of finalization.
 */
__host__ Status
roc_shmem_finalize();

/**
 * @brief Allocate memory of \p size bytes from the symmetric heap.
 * This is a collective operation and must be called by all PEs.
 *
 * @param[in] size Memory allocation size in bytes.
 *
 * @return A pointer to the allocated memory on the symmetric heap.
 *
 * @todo Return error code instead of ptr.
 */
__host__ void*
roc_shmem_malloc(size_t size);

/**
 * @brief Free a memory allocation from the symmetric heap.
 * This is a collective operation and must be called by all PEs.
 *
 * @param[in] ptr Pointer to previously allocated memory on the symmetric heap.
 *
 * @return Status of the operation.
 */
__host__ void
roc_shmem_free(void* ptr);

/**
 * @brief Query for the number of PEs.
 *
 * @return Number of PEs.
 */
__host__ int
roc_shmem_n_pes();

/**
 * @brief Query the PE ID of the caller.
 *
 * @return PE ID of the caller.
 */
__host__ int
roc_shmem_my_pe();

/**
 * @brief Creates an OpenSHMEM context.
 *
 * @param[in] options Options for context creation. Ignored in current design.
 * @param[out] ctx    Context handle.
 *
 * @return Zero on success and nonzero otherwise.
 */
__host__ int
roc_shmem_ctx_create(int64_t options,
                     roc_shmem_ctx_t *ctx);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * @param[out] ctx    Context handle.
 *
 * @return void.
 */
__host__ void
roc_shmem_ctx_destroy(roc_shmem_ctx_t ctx);

/**
 * @brief Translate the PE in src_team to that in dest_team.
 *
 * @param[in] src_team  Handle of the team from which to translate
 * @param[in] src_pe    PE-of-interest's index in src_team
 * @param[in] dest_team Handle of the team to which to translate
 *
 * @return PE of src_pe in dest_team. If any input is invalid
 *         or if src_pe is not in both source and destination
 *         teams, a value of -1 is returned.
 */
__host__ int
roc_shmem_team_translate_pe(roc_shmem_team_t src_team,
                            int src_pe,
                            roc_shmem_team_t dest_team);

/**
 * @brief Query the number of PEs in a team.
 *
 * @param[in] team The team to query PE ID in.
 *
 * @return Number of PEs in the provided team.
 */
__host__ int
roc_shmem_team_n_pes(roc_shmem_team_t team);

/**
 * @brief Query the PE ID of the caller in a team.
 *
 * @param[in] team The team to query PE ID in.
 *
 * @return PE ID of the caller in the provided team.
 */
__host__ int
roc_shmem_team_my_pe(roc_shmem_team_t team);

/**
 * @brief Create a new a team of PEs. Must be called by all PEs
 * in the parent team.
 *
 * @param[in] parent_team The team to split from.
 * @param[in] start       The lowest PE number of the subset of the PEs
 *                        from the parent team that will form the new
 *                        team.
 * @param[in] stide       The stride between team PE members in the
 *                        parent team that comprise the subset of PEs
 *                        that will form the new team.
 * @param[in] size        The number of PEs in the new team.
 * @param[in] config      Pointer to the config parameters for the new
 *                        team.
 * @param[in] config_mask Bitwise mask representing parameters to use
 *                        from config
 * @param[out] new_team   Pointer to the newly created team. If an error
 *                        occurs during team creation, or if the PE in
 *                        the parent team is not in the new team, the
 *                        value will be ROC_SHMEM_TEAM_INVALID.
 *
 * @return Zero upon successful team creation; non-zero if erroneous.
 */
__host__ int
roc_shmem_team_split_strided(roc_shmem_team_t parent_team,
                             int start,
                             int stride,
                             int size,
                             const roc_shmem_team_config_t *config,
                             long config_mask,
                             roc_shmem_team_t *new_team);

/**
 * @brief Destroy a team. Must be called by all PEs in the team.
 * The user must destroy all private contexts created in the
 * team before destroying this team. Otherwise, the behavior
 * is undefined. This call will destroy only the shareable contexts
 * created from the referenced team.
 *
 * @param[in] team The team to destroy. The behavior is undefined if
 *                 the input team is ROC_SHMEM_TEAM_WORLD or any other
 *                 invalid team. If the input is ROC_SHMEM_TEAM_INVALID,
 *                 this function will not perform any operation.
 *
 * @return None.
 */
__host__ void
roc_shmem_team_destroy(roc_shmem_team_t team);

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
 */
__host__ Status
roc_shmem_dynamic_shared(size_t *shared_bytes);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into __host__ roc_shmem_quiet() if remote completion is required.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__host__ void
roc_shmem_ctx_putmem(roc_shmem_ctx_t ctx,
                     void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

__host__ void
roc_shmem_putmem(void *dest,
                 const void *source,
                 size_t nelems,
                 int pe);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * _host__ roc_shmem_quiet() if completion notification is required.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__host__ void
roc_shmem_ctx_putmem_nbi(roc_shmem_ctx_t ctx,
                         void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

__host__ void
roc_shmem_putmem_nbi(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

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
 */
__host__ void
roc_shmem_ctx_getmem(roc_shmem_ctx_t ctx,
                     void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

__host__ void
roc_shmem_getmem(void *dest,
                 const void *source,
                 size_t nelems,
                 int pe);

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller will
 * return as soon as the request is posted. The caller must call
 * __host__ roc_shmem_quiet() on the same context if completion notification is
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
 */
__host__ void
roc_shmem_ctx_getmem_nbi(roc_shmem_ctx_t ctx,
                         void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

__host__ void
roc_shmem_getmem_nbi(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * @param[in] ctx     Context with which to perform this operation.
 *
 * @return void.
 */
__host__ void
roc_shmem_ctx_fence(roc_shmem_ctx_t ctx);

__host__ void
roc_shmem_fence();

/**
 * @brief Completes all previous operations posted on the host.
 *
 * @param[in] ctx     Context with which to perform this operation.
 *
 * @return void.
 */
__host__ void
roc_shmem_ctx_quiet(roc_shmem_ctx_t ctx);

__host__ void
roc_shmem_quiet();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * @return void
 */
__host__ void
roc_shmem_barrier_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * @return void
 */
__host__ void
roc_shmem_sync_all();

/**
 * @brief allows any PE to force the termination of an entire program.
 *
 * @param[in] status    The exit status from the main program.
 *
 * @return void
 */
__host__ void
roc_shmem_global_exit(int status);


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
 */
__device__ void
roc_shmem_wg_init();

/**
 * @brief Finalizes device-side ROC_SHMEM resources. Must be called before
 * work-group completion if the work-group also called roc_shmem_wg_init().
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @return void.
 */
__device__ void
roc_shmem_wg_finalize();

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
 */
__device__ void
roc_shmem_wg_init_thread(int requested,
                         int *provided);

/**
 * @brief Query the thread mode used by the runtime.
 *
 * @param[out] provided Thread mode the runtime is operating in.
 *
 * @return void.
 */
__device__ void
roc_shmem_query_thread(int *provided);

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
 */
__device__ void
roc_shmem_wg_ctx_create(int64_t options,
                        roc_shmem_ctx_t *ctx);

__device__ int
roc_shmem_wg_team_create_ctx(roc_shmem_team_t team,
                             long options,
                             roc_shmem_ctx_t *ctx);

/**
 * @brief Destroys an OpenSHMEM context.
 *
 * Must be called collectively by all threads in the work-group.
 *
 * @param[in] The context to destroy.
 *
 * @return void.
 */
__device__ void
roc_shmem_wg_ctx_destroy(roc_shmem_ctx_t ctx);

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
 */
__device__ void
roc_shmem_ctx_putmem(roc_shmem_ctx_t ctx,
                     void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

__device__ void
roc_shmem_putmem(void *dest,
                 const void *source,
                 size_t nelems,
                 int pe);

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
 */
__device__ void
roc_shmem_ctx_getmem(roc_shmem_ctx_t ctx,
                     void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

__device__ void
roc_shmem_getmem(void *dest,
                 const void *source,
                 size_t nelems,
                 int pe);

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
 */
__device__ void
roc_shmem_ctx_putmem_nbi(roc_shmem_ctx_t ctx,
                         void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

__device__ void
roc_shmem_putmem_nbi(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

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
 */
__device__ void
roc_shmem_ctx_getmem_nbi(roc_shmem_ctx_t ctx,
                         void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

__device__ void
roc_shmem_getmem_nbi(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

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
 */
__device__ void
roc_shmem_ctx_fence(roc_shmem_ctx_t ctx);

__device__ void
roc_shmem_fence();


/**
 * @brief Guarantees order between messages in this context in accordance with
 * OpenSHMEM semantics.
 *
 * This function  is an extension as it is per PE. has same semantics as default
 * API but it is per PE
 *
 * @param[in] ctx Context with which to perform this operation.
 * @param[in] pe destination pe.
 *
 * @return void.
 */
__device__ void
roc_shmem_ctx_fence(roc_shmem_ctx_t ctx, int pe);

__device__ void
roc_shmem_fence(int pe);

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
 */
__device__ void
roc_shmem_ctx_quiet(roc_shmem_ctx_t ctx);

__device__ void
roc_shmem_quiet();

/**
 * @brief Query the total number of PEs.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] ctx GPU side handle.
 *
 * @return Total number of PEs.
 */
__device__ int
roc_shmem_ctx_n_pes(roc_shmem_ctx_t ctx);

__device__ int
roc_shmem_n_pes();

/**
 * @brief Query the PE ID of the caller.
 *
 * Can be called per thread with no performance penalty.
 *
 * @param[in] ctx GPU side handle
 *
 * @return PE ID of the caller.
 */
__device__ int
roc_shmem_ctx_my_pe(roc_shmem_ctx_t ctx);

__device__ int
roc_shmem_my_pe();

/**
 * @brief Translate the PE in src_team to that in dest_team.
 *
 * @param[in] src_team  Handle of the team from which to translate
 * @param[in] src_pe    PE-of-interest's index in src_team
 * @param[in] dest_team Handle of the team to which to translate
 *
 * @return PE of src_pe in dest_team. If any input is invalid
 *         or if src_pe is not in both source and destination
 *         teams, a value of -1 is returned.
 */
__device__ int
roc_shmem_team_translate_pe(roc_shmem_team_t src_team,
                            int src_pe,
                            roc_shmem_team_t dest_team);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @return void
 */
__device__ void
roc_shmem_ctx_wg_barrier_all(roc_shmem_ctx_t ctx);

__device__ void
roc_shmem_wg_barrier_all();

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
 */
__device__ void
roc_shmem_ctx_wg_sync_all(roc_shmem_ctx_t ctx);

__device__ void
roc_shmem_wg_sync_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ void
roc_shmem_ctx_wg_team_sync(roc_shmem_ctx_t ctx, roc_shmem_team_t team);

__device__ void
roc_shmem_wg_team_sync(roc_shmem_team_t team);

/**
 * @brief Query a local pointer to a symmetric data object on the
 * specified \pe . Returns an address that may be used to directly reference
 * dest on the specified \pe. This address can be accesses with LD/ST ops.
 *
 * Can be called per thread with no performance penalty.
 */
__device__ void*
roc_shmem_ptr(const void * dest, int pe);

/**
 * @brief Query the current time. Similar to gettimeofday() on the CPU. To use
 * this function, ROC_SHMEM must be configured with profiling support
 * (--enable-profile).
 *
 * Can be called per thread with no performance penalty.
 *
 * @return Time in micro-seconds.
 */
__device__ uint64_t
roc_shmem_timer();

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
 */
__device__ void
roc_shmem_ctx_threadfence_system(roc_shmem_ctx_t ctx);

/*
 * MACRO DECLARE SHMEM_REDUCTION APIs
 */
#define REDUCTION_API_GEN(T, TNAME, Op_API) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(roc_shmem_ctx_t ctx, \
                                                 T *dest, \
                                                 const T *source, \
                                                 int nreduce, \
                                                 int PE_start, \
                                                 int logPE_stride, \
                                                 int PE_size, \
                                                 T *pWrk, \
                                                 long *pSync);  /* NOLINT */ \
    __device__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(roc_shmem_ctx_t ctx, \
                                                 roc_shmem_team_t team, \
                                                 T *dest, \
                                                 const T *source, \
                                                 int nreduce); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_to_all(roc_shmem_ctx_t ctx, \
                                              T *dest, \
                                              const T *source, \
                                              int nreduce, \
                                              int PE_start, \
                                              int logPE_stride, \
                                              int PE_size, \
                                              T *pWrk, \
                                              long *pSync);     /* NOLINT */ \
    __host__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_to_all(roc_shmem_ctx_t ctx, \
                                              roc_shmem_team_t team, \
                                              T *dest, \
                                              const T *source, \
                                              int nreduce);

#define ARITH_REDUCTION_API_GEN(T, TNAME) \
    REDUCTION_API_GEN(T, TNAME, sum) \
    REDUCTION_API_GEN(T, TNAME, min) \
    REDUCTION_API_GEN(T, TNAME, max) \
    REDUCTION_API_GEN(T, TNAME, prod)

#define BITWISE_REDUCTION_API_GEN(T, TNAME) \
    REDUCTION_API_GEN(T, TNAME, or) \
    REDUCTION_API_GEN(T, TNAME, and) \
    REDUCTION_API_GEN(T, TNAME, xor)

#define INT_REDUCTION_API_GEN(T, TNAME) \
    ARITH_REDUCTION_API_GEN(T, TNAME) \
    BITWISE_REDUCTION_API_GEN(T, TNAME)

#define FLOAT_REDUCTION_API_GEN(T, TNAME) \
    ARITH_REDUCTION_API_GEN(T, TNAME)

/*
 * MACRO DECLARE SHMEM_BROADCAST APIs
 */
#define BROADCAST_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_broadcast(roc_shmem_ctx_t ctx, \
                                         T *dest, \
                                         const T *source, \
                                         int nelem, \
                                         int pe_root, \
                                         int pe_start, \
                                         int log_pe_stride, \
                                         int pe_size, \
                                         long *p_sync);                 /* NOLINT */ \
    __host__ void \
    roc_shmem_ctx_##TNAME##_broadcast(roc_shmem_ctx_t ctx, \
                                      T *dest, \
                                      const T *source, \
                                      int nelem, \
                                      int pe_root, \
                                      int pe_start, \
                                      int log_pe_stride, \
                                      int pe_size, \
                                      long *p_sync);                    /* NOLINT */ \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_broadcast(roc_shmem_ctx_t ctx, \
                                         roc_shmem_team_t team, \
                                         T *dest, \
                                         const T *source, \
                                         int nelem, \
                                         int pe_root);                  /* NOLINT */ \
    __host__ void \
    roc_shmem_ctx_##TNAME##_broadcast(roc_shmem_ctx_t ctx, \
                                      roc_shmem_team_t team, \
                                      T *dest, \
                                      const T *source, \
                                      int nelem, \
                                      int pe_root);                  /* NOLINT */

/*
 * MACRO DECLARE SHMEM_ALLTOALL APIs
 */
#define ALLTOALL_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_alltoall(roc_shmem_ctx_t ctx, \
                                        roc_shmem_team_t team, \
                                        T *dest, \
                                        const T *source, \
                                        int nelem);                  /* NOLINT */
/*
 * MACRO DECLARE SHMEM_FCOLLECT APIs
 */
#define FCOLLECT_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_fcollect(roc_shmem_ctx_t ctx, \
                                        roc_shmem_team_t team, \
                                        T *dest, \
                                        const T *source, \
                                        int nelem);                  /* NOLINT */

/*
 * MACRO DECLARE SHMEM_PUT APIs
 */
#define PUT_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_put(roc_shmem_ctx_t ctx, \
                                T *dest, \
                                const T *source, \
                                size_t nelems, \
                                int pe); \
    __device__ void \
    roc_shmem_##TNAME##_put(T *dest, \
                            const T *source, \
                            size_t nelems, \
                            int pe); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_put(roc_shmem_ctx_t ctx, \
                                T *dest, \
                                const T *source, \
                                size_t nelems, \
                                int pe); \
    __host__ void \
    roc_shmem_##TNAME##_put(T *dest, \
                            const T *source, \
                            size_t nelems, \
                            int pe);

/*
 * MACRO DECLARE SHMEM_P APIs
 */
#define P_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_p(roc_shmem_ctx_t ctx, \
                              T *dest, \
                              T value, \
                              int pe); \
    __device__ void \
    roc_shmem_##TNAME##_p(T *dest, \
                          T value, \
                          int pe); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_p(roc_shmem_ctx_t ctx, \
                              T *dest, \
                              T value, \
                              int pe); \
    __host__ void \
    roc_shmem_##TNAME##_p(T *dest, \
                          T value, \
                          int pe);

/*
 * MACRO DECLARE SHMEM_GET APIs
 */
#define GET_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_get(roc_shmem_ctx_t ctx, \
                                T *dest, \
                                const T *source, \
                                size_t nelems, int pe); \
    __device__ void \
    roc_shmem_##TNAME##_get(T *dest, \
                            const T *source, \
                            size_t nelems, \
                            int pe); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_get(roc_shmem_ctx_t ctx, \
                                T *dest, \
                                const T *source, \
                                size_t nelems, \
                                int pe); \
    __host__ void \
    roc_shmem_##TNAME##_get(T *dest, \
                            const T *source, \
                            size_t nelems, \
                            int pe);

/*
 * MACRO DECLARE SHMEM_G APIs
 */
#define G_API_GEN(T, TNAME) \
    __device__ T \
    roc_shmem_ctx_##TNAME##_g(roc_shmem_ctx_t ctx, \
                              const T *source, \
                              int pe); \
    __device__ T \
    roc_shmem_##TNAME##_g(const T *source, \
                          int pe); \
    __host__ T \
    roc_shmem_ctx_##TNAME##_g(roc_shmem_ctx_t ctx, \
                              const T *source, \
                              int pe); \
    __host__ T \
    roc_shmem_##TNAME##_g(const T *source, \
                          int pe);

/*
 * MACRO DECLARE SHMEM_PUT_NBI APIs
 */
#define PUT_NBI_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_put_nbi(roc_shmem_ctx_t ctx, \
                                    T *dest, \
                                    const T *source, \
                                    size_t nelems, \
                                    int pe); \
    __device__ void \
    roc_shmem_##TNAME##_put_nbi(T *dest, \
                                const T *source, \
                                size_t nelems, \
                                int pe); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_put_nbi(roc_shmem_ctx_t ctx, \
                                    T *dest, \
                                    const T *source, \
                                    size_t nelems, \
                                    int pe); \
    __host__ void \
    roc_shmem_##TNAME##_put_nbi(T *dest, \
                                const T *source, \
                                size_t nelems, \
                                int pe);

/*
 * MACRO DECLARE SHMEM_GET_NBI APIs
 */
#define GET_NBI_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_get_nbi(roc_shmem_ctx_t ctx, \
                                    T *dest, \
                                    const T *source, \
                                    size_t nelems, \
                                    int pe); \
    __device__ void \
    roc_shmem_##TNAME##_get_nbi(T *dest, \
                                const T *source, \
                                size_t nelems, \
                                int pe); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_get_nbi(roc_shmem_ctx_t ctx, \
                                    T *dest, \
                                    const T *source, \
                                    size_t nelems, \
                                    int pe); \
    __host__ void \
    roc_shmem_##TNAME##_get_nbi(T *dest, \
                                const T *source, \
                                size_t nelems, \
                                int pe);

/*
 * MACRO DECLARE SHMEM_ATOMIC_FETCH_ADD APIs
 */
#define ATOMIC_FETCH_ADD_API_GEN(T, TNAME) \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_add(roc_shmem_ctx_t ctx, \
                                             T *dest, \
                                             T value, \
                                             int pe); \
    __device__ T \
    roc_shmem_##TNAME##_atomic_fetch_add(T *dest, \
                                         T value, \
                                         int pe); \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_add(roc_shmem_ctx_t ctx, \
                                             T *dest, \
                                             T value, \
                                             int pe); \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch_add(T *dest, \
                                         T value, \
                                         int pe);

/*
 * MACRO DECLARE SHMEM_ATOMIC_COMPARE_SWAP APIs
 */
#define ATOMIC_COMPARE_SWAP_API_GEN(T, TNAME) \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_compare_swap(roc_shmem_ctx_t ctx, \
                                                T *dest, \
                                                T cond, \
                                                T value, \
                                                int pe); \
    __device__ T \
    roc_shmem_##TNAME##_atomic_compare_swap(T *dest, \
                                            T cond, \
                                            T value, \
                                            int pe); \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_compare_swap(roc_shmem_ctx_t ctx, \
                                                T *dest, \
                                                T cond, \
                                                T value, \
                                                int pe); \
    __host__ T \
    roc_shmem_##TNAME##_atomic_compare_swap(T *dest, \
                                            T cond, \
                                            T value, \
                                            int pe);

/*
 * MACRO DECLARE SHMEM_ATOMIC_FETCH_INC APIs
 */
#define ATOMIC_FETCH_INC_API_GEN(T, TNAME) \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_inc(roc_shmem_ctx_t ctx, \
                                             T *dest, \
                                             int pe); \
    __device__ T \
    roc_shmem_##TNAME##_atomic_fetch_inc(T *dest, \
                                         int pe); \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_inc(roc_shmem_ctx_t ctx, \
                                             T *dest, \
                                             int pe); \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch_inc(T *dest, \
                                         int pe);

/*
 * MACRO DECLARE SHMEM_ATOMIC_FETCH APIs
 */
#define ATOMIC_FETCH_API_GEN(T, TNAME) \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch(roc_shmem_ctx_t ctx, \
                                         T *dest, \
                                         int pe); \
    __device__ T \
    roc_shmem_##TNAME##_atomic_fetch(T *dest, \
                                     int pe); \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch(roc_shmem_ctx_t ctx, \
                                         T *dest, \
                                         int pe); \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch(T *dest, \
                                     int pe);

/*
 * MACRO DECLARE SHMEM_ATOMIC_ADD APIs
 */
#define ATOMIC_ADD_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_atomic_add(roc_shmem_ctx_t ctx, \
                                       T *dest, \
                                       T value, \
                                       int pe); \
    __device__ void \
    roc_shmem_##TNAME##_atomic_add(T *dest, \
                                   T value, \
                                   int pe); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_atomic_add(roc_shmem_ctx_t ctx, \
                                       T *dest, \
                                       T value, \
                                       int pe); \
    __host__ void \
    roc_shmem_##TNAME##_atomic_add(T *dest, \
                                   T value, \
                                   int pe);

/*
 * MACRO DECLARE SHMEM_ATOMIC_INC APIs
 */
#define ATOMIC_INC_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_atomic_inc(roc_shmem_ctx_t ctx, \
                                       T *dest, \
                                       int pe); \
    __device__ void \
    roc_shmem_##TNAME##_atomic_inc(T *dest, \
                                   int pe); \
    __host__ void \
    roc_shmem_ctx_##TNAME##_atomic_inc(roc_shmem_ctx_t ctx, \
                                       T *dest, \
                                       int pe); \
    __host__ void \
    roc_shmem_##TNAME##_atomic_inc(T *dest, \
                                   int pe);

/*
 * MACRO DECLARE SHMEM_WAIT_UNTIL APIs
 */
#define WAIT_UNTIL_API_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_##TNAME##_wait_until(T *ptr, \
                                   roc_shmem_cmps cmp, \
                                   T val); \
    __host__ void \
    roc_shmem_##TNAME##_wait_until(T *ptr, \
                                   roc_shmem_cmps cmp, \
                                   T val);

/*
 * MACRO DECLARE SHMEM_TEST APIs
 */
#define TEST_API_GEN(T, TNAME) \
    __device__ int \
    roc_shmem_##TNAME##_test(T *ptr, \
                             roc_shmem_cmps cmp, \
                             T val); \
    __host__ int \
    roc_shmem_##TNAME##_test(T *ptr, \
                             roc_shmem_cmps cmp, \
                             T val);

/**
 * @name SHMEM_REDUCTIONS
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
                           ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE).
 * @param[in] pSync        Temporary sync buffer provided to ROC_SHMEM. Must
                           be of size at least ROC_SHMEM_REDUCE_SYNC_SIZE.
 * @param[in] handle       GPU side handle.
 *
 * @return void
 */
///@{
INT_REDUCTION_API_GEN(int, int)
INT_REDUCTION_API_GEN(short, short)                     // NOLINT(runtime/int)
INT_REDUCTION_API_GEN(long, long)                       // NOLINT(runtime/int)
INT_REDUCTION_API_GEN(long long, longlong)              // NOLINT(runtime/int)
FLOAT_REDUCTION_API_GEN(float, float)
FLOAT_REDUCTION_API_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_API_GEN(long double, longdouble)
///@}

/**
 * @name SHMEM_BROADCAST
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcase completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelement     Size of the buffer to participate in the broadcast.
 * @param[in] PE_root      Zero-based ordinal of the PE, with respect to the
                           active set, from which the data is copied
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pSync        Temporary sync buffer provided to ROC_SHMEM. Must
                           be of size at least ROC_SHMEM_REDUCE_SYNC_SIZE.
 *
 * @return void
 */
///@{
BROADCAST_API_GEN(float, float)
BROADCAST_API_GEN(double, double)
BROADCAST_API_GEN(char, char)
// BROADCAST_API_GEN(long double, longdouble)
BROADCAST_API_GEN(signed char, schar)
BROADCAST_API_GEN(short, short)                         // NOLINT(runtime/int)
BROADCAST_API_GEN(int, int)
BROADCAST_API_GEN(long, long)                           // NOLINT(runtime/int)
BROADCAST_API_GEN(long long, longlong)                  // NOLINT(runtime/int)
BROADCAST_API_GEN(unsigned char, uchar)
BROADCAST_API_GEN(unsigned short, ushort)               // NOLINT(runtime/int)
BROADCAST_API_GEN(unsigned int, uint)
BROADCAST_API_GEN(unsigned long, ulong)                 // NOLINT(runtime/int)
BROADCAST_API_GEN(unsigned long long, ulonglong)        // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_ALLTOALL
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs 
 * of PEs participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return void
 */
///@{
ALLTOALL_API_GEN(float, float)
ALLTOALL_API_GEN(double, double)
ALLTOALL_API_GEN(char, char)
// ALLTOALL_API_GEN(long double, longdouble)
ALLTOALL_API_GEN(signed char, schar)
ALLTOALL_API_GEN(short, short)                         // NOLINT(runtime/int)
ALLTOALL_API_GEN(int, int)
ALLTOALL_API_GEN(long, long)                           // NOLINT(runtime/int)
ALLTOALL_API_GEN(long long, longlong)                  // NOLINT(runtime/int)
ALLTOALL_API_GEN(unsigned char, uchar)
ALLTOALL_API_GEN(unsigned short, ushort)               // NOLINT(runtime/int)
ALLTOALL_API_GEN(unsigned int, uint)
ALLTOALL_API_GEN(unsigned long, ulong)                 // NOLINT(runtime/int)
ALLTOALL_API_GEN(unsigned long long, ulonglong)        // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_FCOLLECT
 * @brief Concatenates blocks of data from multiple PEs to an array in every 
 * PE participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks in source array.
 *
 * @return void
 */
///@{
FCOLLECT_API_GEN(float, float)
FCOLLECT_API_GEN(double, double)
FCOLLECT_API_GEN(char, char)
// FCOLLECT_API_GEN(long double, longdouble)
FCOLLECT_API_GEN(signed char, schar)
FCOLLECT_API_GEN(short, short)                         // NOLINT(runtime/int)
FCOLLECT_API_GEN(int, int)
FCOLLECT_API_GEN(long, long)                           // NOLINT(runtime/int)
FCOLLECT_API_GEN(long long, longlong)                  // NOLINT(runtime/int)
FCOLLECT_API_GEN(unsigned char, uchar)
FCOLLECT_API_GEN(unsigned short, ushort)               // NOLINT(runtime/int)
FCOLLECT_API_GEN(unsigned int, uint)
FCOLLECT_API_GEN(unsigned long, ulong)                 // NOLINT(runtime/int)
FCOLLECT_API_GEN(unsigned long long, ulonglong)        // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_PUT
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
 */
///@{
PUT_API_GEN(float, float)
PUT_API_GEN(double, double)
PUT_API_GEN(char, char)
// PUT_API_GEN(long double, longdouble)
PUT_API_GEN(signed char, schar)
PUT_API_GEN(short, short)                               // NOLINT(runtime/int)
PUT_API_GEN(int, int)
PUT_API_GEN(long, long)                                 // NOLINT(runtime/int)
PUT_API_GEN(long long, longlong)                        // NOLINT(runtime/int)
PUT_API_GEN(unsigned char, uchar)
PUT_API_GEN(unsigned short, ushort)                     // NOLINT(runtime/int)
PUT_API_GEN(unsigned int, uint)
PUT_API_GEN(unsigned long, ulong)                       // NOLINT(runtime/int)
PUT_API_GEN(unsigned long long, ulonglong)              // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_P
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
 */
///@{
P_API_GEN(float, float)
P_API_GEN(double, double)
P_API_GEN(char, char)
// P_API_GEN(long double, longdouble)
P_API_GEN(signed char, schar)
P_API_GEN(short, short)                                 // NOLINT(runtime/int)
P_API_GEN(int, int)
P_API_GEN(long, long)                                   // NOLINT(runtime/int)
P_API_GEN(long long, longlong)                          // NOLINT(runtime/int)
P_API_GEN(unsigned char, uchar)
P_API_GEN(unsigned short, ushort)                       // NOLINT(runtime/int)
P_API_GEN(unsigned int, uint)
P_API_GEN(unsigned long, ulong)                         // NOLINT(runtime/int)
P_API_GEN(unsigned long long, ulonglong)                // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_GET
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
 */
///@{
GET_API_GEN(float, float)
GET_API_GEN(double, double)
GET_API_GEN(char, char)
// GET_API_GEN(long double, longdouble)
GET_API_GEN(signed char, schar)
GET_API_GEN(short, short)                               // NOLINT(runtime/int)
GET_API_GEN(int, int)
GET_API_GEN(long, long)                                 // NOLINT(runtime/int)
GET_API_GEN(long long, longlong)                        // NOLINT(runtime/int)
GET_API_GEN(unsigned char, uchar)
GET_API_GEN(unsigned short, ushort)                     // NOLINT(runtime/int)
GET_API_GEN(unsigned int, uint)
GET_API_GEN(unsigned long, ulong)                       // NOLINT(runtime/int)
GET_API_GEN(unsigned long long, ulonglong)              // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_G
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
 */
///@{
G_API_GEN(float, float)
G_API_GEN(double, double)
G_API_GEN(char, char)
// G_API_GEN(long double, longdouble)
G_API_GEN(signed char, schar)
G_API_GEN(short, short)                                 // NOLINT(runtime/int)
G_API_GEN(int, int)
G_API_GEN(long, long)                                   // NOLINT(runtime/int)
G_API_GEN(long long, longlong)                          // NOLINT(runtime/int)
G_API_GEN(unsigned char, uchar)
G_API_GEN(unsigned short, ushort)                       // NOLINT(runtime/int)
G_API_GEN(unsigned int, uint)
G_API_GEN(unsigned long, ulong)                         // NOLINT(runtime/int)
G_API_GEN(unsigned long long, ulonglong)                // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_PUT_NBI
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
 */
///@{
PUT_NBI_API_GEN(float, float)
PUT_NBI_API_GEN(double, double)
PUT_NBI_API_GEN(char, char)
// PUT_NBI_API_GEN(long double, longdouble)
PUT_NBI_API_GEN(signed char, schar)
PUT_NBI_API_GEN(short, short)                           // NOLINT(runtime/int)
PUT_NBI_API_GEN(int, int)
PUT_NBI_API_GEN(long, long)                             // NOLINT(runtime/int)
PUT_NBI_API_GEN(long long, longlong)                    // NOLINT(runtime/int)
PUT_NBI_API_GEN(unsigned char, uchar)
PUT_NBI_API_GEN(unsigned short, ushort)                 // NOLINT(runtime/int)
PUT_NBI_API_GEN(unsigned int, uint)
PUT_NBI_API_GEN(unsigned long, ulong)                   // NOLINT(runtime/int)
PUT_NBI_API_GEN(unsigned long long, ulonglong)          // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_GET_NBI
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
 */
///@{
GET_NBI_API_GEN(float, float)
GET_NBI_API_GEN(double, double)
GET_NBI_API_GEN(char, char)
// GET_NBI_API_GEN(long double, longdouble)
GET_NBI_API_GEN(signed char, schar)
GET_NBI_API_GEN(short, short)                           // NOLINT(runtime/int)
GET_NBI_API_GEN(int, int)
GET_NBI_API_GEN(long, long)                             // NOLINT(runtime/int)
GET_NBI_API_GEN(long long, longlong)                    // NOLINT(runtime/int)
GET_NBI_API_GEN(unsigned char, uchar)
GET_NBI_API_GEN(unsigned short, ushort)                 // NOLINT(runtime/int)
GET_NBI_API_GEN(unsigned int, uint)
GET_NBI_API_GEN(unsigned long, ulong)                   // NOLINT(runtime/int)
GET_NBI_API_GEN(unsigned long long, ulonglong)          // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_ATOMIC_FETCH_ADD
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
 */
///@{
ATOMIC_FETCH_ADD_API_GEN(int64_t, int64)
ATOMIC_FETCH_ADD_API_GEN(uint64_t, uint64)
// ATOMIC_FETCH_ADD_API_GEN(long long, longlong)
// ATOMIC_FETCH_ADD_API_GEN(unsigned long long, ulonglong)
// ATOMIC_FETCH_ADD_API_GEN(size_t, size)
// ATOMIC_FETCH_ADD_API_GEN(ptrdiff_t, ptrdiff)
///@}

/**
 * @name SHMEM_ATOMIC_COMPARE_SWAP
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
 */
///@{
ATOMIC_COMPARE_SWAP_API_GEN(int64_t, int64)
ATOMIC_COMPARE_SWAP_API_GEN(uint64_t, uint64)
// ATOMIC_COMPARE_SWAP_API_GEN(long long, longlong)
// ATOMIC_COMPARE_SWAP_API_GEN(unsigned long long, ulonglong)
// ATOMIC_COMPARE_SWAP_API_GEN(size_t, size)
// ATOMIC_COMPARE_SWAP_API_GEN(ptrdiff_t, ptrdiff)
///@}

/**
 * @name SHMEM_ATOMIC_FETCH_INC
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
 */
///@{
ATOMIC_FETCH_INC_API_GEN(int64_t, int64)
ATOMIC_FETCH_INC_API_GEN(uint64_t, uint64)
// ATOMIC_FETCH_INC_API_GEN(long long, longlong)
// ATOMIC_FETCH_INC_API_GEN(unsigned long long, ulonglong)
// ATOMIC_FETCH_INC_API_GEN(size_t, size)
// ATOMIC_FETCH_INC_API_GEN(ptrdiff_t, ptrdiff)
///@}

/**
 * @name SHMEM_ATOMIC_FETCH
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
 */
///@{
ATOMIC_FETCH_API_GEN(int64_t, int64)
ATOMIC_FETCH_API_GEN(uint64_t, uint64)
// ATOMIC_FETCH_API_GEN(long long, longlong)
// ATOMIC_FETCH_API_GEN(unsigned long long, ulonglong)
// ATOMIC_FETCH_API_GEN(size_t, size)
// ATOMIC_FETCH_API_GEN(ptrdiff_t, ptrdiff)
///@}

/**
 * @name SHMEM_ATOMIC_ADD
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
 */
///@{
ATOMIC_ADD_API_GEN(int64_t, int64)
ATOMIC_ADD_API_GEN(uint64_t, uint64)
// ATOMIC_ADD_API_GEN(long long, longlong)
// ATOMIC_ADD_API_GEN(unsigned long long, ulonglong)
// ATOMIC_ADD_API_GEN(size_t, size)
// ATOMIC_ADD_API_GEN(ptrdiff_t, ptrdiff)
///@}

/**
 * @name SHMEM_ATOMIC_INC
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
 */
///@{
ATOMIC_INC_API_GEN(int64_t, int64)
ATOMIC_INC_API_GEN(uint64_t, uint64)
// ATOMIC_INC_API_GEN(long long, longlong)
// ATOMIC_INC_API_GEN(unsigned long long, ulonglong)
// ATOMIC_INC_API_GEN(size_t, size)
// ATOMIC_INC_API_GEN(ptrdiff_t, ptrdiff)
///@}

/**
 * @name SHMEM_WAIT_UNTIL
 * @brief Block the caller until the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ptr Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return void
 */
///@{
WAIT_UNTIL_API_GEN(float, float)
WAIT_UNTIL_API_GEN(double, double)
WAIT_UNTIL_API_GEN(char, char)
// WAIT_UNTIL_API_GEN(long double, longdouble)
WAIT_UNTIL_API_GEN(signed char, schar)
WAIT_UNTIL_API_GEN(short, short)                        // NOLINT(runtime/int)
WAIT_UNTIL_API_GEN(int, int)
WAIT_UNTIL_API_GEN(long, long)                          // NOLINT(runtime/int)
WAIT_UNTIL_API_GEN(long long, longlong)                 // NOLINT(runtime/int)
WAIT_UNTIL_API_GEN(unsigned char, uchar)
WAIT_UNTIL_API_GEN(unsigned short, ushort)              // NOLINT(runtime/int)
WAIT_UNTIL_API_GEN(unsigned int, uint)
WAIT_UNTIL_API_GEN(unsigned long, ulong)                // NOLINT(runtime/int)
WAIT_UNTIL_API_GEN(unsigned long long, ulonglong)       // NOLINT(runtime/int)
///@}

/**
 * @name SHMEM_TEST
 * @brief test if the condition (* \p ptr \p cmps \p val) is
 * true.
 *
 * This function can be called from divergent control paths at per-thread
 * granularity. However, performance may be improved if the caller can
 * coalesce contiguous messages and elect a leader thread to call into the
 * ROC_SHMEM function.
 *
 * @param[in] ptr Pointer to memory on the symmetric heap to wait for.
 * @param[in] cmp Operation for the comparison.
 * @param[in] val Value to compare the memory at \p ptr to.
 *
 * @return 1 if the evaluation is true else 0
 */
///@{
TEST_API_GEN(float, float)
TEST_API_GEN(double, double)
TEST_API_GEN(char, char)
// TEST_API_GEN(long double, longdouble)
TEST_API_GEN(signed char, schar)
TEST_API_GEN(short, short)                              // NOLINT(runtime/int)
TEST_API_GEN(int, int)
TEST_API_GEN(long, long)                                // NOLINT(runtime/int)
TEST_API_GEN(long long, longlong)                       // NOLINT(runtime/int)
TEST_API_GEN(unsigned char, uchar)
TEST_API_GEN(unsigned short, ushort)                    // NOLINT(runtime/int)
TEST_API_GEN(unsigned int, uint)
TEST_API_GEN(unsigned long, ulong)                      // NOLINT(runtime/int)
TEST_API_GEN(unsigned long long, ulonglong)             // NOLINT(runtime/int)
///@}


/******************************************************************************
 ***************************** API EXTENSIONS *********************************
 *****************************************************************************/

/*
 * MACRO DECLARE SHMEMX_PUT APIs
 */
#define PUT_API_EXT_GEN(GRAN, T, TNAME) \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_put_##GRAN(roc_shmem_ctx_t ctx, \
                                        T *dest, \
                                        const T *source, \
                                        size_t nelems, \
                                        int pe); \
    __device__ void \
    roc_shmemx_##TNAME##_put_##GRAN(T *dest, \
                                    const T *source, \
                                    size_t nelems, \
                                    int pe);

/*
 * MACRO DECLARE SHMEMX_GET APIs
 */
#define GET_API_EXT_GEN(GRAN, T, TNAME) \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_get_##GRAN(roc_shmem_ctx_t ctx, \
                                        T *dest, \
                                        const T *source, \
                                        size_t nelems, \
                                        int pe); \
    __device__ void \
    roc_shmemx_##TNAME##_get_##GRAN(T *dest, \
                                    const T *source, \
                                    size_t nelems, \
                                    int pe);

/*
 * MACRO DECLARE SHMEMX_PUT_NBI APIs
 */
#define PUT_NBI_API_EXT_GEN(GRAN, T, TNAME) \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_put_nbi_##GRAN(roc_shmem_ctx_t ctx, \
                                            T *dest, \
                                            const T *source, \
                                            size_t nelems, \
                                            int pe); \
    __device__ void \
    roc_shmemx_##TNAME##_put_nbi_##GRAN(T *dest, \
                                        const T *source, \
                                        size_t nelems, \
                                        int pe);

/*
 * MACRO DECLARE SHMEMX_GET_NBI APIs
 */
#define GET_NBI_API_EXT_GEN(GRAN, T, TNAME) \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_get_nbi_##GRAN(roc_shmem_ctx_t ctx, \
                                            T *dest, \
                                            const T *source, \
                                            size_t nelems, \
                                            int pe); \
    __device__ void \
    roc_shmemx_##TNAME##_get_nbi_##GRAN(T *dest, \
                                        const T *source, \
                                        size_t nelems, \
                                        int pe);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into roc_shmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must participate in the
 * call using the same parameters.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_putmem_wave(roc_shmem_ctx_t ctx,
                           void *dest,
                           const void *source,
                           size_t nelems,
                           int pe);

__device__ void
roc_shmemx_putmem_wave(void *dest,
                       const void *source,
                       size_t nelems,
                       int pe);

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into roc_shmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * (WG) granularity. However, all threads in the workgroup must participate in
 * the call using the same parameters.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_putmem_wg(roc_shmem_ctx_t ctx,
                         void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

__device__ void
roc_shmemx_putmem_wg(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into roc_shmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must collectively participate
 * in the call using the same arguments
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
///@{
PUT_API_EXT_GEN(wave, float, float)
PUT_API_EXT_GEN(wave, double, double)
PUT_API_EXT_GEN(wave, char, char)
// PUT_API_EXT_GEN(wave, long double, longdouble)
PUT_API_EXT_GEN(wave, signed char, schar)
PUT_API_EXT_GEN(wave, short, short)                     // NOLINT(runtime/int)
PUT_API_EXT_GEN(wave, int, int)
PUT_API_EXT_GEN(wave, long, long)                       // NOLINT(runtime/int)
PUT_API_EXT_GEN(wave, long long, longlong)              // NOLINT(runtime/int)
PUT_API_EXT_GEN(wave, unsigned char, uchar)
PUT_API_EXT_GEN(wave, unsigned short, ushort)           // NOLINT(runtime/int)
PUT_API_EXT_GEN(wave, unsigned int, uint)
PUT_API_EXT_GEN(wave, unsigned long, ulong)             // NOLINT(runtime/int)
PUT_API_EXT_GEN(wave, unsigned long long, ulonglong)    // NOLINT(runtime/int)
///@}

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest at \p pe. The caller will block until the operation
 * completes locally (it is safe to reuse \p source). The caller must
 * call into roc_shmem_quiet() if remote completion is required.
 *
 * This function can be called from divergent control paths at per-workgroub (WG)
 * granularity. However, All threads in a WG must collectively participate in
 * the call using the same arguments.
 *
 * @param[in] ctx    Context with which to perform this operation.
 * @param[in] dest   Destination address. Must be an address on the symmetric
 *                   heap.
 * @param[in] source Source address. Must be an address on the symmetric heap.
 * @param[in] nelems Size of the transfer in number of elements.
 * @param[in] pe     PE of the remote process.
 *
 * @return void.
 */
///@{
PUT_API_EXT_GEN(wg, float, float)
PUT_API_EXT_GEN(wg, double, double)
PUT_API_EXT_GEN(wg, char, char)
// PUT_API_EXT_GEN(wg, long double, longdouble)
PUT_API_EXT_GEN(wg, signed char, schar)
PUT_API_EXT_GEN(wg, short, short)                       // NOLINT(runtime/int)
PUT_API_EXT_GEN(wg, int, int)
PUT_API_EXT_GEN(wg, long, long)                         // NOLINT(runtime/int)
PUT_API_EXT_GEN(wg, long long, longlong)                // NOLINT(runtime/int)
PUT_API_EXT_GEN(wg, unsigned char, uchar)
PUT_API_EXT_GEN(wg, unsigned short, ushort)             // NOLINT(runtime/int)
PUT_API_EXT_GEN(wg, unsigned int, uint)
PUT_API_EXT_GEN(wg, unsigned long, ulong)               // NOLINT(runtime/int)
PUT_API_EXT_GEN(wg, unsigned long long, ulonglong)      // NOLINT(runtime/int)
///@}

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a the wave must participate in the
 * call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_getmem_wave(roc_shmem_ctx_t ctx,
                           void *dest,
                           const void *source,
                           size_t nelems,
                           int pe);

__device__ void
roc_shmemx_getmem_wave(void *dest,
                       const void *source,
                       size_t nelems,
                       int pe);

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-workgroup
 * (WG) granularity. However, all threads in the workgroup must participate
 * in the call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_getmem_wg(roc_shmem_ctx_t ctx,
                         void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

__device__ void
roc_shmemx_getmem_wg(void *dest,
                     const void *source,
                     size_t nelems,
                     int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However,  all threads in the wave must participate in the
 * call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
///@{
GET_API_EXT_GEN(wave, float, float)
GET_API_EXT_GEN(wave, double, double)
GET_API_EXT_GEN(wave, char, char)
// GET_API_EXT_GEN(wave, long double, longdouble)
GET_API_EXT_GEN(wave, signed char, schar)
GET_API_EXT_GEN(wave, short, short)                     // NOLINT(runtime/int)
GET_API_EXT_GEN(wave, int, int)
GET_API_EXT_GEN(wave, long, long)                       // NOLINT(runtime/int)
GET_API_EXT_GEN(wave, long long, longlong)              // NOLINT(runtime/int)
GET_API_EXT_GEN(wave, unsigned char, uchar)
GET_API_EXT_GEN(wave, unsigned short, ushort)           // NOLINT(runtime/int)
GET_API_EXT_GEN(wave, unsigned int, uint)
GET_API_EXT_GEN(wave, unsigned long, ulong)             // NOLINT(runtime/int)
GET_API_EXT_GEN(wave, unsigned long long, ulonglong)    // NOLINT(runtime/int)
///@}

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The calling work-group will block until the
 * operation completes (data has been placed in \p dest).
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However,  all threads in the workgroup must participate in
 * the call using the same parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
///@{
GET_API_EXT_GEN(wg, float, float)
GET_API_EXT_GEN(wg, double, double)
GET_API_EXT_GEN(wg, char, char)
// GET_API_EXT_GEN(wg, long double, longdouble)
GET_API_EXT_GEN(wg, signed char, schar)
GET_API_EXT_GEN(wg, short, short)                       // NOLINT(runtime/int)
GET_API_EXT_GEN(wg, int, int)
GET_API_EXT_GEN(wg, long, long)                         // NOLINT(runtime/int)
GET_API_EXT_GEN(wg, long long, longlong)                // NOLINT(runtime/int)
GET_API_EXT_GEN(wg, unsigned char, uchar)
GET_API_EXT_GEN(wg, unsigned short, ushort)             // NOLINT(runtime/int)
GET_API_EXT_GEN(wg, unsigned int, uint)
GET_API_EXT_GEN(wg, unsigned long, ulong)               // NOLINT(runtime/int)
GET_API_EXT_GEN(wg, unsigned long long, ulonglong)      // NOLINT(runtime/int)
///@}

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in a wave must call in with the same
 * parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_putmem_nbi_wave(roc_shmem_ctx_t ctx,
                               void *dest,
                               const void *source,
                               size_t nelems,
                               int pe);

__device__ void
roc_shmemx_putmem_nbi_wave(void *dest,
                           const void *source,
                           size_t nelems,
                           int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
///@{
PUT_NBI_API_EXT_GEN(wave, float, float)
PUT_NBI_API_EXT_GEN(wave, double, double)
PUT_NBI_API_EXT_GEN(wave, char, char)
// PUT_NBI_API_EXT_GEN(wave, long double, longdouble)
PUT_NBI_API_EXT_GEN(wave, signed char, schar)
PUT_NBI_API_EXT_GEN(wave, short, short)                 // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wave, int, int)
PUT_NBI_API_EXT_GEN(wave, long, long)                   // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wave, long long, longlong)          // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wave, unsigned char, uchar)
PUT_NBI_API_EXT_GEN(wave, unsigned short, ushort)       // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wave, unsigned int, uint)
PUT_NBI_API_EXT_GEN(wave, unsigned long, ulong)         // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wave, unsigned long long, ulonglong)             // NOLINT
///@}

/**
 * @brief Writes contiguous data of \p nelems bytes from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in a WG must call in with the same
 * parameters
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_putmem_nbi_wg(roc_shmem_ctx_t ctx,
                             void *dest,
                             const void *source,
                             size_t nelems,
                             int pe);

__device__ void
roc_shmemx_putmem_nbi_wg(void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

/**
 * @brief Writes contiguous data of \p nelems elements from \p source on the
 * calling PE to \p dest on \p pe. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the sameo
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
                      heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
///@{
PUT_NBI_API_EXT_GEN(wg, float, float)
PUT_NBI_API_EXT_GEN(wg, double, double)
PUT_NBI_API_EXT_GEN(wg, char, char)
// PUT_NBI_API_EXT_GEN(wg, long double, longdouble)
PUT_NBI_API_EXT_GEN(wg, signed char, schar)
PUT_NBI_API_EXT_GEN(wg, short, short)                   // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wg, int, int)
PUT_NBI_API_EXT_GEN(wg, long, long)                     // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wg, long long, longlong)            // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wg, unsigned char, uchar)
PUT_NBI_API_EXT_GEN(wg, unsigned short, ushort)         // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wg, unsigned int, uint)
PUT_NBI_API_EXT_GEN(wg, unsigned long, ulong)           // NOLINT(runtime/int)
PUT_NBI_API_EXT_GEN(wg, unsigned long long, ulonglong)  // NOLINT(runtime/int)
///@}

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_getmem_nbi_wave(roc_shmem_ctx_t ctx,
                               void *dest,
                               const void *source,
                               size_t nelems,
                               int pe);

__device__ void
roc_shmemx_getmem_nbi_wave(void *dest,
                           const void *source,
                           size_t nelems,
                           int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-wave
 * granularity. However, all threads in the wave must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
///@{
GET_NBI_API_EXT_GEN(wave, float, float)
GET_NBI_API_EXT_GEN(wave, double, double)
GET_NBI_API_EXT_GEN(wave, char, char)
// GET_NBI_API_EXT_GEN(wave, long double, longdouble)
GET_NBI_API_EXT_GEN(wave, signed char, schar)
GET_NBI_API_EXT_GEN(wave, short, short)                 // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wave, int, int)
GET_NBI_API_EXT_GEN(wave, long, long)                   // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wave, long long, longlong)          // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wave, unsigned char, uchar)
GET_NBI_API_EXT_GEN(wave, unsigned short, ushort)       // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wave, unsigned int, uint)
GET_NBI_API_EXT_GEN(wave, unsigned long, ulong)         // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wave, unsigned long long, ulonglong)             // NOLINT
///@}

/**
 * @brief Reads contiguous data of \p nelems bytes from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
__device__ void
roc_shmemx_ctx_getmem_nbi_wg(roc_shmem_ctx_t ctx,
                             void *dest,
                             const void *source,
                             size_t nelems,
                             int pe);

__device__ void
roc_shmemx_getmem_nbi_wg(void *dest,
                         const void *source,
                         size_t nelems,
                         int pe);

/**
 * @brief Reads contiguous data of \p nelems elements from \p source on \p pe
 * to \p dest on the calling PE. The operation is not blocking. The caller
 * will return as soon as the request is posted. The caller must call
 * roc_shmem_quiet() on the same context if completion notification is
 * required.
 *
 * This function can be called from divergent control paths at per-workgroup
 * granularity. However, all threads in the WG must call in with the same
 * arguments.
 *
 * @param[in] ctx     Context with which to perform this operation.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Size of the transfer in bytes.
 * @param[in] pe      PE of the remote process.
 *
 * @return void.
 */
///@{
GET_NBI_API_EXT_GEN(wg, float, float)
GET_NBI_API_EXT_GEN(wg, double, double)
GET_NBI_API_EXT_GEN(wg, char, char)
// GET_NBI_API_EXT_GEN(wg, long double, longdouble)
GET_NBI_API_EXT_GEN(wg, signed char, schar)
GET_NBI_API_EXT_GEN(wg, short, short)                   // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wg, int, int)
GET_NBI_API_EXT_GEN(wg, long, long)                     // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wg, long long, longlong)            // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wg, unsigned char, uchar)
GET_NBI_API_EXT_GEN(wg, unsigned short, ushort)         // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wg, unsigned int, uint)
GET_NBI_API_EXT_GEN(wg, unsigned long, ulong)           // NOLINT(runtime/int)
GET_NBI_API_EXT_GEN(wg, unsigned long long, ulonglong)  // NOLINT(runtime/int)
///@}

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_INCLUDE_ROC_SHMEM_HPP
