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

#ifndef ROCSHMEM_LIBRARY_SRC_BACKEND_BC_HPP
#define ROCSHMEM_LIBRARY_SRC_BACKEND_BC_HPP

/**
 * @file backend_bc.hpp
 * Defines the Backend base class.
 *
 * The backend base class sets up most of the host-side library resources.
 * It is the top-level interface for these resources.
 */

#include <mpi.h>
#include <roc_shmem.hpp>
#include <vector>

#include "backend_type.hpp"
#include "config.h"  // NOLINT(build/include_subdir)
#include "ipc_policy.hpp"
#include "stats.hpp"
#include "symmetric_heap.hpp"
#include "team_tracker.hpp"

namespace rocshmem {

class Team;
class TeamInfo;

/**
 * @class Backend backend.hpp
 * @brief Container class for the persistent state used by the library.
 *
 * Backend is populated by host-side initialization and allocation calls.
 * It uses this state to populate Context objects which the GPU may use to
 * perform networking operations.
 *
 * The roc_shmem.cpp implementation file wraps many the Backend public
 * members to implement the library's public API.
 */
class Backend {
 public:
    /**
     * @brief Constructor.
     *
     * @param[in] num_wgs Number of device workgroups (which need resources).
     *
     * @note Implementation may reduce the number of workgroups if the
     * number exceeds hardware limits.
     */
    explicit Backend(size_t num_wgs);

    /**
     * @brief Destructor.
     */
    virtual ~Backend();

    /**
     * @brief Create a new team object and initialize it.
     *
     * @param[in] parent_team Pointer to the parrent team object.
     * @param[in] team_info_wrt_parent TeamInfo object wrt parent team.
     * @param[in] team_info_wrt_world TeamInfo object wrt TEAM_WORLD.
     * @param[in] num_pes Number of PEs in this team.
     * @param[in] my_pe_in_new_team Index of this PE in the new team.
     * @param[in] team_comm MPI communicator for this team.
     *
     * @param[out] new_team pointer to the new team.
     *
     * @return Status code containing outcome.
     */
    virtual Status create_new_team(Team* parent_team,
                                   TeamInfo* team_info_wrt_parent,
                                   TeamInfo* team_info_wrt_world,
                                   int num_pes,
                                   int my_pe_in_new_team,
                                   MPI_Comm team_comm,
                                   roc_shmem_team_t* new_team) = 0;

    /**
     * @brief Destruct a team
     *
     * @param[in] team Handle to the team to destroy.
     */
    virtual Status team_destroy(roc_shmem_team_t team) = 0;

    /**
     * @brief Reports processing element number id.
     *
     * @return Unique numeric identifier for each processing element.
     */
    __host__ __device__
    int
    getMyPE() const {
        return my_pe;
    }

    /**
     * @brief Reports number of processing elements.
     *
     * @return Number of active processing elements tracked by library.
     */
    __host__ __device__
    int
    getNumPEs() const {
        return num_pes;
    }

    /**
     * @brief Allocates and initializes device-side library state.
     *
     * @return void
     */
    __device__ void
    create_wg_state();

    /**
     * @brief Frees device-side library resources.
     *
     * @return void
     */
    __device__ void
    finalize_wg_state();

    /**
     * @brief blocks until CPU init is done
     *
     * @return void
     */
    __device__ void
    wait_wg_init_done();

    /**
     * @brief Dumps statistics for public API invocations.
     *
     * @return Status code containing outcome.
     *
     * @note Implementation may dump additional statistics from backend
     * derived classes when calling this function. If so, the method,
     * dump_backend_stats, will be used as the interface for the
     * additional statistics.
     */
    Status
    dump_stats();

    /**
     * @brief Resets statistics for public API invocations.
     *
     * @return Status code containing outcome.
     *
     * @note Implementation may reset additional statistics from backend
     * derived classes when calling this function. If so, the method,
     * reset_backend_stats, will be used as the interface for the
     * additional statistics.
     */
    Status
    reset_stats();

    /**
     * @brief Abort the application.
     *
     * @param[in] status Exit code.
     *
     * @return void.
     *
     * @note This routine terminates the entire application.
     */
    virtual void
    global_exit(int status) = 0;

    /**
     * @brief Creates a new OpenSHMEM context.
     *
     * @param[in] options Options for context creation
     * @param[in] ctx     Address of the pointer to the new context
     *
     * @return Zero on success, nonzero otherwise.
     */
    virtual void ctx_create(int64_t options, void **ctx) = 0;

    /**
     * @brief Destroys a context.
     *
     * @param[in] ctx Context handle.
     *
     * @return void.
     */
    virtual void ctx_destroy(Context *ctx) = 0;

    /**
     * @brief High level device stats that do not depend on backend type.
     */
    ROCStats globalStats {};

    /**
     * @brief High level host stats that do not depend on backend type.
     */
    ROCHostStats globalHostStats {};

    /**
     * @brief Total number of workgroups launched on device.
     */
    size_t num_wg {0};

    /**
     * @brief Tracks per-workgroup buffer usage on the device.
     *
     * The implementation employs a dynamically allocated array (which is
     * the size of num_wg). The entries are integers, but only '0' and '1'
     * are used.
     *
     * A value of '0' means that the buffers associated with the index
     * are free and can be claimed by a workgroup during init. A value of
     * '1' means that the buffers are currently in use by a workgroup.
     *
     * @todo Change to bool.
     */
    unsigned* bufferTokens {nullptr};

    /**
     * @brief Number of processing elements running in job.
     *
     * @todo Change to size_t.
     */
    int num_pes {0};

    /**
     * @brief Unique numeric identifier ranging from 0 (inclusive) to
     * num_pes (exclusive) [0 ... num_pes).
     *
     * @todo Change to size_t and set invalid entry to max size.
     */
    int my_pe {-1};

    /**
     * @brief indicate when init is done on the CPU. Non-blocking init is only
     * available with GPU-IB
     */
    uint8_t* done_init {nullptr};

    /**
     * @todo document where this is used and try to coalesce this into another
     * class
     */
    MPI_Comm thread_comm {};

    /**
     * @brief Object contains the interface and internal data structures
     * needed to allocate/free memory on the symmetric heap.
     */
    SymmetricHeap heap {};

    /**
     * @brief Determines which device to launch device kernels onto.
     *
     * Multi-device nodes can specify which one they would like to use.
     */
    int hip_dev_id {0};

    /**
     * @brief Add ctx from the list of user-created ctxs
     */
    void track_ctx(Context *ctx, int64_t options);

    /**
     * @brief Remove ctx from the list of user-created ctxs
     */
    void untrack_ctx(Context *ctx);

    /**
     * @brief Remove all ctxs from the list of user-created ctxs
     */
    void destroy_remaining_ctxs();

    /**
     * @brief Compile-time configuration policy for intra-node shared memory
     * accesses.
     *
     * The configuration option "USE_IPC" can be enabled to allow shared
     * memory accesses to the symmetric heap from processing elements
     * co-located on the same node.
     */
    IpcImpl ipcImpl {};

    /**
     * @brief Maintains information about teams
     */
    TeamTracker team_tracker {};

 protected:
    /**
     * @brief Required to support static inheritance for device calls.
     *
     * The Context DISPATCH implementation requires this member.
     * The implementation needs to know the derived class type to
     * issue a static_cast.
     *
     * GPU devices do not support virtual functions. Therefore, we cannot
     * rely on the normal inheritance mechanism to tailor behavior for
     * derived backend types.
     */
    BackendType type {BackendType::GPU_IB_BACKEND};

    /**
     * @brief Dumps derived class statistics.
     *
     * @return Status code containing outcome.
     */
    virtual Status
    dump_backend_stats() = 0;

    /**
     * @brief Resets derived class statistics.
     *
     * @return Status code containing outcome.
     */
    virtual Status
    reset_backend_stats() = 0;

 private:
    /**
     * @brief List of ctxs created by the user.
     */
    std::vector<Context *> list_of_ctxs {};
};

/**
 * @brief Global handle used by the device to access the backend.
 */
extern __constant__ Backend* device_backend_proxy;

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_BACKEND_BC_HPP
