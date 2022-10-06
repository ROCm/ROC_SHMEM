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

#ifndef ROCSHMEM_LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP
#define ROCSHMEM_LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP

#include "backend_bc.hpp"
#include "network_policy.hpp"
#include "hip_allocator.hpp"
#include "hdp_policy.hpp"
#include "hdp_proxy.hpp"

namespace rocshmem {
Status
gpu_ib_get_dynamic_shared(size_t *shared_bytes, int num_pes);


class HostInterface;

/**
 * @class GPUIBBackend backend.hpp
 * @brief InfiniBand specific backend.
 *
 * The InfiniBand (GPUIB) backend enables the device to enqueue network
 * requests to InfiniBand queues (with minimal host intervention). The setup
 * requires some effort from the host, but the device is able to craft
 * InfiniBand requests and send them on its own.
 */
class GPUIBBackend : public Backend {
 public:
    /**
     * @copydoc Backend::Backend(unsigned)
     */
    explicit GPUIBBackend(size_t num_wg);

    /**
     * @copydoc Backend::~Backend()
     */
    virtual ~GPUIBBackend();

    /**
     * @brief Abort the application.
     *
     * @param[in] status Exit code.
     *
     * @return void.
     *
     * @note This routine terminates the entire application.
     */
    void global_exit(int status) override;

    /**
     * @copydoc Backend::create_new_team
     */
    Status create_new_team(Team *parent_team,
                           TeamInfo *team_info_wrt_parent,
                           TeamInfo *team_info_wrt_world,
                           int num_pes,
                           int my_pe_in_new_team,
                           MPI_Comm team_comm,
                           roc_shmem_team_t *new_team) override;

    /**
     * @copydoc Backend::team_destroy(roc_shmem_team_t)
     */
    Status team_destroy(roc_shmem_team_t team) override;


    /**
     * @copydoc Backend::ctx_create
     */
    void ctx_create(int64_t options, void **ctx) override;

    /**
     * @copydoc Backend::ctx_destroy
     */
    void ctx_destroy(Context *ctx) override;

 protected:
    /**
     * @copydoc Backend::dump_backend_stats()
     */
    Status dump_backend_stats() override;

    /**
     * @copydoc Backend::reset_backend_stats()
     */
    Status reset_backend_stats() override;

    /**
     * @brief spawn a new thread to perform the rest of initialization
     */
    std::thread thread_spawn(GPUIBBackend *b);

    /**
     * @brief overheads for helper thread to run
     *
     * @param[in] the thread needs access to the class
     *
     * @return void
     */
    void thread_func_internal(GPUIBBackend *b);

    /**
     * @brief initialize MPI.
     *
     * GPUIB relies on MPI just to exchange the connection information.
     *
     * todo: remove the dependency on MPI and make it generic to PMI-X or just
     * to OpenSHMEM to have support for both CPU and GPU
     */
    Status init_mpi_once();

    /**
     * @brief init the network support
     */
    Status initialize_network();

    /**
     * @brief Invokes the IPC policy class initialization method.
     *
     * This method delegates Inter Process Communication (IPC)
     * initialization to the appropriate policy class. The initialization
     * needs to be exposed to the Backed due to initialization ordering
     * constraints. (The symmetric heaps needs to be allocated and
     * initialized before this method can be called.)
     *
     * The policy class encapsulates what the initialization process so
     * refer to that class for more details.
     *
     * @return Status code containing outcome.
     */
    Status initialize_ipc();

    /**
     * @brief Allocate and initialize the ROC_SHMEM_CTX_DEFAULT variable.
     *
     * @return Status code containing outcome.
     *
     * @todo The default_ctx member looks unused after it is copied into
     * the ROC_SHMEM_CTX_DEFAULT variable.
     */
    Status setup_default_ctx();

    /**
     * @brief Allocate and initialize the default context for host
     * operations.
     *
     * @return Status code containing outcome.
     *
     */
    Status setup_default_host_ctx();

    /**
     * @brief Allocate and initialize team world.
     *
     * @return Status code containing outcome.
     *
     */
    Status setup_team_world();

    /**
     * @brief Initialize the resources required to support teams
     */
    void teams_init();

    /**
     * @brief Destruct the resources required to support teams
     */
    void teams_destroy();

    /**
     * @brief Allocate and initialize barrier operation addresses on
     * symmetric heap.
     *
     * When this method completes, the barrier_sync member will be available
     * for use.
     */
    void roc_shmem_collective_init();

 public:
    /**
     * @brief The host-facing interface that will be used
     * by all contexts of the GPUIBBackend
     */
    HostInterface *host_interface {nullptr};

    /**
     * @brief Handle for raw memory for barrier sync
     */
    long *barrier_pSync_pool {nullptr};

    /**
     * @brief Handle for raw memory for reduce sync
     */
    long *reduce_pSync_pool {nullptr};

    /**
     * @brief Handle for raw memory for broadcast sync
     */
    long *bcast_pSync_pool {nullptr};

    /**
     * @brief Handle for raw memory for alltoall sync
     */
    long *alltoall_pSync_pool {nullptr};

    /**
     * @brief Handle for raw memory for work
     */
    void *pWrk_pool {nullptr};

    /**
     * @brief Handle for raw memory for alltoall
     */
    void *pAta_pool {nullptr};

    /**
     * @brief ROC_SHMEM's copy of MPI_COMM_WORLD (for interoperability
     * with orthogonal MPI usage in an MPI+ROC_SHMEM program).
     */
    MPI_Comm gpu_ib_comm_world {};

  private:
    /**
     * @brief Allocates cacheable, device memory for the hdp policy.
     *
     * @note Internal data ownership is managed by the proxy
     */
    HdpProxy<HIPAllocator> hdp_proxy_ {};

  public:
    /**
     * @brief Policy choice for two HDP implementations.
     *
     * @todo Combine HDP related stuff together into a class with a
     * reasonable interface. The functionality does not need to exist in
     * multiple pieces in the Backend and QueuePair classes. The hdp_rkey,
     * hdp_addresses, and hdp_policy fields should all live in the class.
     */
    HdpPolicy *hdp_policy {hdp_proxy_.get()};

    /**
     * @brief Scratchpad for the internal barrier algorithms.
     */
    int64_t *barrier_sync {nullptr};

    /**
     * @brief Compile-time configuration policy for network (IB)
     *
     *
     * The configuration option "USE_SINGLE_NODE" can be enabled to not build
     * with network support.
     */
    NetworkImpl networkImpl {};

 private:
    /**
     * @brief The bitmask representing the availability of teams in the pool
     */
    char *pool_bitmask_ {nullptr};

    /**
     * @brief Bitmask to store the reduced result of bitmasks on pariticipating PEs
     *
     * With no thread-safety for this bitmask, multithreaded creation of teams is
     * not supported.
     */
    char *reduced_bitmask_ {nullptr};

    /**
     * @brief Size of the bitmask
     */
    int bitmask_size_ {-1};

    /**
     * @brief a helper thread to perform the initialization (non-blocking init)
     */
    std::thread async_thread_ {};

    /**
     * @brief Holds a copy of the default context (see OpenSHMEM
     * specification).
     *
     * @todo Remove this member from the backend class. There is another
     * copy stored in ROC_SHMEM_CTX_DEFAULT.
     */
    GPUIBContext *default_ctx_ {nullptr};

    /**
     * @brief Holds a copy of the default context for host functions
     */
    GPUIBHostContext *default_host_ctx_ {nullptr};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_GPU_IB_BACKEND_IB_HPP
