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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP

#include "backend_bc.hpp"

#include "backend_proxy.hpp"
#include "barrier_proxy.hpp"
#include "context_pool_proxy.hpp"
#include "default_context_proxy.hpp"
#include "hdp_proxy.hpp"
#include "hip_allocator.hpp"
#include "mpi_transport.hpp"
#include "profiler_proxy.hpp"
#include "queue_proxy.hpp"
#include "queue_desc_proxy.hpp"
#include "queue_element_proxy.hpp"
#include "ro_team_proxy.hpp"
#include "team_info_proxy.hpp"
#include "win_pool_bitmask_proxy.hpp"

namespace rocshmem {

class HostInterface;
class ROHostContext;

Status
ro_net_get_dynamic_shared(size_t *shared_bytes);

/**
 * @class ROBackend backend.hpp
 * @brief Reverse Offload Transports specific backend.
 *
 * The Reverse Offload (RO) backend class forwards device network requests to
 * the host (which allows the device to initiate network requests).
 * The word, "Reverse", denotes that the device is doing the offloading to
 * the host (which is an inversion of the normal behavior).
 */
class ROBackend : public Backend {
 public:
    /**
     * @copydoc Backend::Backend(unsigned)
     */
    explicit ROBackend(size_t num_wg);

    /**
     * @copydoc Backend::~Backend()
     */
    virtual ~ROBackend();

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

    /**
     * @brief Free all resources associated with the backend.
     *
     * The memory allocated to the handle param is deallocated during this
     * method. The handle should be treated as a nullptr after the call.
     *
     * The destructor treats this method as a helper function to destroy
     * this object.
     *
     * @return Status code containing outcome.
     *
     * @todo The method needs to be broken into smaller pieces and most
     * of these internal resources need to be moved into subclasses using
     * RAII.
     */
    Status ro_net_free_runtime();

    /**
     * @brief Try to process one element from the queue.
     *
     * @param[in] queue_idx Index to access the queue_desc and queues fields.
     *
     * @return Boolean value with "True" indicating that one element was
     * process and "False" indicating that no valid queue element was
     * found.
     */
    bool ro_net_process_queue(int queue_idx);

    /**
     * @brief The host-facing interface that will be used
     * by all contexts of the ROBackend
     */
    HostInterface *host_interface {nullptr};

    /**
     * @brief Handle to device memory fields.
     */
    BackendProxyT backend_proxy {};

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
     * @brief Service thread routine which spins on a number of queues until
     * the host calls net_finalize.
     *
     * @param[in] thread_id
     * @param[in] num_threads
     *
     * @todo Fix the assumption that only one gpu device exists in the
     * node.
     */
    void ro_net_poll(int thread_id, int num_threads);

    /**
     * @brief Helper to initialize IPC interface.
     */
    void
    initIPC();

    /**
     * @brief Handle for the transport class object.
     *
     * See the transport class for more details.
     */
    MPITransport transport_ {};

    /**
     * @brief Proxy for the team info used by the device.
     *
     * See the transport class for more details.
     */
    ROTeamProxyT team_world_proxy_ {this};

    /**
     * @brief Workers used to poll on the device network request queues.
     */
    std::vector<std::thread> worker_threads {};

    /**
     * @brief Holds a copy of the default context for host functions
     */
    std::unique_ptr<ROHostContext> default_host_ctx {nullptr};

    /**
     * @brief Maximum number of RO contexts in the application
     */
    int max_num_ctxs_ {-1};

    /**
     * @brief Pool of contexts for RO_NET
     *
     * TODO: @Brandon - This does not seem to be used. Figure out why
     * this was added and then not used.
     */
    ContextPoolProxyT ro_context_pool_proxy_ {&heap};

    /**
     * @brief Bitmask for allocating window pools
     *
     * TODO: @Brandon - This is not really a bitmask; it uses integers.
     * internally. Fix this as some point.
     */
    WinPoolBitmaskProxyT win_pool_bitmask_proxy_ {};

    /**
     * @brief Handle to device barrier memory.
     *
     * @note Internal data ownership is managed by the proxy
     */
    BarrierProxyT barrier_proxy_ {};

    /**
     * @brief Allocates uncacheable host memory for the hdp policy.
     *
     * @note Internal data ownership is managed by the proxy
     */
    HdpProxy<HIPHostAllocator> hdp_proxy_ {};

    /**
     * @brief Handle to device profiler memory
     *
     * @note Internal data ownership is managed by the proxy
     */
    ProfilerProxyT profiler_proxy_;  // init handled in constructor

    /**
     * @brief Host buffer for device memory copies
     *
     * @note Internal data ownership is managed by the proxy
     */
    QueueProxyT queue_proxy_ {};

    /**
     * @brief Host buffer for device memory copies
     *
     * @note Internal data ownership is managed by the proxy
     */
    QueueDescProxyT queue_desc_proxy_ {};

    /**
     * @brief Host buffer for device memory copies
     *
     * @note Internal data ownership is managed by the proxy
     */
    QueueElementProxyT queue_element_proxy_ {};

    /**
     * @brief Host-side buffer which holds a single device-side queue element.
     *
     * This small buffer is intended to be used as an optimization to access
     * the queue elements in device memory.
     */
    queue_element_t *queue_element_cache_ {queue_element_proxy_.get()};

    /**
     * @brief Proxy for the default context
     *
     * @note Internal data ownership is managed by the proxy
     */
    DefaultContextProxyT default_context_proxy_; // init handled in constructor
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP
