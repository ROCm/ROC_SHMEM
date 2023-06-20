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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP_

#include <memory>
#include <vector>

#include "src/backend_bc.hpp"
#include "src/hdp_proxy.hpp"
#include "src/memory/hip_allocator.hpp"
#include "src/reverse_offload/backend_proxy.hpp"
#include "src/reverse_offload/barrier_proxy.hpp"
#include "src/reverse_offload/block_handle.hpp"
#include "src/reverse_offload/window_proxy.hpp"
#include "src/reverse_offload/context_proxy.hpp"
#include "src/reverse_offload/mpi_transport.hpp"
#include "src/reverse_offload/profiler.hpp"
#include "src/reverse_offload/queue.hpp"
#include "src/reverse_offload/ro_team_proxy.hpp"
#include "src/reverse_offload/team_info_proxy.hpp"

namespace rocshmem {

class HostInterface;
class ROHostContext;

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
 const unsigned MAX_NUM_BLOCKS {65536};

 public:
  /**
   * @copydoc Backend::Backend(unsigned)
   */
  explicit ROBackend(MPI_Comm comm);

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
  void create_new_team(Team *parent_team, TeamInfo *team_info_wrt_parent,
                         TeamInfo *team_info_wrt_world, int num_pes,
                         int my_pe_in_new_team, MPI_Comm team_comm,
                         roc_shmem_team_t *new_team) override;

  /**
   * @copydoc Backend::team_destroy(roc_shmem_team_t)
   */
  void team_destroy(roc_shmem_team_t team) override;

  __device__ void create_ctx(int64_t options, roc_shmem_ctx_t *ctx);

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
   * @todo The method needs to be broken into smaller pieces and most
   * of these internal resources need to be moved into subclasses using
   * RAII.
   */
  void ro_net_free_runtime();

  /**
   * @brief The host-facing interface that will be used
   * by all contexts of the ROBackend
   */
  HostInterface *host_interface{nullptr};

  /**
   * @brief Handle to device memory fields.
   */
  BackendProxyT backend_proxy{};

  /**
   * @brief Handle to block resources
   */
  BlockHandleProxyT block_handle_proxy_;

  /**
   * @brief Handle to block resources
   */
  DefaultBlockHandleProxyT default_block_handle_proxy_;

 protected:
  /**
   * @copydoc Backend::dump_backend_stats()
   */
  void dump_backend_stats() override;

  /**
   * @copydoc Backend::reset_backend_stats()
   */
  void reset_backend_stats() override;

  /**
   * @brief Service thread routine which spins on a number of queues until
   * the host calls net_finalize.
   *
   * @todo Fix the assumption that only one gpu device exists in the
   * node.
   */
  void ro_net_poll();

  /**
   * @brief Helper to initialize IPC interface.
   */
  void initIPC();

  /**
   * @brief Handle for the transport class object.
   *
   * See the transport class for more details.
   */
  MPITransport *transport_;

  /**
   * @brief Proxy for the team info used by the device.
   *
   * See the transport class for more details.
   */
  ROTeamProxyT *team_world_proxy_;

  /**
   * @brief Workers used to poll on the device network request queues.
   */
  std::thread worker_thread{};

  /**
   * @brief Holds a copy of the default context for host functions
   */
  std::unique_ptr<ROHostContext> default_host_ctx{nullptr};

 public:
  /**
   * @brief Pool of contexts for RO_NET
   */
  WindowProxyT *ro_window_proxy_;

 protected:
  /**
   * @brief Handle to device barrier memory.
   *
   * @note Internal data ownership is managed by the proxy
   */
  BarrierProxyT barrier_proxy_{};

  /**
   * @brief Allocates uncacheable host memory for the hdp policy.
   *
   * @note Internal data ownership is managed by the proxy
   */
  HdpProxy<HIPHostAllocator> hdp_proxy_{};

  /**
   * @brief Handle to device profiler memory
   *
   * @note Internal data ownership is managed by the proxy
   */
  ProfilerProxyT profiler_proxy_;  // init handled in constructor

 public:
  /**
   * @brief Handle to network queues.
   */
  Queue queue_;

 protected:
  /**
   * @brief Proxy for the default context
   *
   * @note Internal data ownership is managed by the proxy
   */
  DefaultContextProxyT default_context_proxy_;  // init handled in constructor

  /**
   * @brief Proxy for the device-side block context
   *
   * @note Internal data ownership is managed by the proxy
   */
  BlockContextProxyT block_context_proxy_;  // init handled in constructor

  /**
   * @brief Controls how many thread blocks are monitored by polling thread.
   */
  size_t poll_block_count_{1};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP_
