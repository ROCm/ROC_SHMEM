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

namespace rocshmem {

struct ro_net_handle;

class HostInterface;
class ROHostContext;
class Transport;

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
     * @copydoc Backend::team_destroy(roc_shmem_team_t)
     */
    Status team_destroy(roc_shmem_team_t team) override;

    /**
     * @copydoc Backend::dynamic_shared(size_t*)
     */
    Status dynamic_shared(size_t *shared_bytes) override;

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
     * @brief The host-facing interface that will be used
     * by all contexts of the ROBackend
     */
    HostInterface *host_interface = nullptr;

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


 protected:
    /**
     * @copydoc Backend::dump_backend_stats()
     */
    Status dump_backend_stats() override;

    /**
     * @copydoc Backend::reset_backend_stats()
     */
    Status reset_backend_stats() override;

 public:
    /**
     * @brief Handle to itself.
     *
     * @todo Remove this member.
     */
    ro_net_handle *backend_handle = nullptr;

    /**
     * @brief Free all resources associated with the backend.
     *
     * @param[in,out] handle Pointer to the Backend object.
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
    Status ro_net_free_runtime(ro_net_handle *handle);

    /**
     * @brief Try to process one element from the queue.
     *
     * @param[in] queue_idx Index to access the queue_desc and queues fields.
     * @param[in] ro_net_gpu_handle Pointer to an internal data structure.
     * @param[in] finalized Unused parameter
     *
     * @return Boolean value with "True" indicating that one element was
     * process and "False" indicating that no valid queue element was
     * found.
     *
     * @todo Remove finalized parameter.
     */
    bool ro_net_process_queue(int queue_idx,
                              struct ro_net_handle *ro_net_gpu_handle,
                              bool *finalized);

    /**
     * @brief Calls into the HIP runtime to get fine-grained memory.
     *
     * @param[in,out] ptr Handle updated with newly allocated memory.
     * @param[in] size Requested memory size in bytes.
     */
    void ro_net_device_uc_malloc(void **ptr, size_t size);

 protected:
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
     * @brief Host-side buffer which holds a single device-side queue element.
     *
     * This small buffer is intended to be used as an optimization to access
     * the queue elements in device memory.
     */
    char *elt = nullptr;

    /**
     * @brief Handle for the transport class object.
     *
     * See the transport class for more details.
     */
    Transport *transport = nullptr;

    /**
     * @brief Workers used to poll on the device network request queues.
     */
    std::vector<std::thread> worker_threads;

    /**
     * @brief Holds a copy of the default context for host functions
     */
    ROHostContext *default_host_ctx = nullptr;
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_RO_HPP
