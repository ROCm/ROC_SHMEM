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

#ifndef BACKEND_HPP
#define BACKEND_HPP

#include "config.h"

#include <roc_shmem.hpp>
#include "stats.hpp"
#include "gpu_ib/connection_policy.hpp"
#include "hdp_policy.hpp"
#include "gpu_ib/ipc_policy.hpp"
#include "util.hpp"

class Context;

/*
 * Backend is a big container for the persistent state used by ROC_SHMEM.
 * It is populated by host-side initialization and allocation calls.  It uses
 * this information to populate Contexts that the GPU can use to perform
 * networking operations.
 */
class Backend
{
  public:

    /*
     * These virtual functions have a 1:1 correspondance with the same
     * operations in the roc_shmem.hpp public API.
     */
    virtual Status net_malloc(void **ptr, size_t size) = 0;

    virtual Status net_free(void *ptr) = 0;

    virtual Status dynamic_shared(size_t *shared_bytes) = 0;

    __host__ __device__ int getMyPE() const { return my_pe; }

    __host__ __device__ int getNumPEs() const { return num_pes; }

    __device__ void create_wg_state();

    __device__ void finalize_wg_state();

    /*
     * Dumps/resets globalStats before dispatching to derived classes
     * to dump more specific stats.
     */
    Status dump_stats();

    Status reset_stats();

    explicit Backend(unsigned num_wgs);

    virtual ~Backend();

    BackendType type = BackendType::GPU_IB_BACKEND;

    /*
     * High level stats that do not depend on choice of backend.
     */
    ROCStats globalStats;

    int num_pes = 0;

    int my_pe = -1;

    /*
     * Maximum number of work-groups that we need to allocate per-wg
     * global memory resources for.
     */
    unsigned num_wg = 0;

    /*
     * A num_wg size array used to keep track of per-wg buffer usage on the
     * GPU.  A value of '0' means that the buffers associated with the index
     * are free and can be claimed by a work-group during init.  A value of
     * '1' means that the buffers are currently in use by a work-group.
     */
    unsigned int *bufferTokens;

  protected:
    virtual Status dump_backend_stats() = 0;

    virtual Status reset_backend_stats() = 0;
};

extern __constant__ Backend *gpu_handle;

class Transport;
struct ro_net_handle;
struct roc_shmem;

/*
 * ROBackend (Revere Offload Transports)forwards GPU Requests to the host.
 */
class ROBackend : public Backend
{
  public:
    ro_net_handle *backend_handle = nullptr;

    Status net_malloc(void **ptr, size_t size) override;

    Status net_free(void *ptr) override;

    Status dynamic_shared(size_t *shared_bytes) override;

    explicit ROBackend(unsigned num_wg);

    virtual ~ROBackend();

    Status ro_net_free_runtime(ro_net_handle *handle);

    bool ro_net_process_queue(int queue_idx,
                              struct ro_net_handle *ro_net_gpu_handle,
                              bool *finalized);

    void ro_net_device_uc_malloc(void **ptr, size_t size);

  protected:
    Status dump_backend_stats() override;

    Status reset_backend_stats() override;

    void ro_net_poll(int thread_id, int num_threads);

    char *elt = nullptr;

    Transport *transport = nullptr;

    std::vector<std::thread> worker_threads;
};

class Connection;
class QueuePair;
class RTNGlobalHandle;
struct ibv_mr;
struct hdp_reg_t;
struct rtn_atomic_ret_t;

/*
 * GPU IB Backend talks to IB adaptors directly from the GPU.
 */
class  GPUIBBackend : public Backend
{
    static constexpr int gibibyte = 1 << 30;

    const int MAX_WG_SIZE = 1024;

    int heap_size = gibibyte;

  public:
    Status net_malloc(void **ptr, size_t size) override;

    Status net_free(void *ptr) override;

    Status dynamic_shared(size_t *shared_bytes) override;

    explicit GPUIBBackend(unsigned num_wg);

    virtual ~GPUIBBackend();

  protected:
    Status exchange_hdp_info();

    Status allocate_heap_memory();

    Status initialize_ipc();

    Status setup_atomic_region();

    Status setup_gpu_qps();

    Status setup_default_ctx();

    Status dump_backend_stats() override;

    Status reset_backend_stats() override;

    void roc_shmem_collective_init();

    void roc_shmem_g_init();

    Connection *connection = nullptr;

  public:
    /*
     * TODO: Too many duplicated fields between GPUIBBackend, GPUIBContext,
     * and QueuePair.  More work needs to be done to figure out which
     * duplications are necessary for performance (e.g. global memory vs LDS)
     * and which are not and can just be removed with better design.
     *
     * Necessary duplications should be factored into a class that can
     * be copy constructed from the original version of the data so we don't
     * have weird renamings of the same thing (e.g. base_heap in GPUIBContext
     * is a copy of heap_base here but named differently).
     */

    /*
     * Array of rkeys for remote HDP registers (one for each PE we can talk
     * to).  Used to implement fence().
     */
    uint32_t *hdp_rkey = nullptr;

    uintptr_t *hdp_address = nullptr;

    HdpPolicy *hdp_policy = nullptr;

    rtn_atomic_ret_t *atomic_ret = nullptr;

    /*
     * Collection of QueuePairs ready for GPUs to use for networking.  Will
     * be checked out to create Context classes.
     *
     * TODO: What we really need here is a collection of Contexts that can
     * either be copied into LDS or used directly by the GPU depending on
     * what type of context it is (shareable, serialized, or private).
     * No need to pool up QueuePairs, they can just be managed by
     * their owning Context.
     *
     * Should then consider pushing into base class since it's not
     * gpu-ib specific.
     */
    QueuePair *gpu_qps = nullptr;

    /*
     * Array of char * pointers corresponding to the heap base pointers VA for
     * each PE that we can communicate with.
     */
    char **heap_bases = nullptr;

    /*
     * Array of rkeys for remote sym heaps (one for each PE we can
     * talk to).
     */
    uint32_t *heap_rkey = nullptr;

    ibv_mr *heap_mr = nullptr;
    ibv_mr *hdp_mr = nullptr;
    ibv_mr *mr = nullptr;

    uint32_t lkey = 0;

    size_t current_heap_offset = 0;

    int64_t *barrier_sync = nullptr;

    /*
     * Buffer used to store the results of a *_g operation.  These ops do not
     * provide a destination buffer, so the runtime must manage one.
     */
    char *g_ret = nullptr;

    ConnectionImpl *connection_policy = nullptr;

    IpcImpl ipcImpl;

  private:
    GPUIBContext* default_ctx = nullptr;

};

#endif //BACKEND_HPP
