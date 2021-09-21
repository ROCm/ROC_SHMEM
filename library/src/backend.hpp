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

#ifndef LIBRARY_SRC_BACKEND_HPP_
#define LIBRARY_SRC_BACKEND_HPP_
/**
 * @file backend.hpp
 * Defines the Backend class.
 *
 * The file contains the Backend class, but also currently contains the
 * definitions for the RO_NET and GPUIB derived backend classes. Ideally,
 * these classes should reside in their own files, but the ROCM 2.10 HIP
 * compiler does not allow these to be declared in separate files. In the
 * future, the derived classes should be moved to their own files (along
 * with any derived class specific header files and forward declarations).
 */

/*
 * These headers are used by all backend classes.
 */
#include <mpi.h>
#include <pthread.h>

#include <map>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include <roc_shmem.hpp>

#include "config.h"  // NOLINT(build/include_subdir)
#include "stats.hpp"
#include "util.hpp"

/*
 * The following headers are used only by GPUIBBackend.
 *
 * Please move these headers to the derived class header file when this file
 * is split into pieces.
 */
#include "gpu_ib/ipc_policy.hpp"
#include "gpu_ib/network_policy.hpp"
#include "hdp_policy.hpp"


class HostInterface;
class WindowInfo;
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
     * @param num_wgs Number of device workgroups (which need resources).
     *
     * @note Implementation may reduce the number of workgroups if the
     * number exceeds hardware limits.
     *
     * @todo Change param type to size_t
     */
    explicit Backend(unsigned num_wgs);

    /**
     * @brief Destructor.
     */
    virtual ~Backend();

    /**
     * @brief Allocates network visible memory and returns ptr to caller.
     *
     * @param[in,out] ptr Pointer to memory handle.
     * @param[in] size Number of bytes of requested.
     *
     * @return Status code containing outcome.
     *
     * @todo Implementation must ensure that the ptr parameter is not nullptr.
     */
    virtual Status net_malloc(void **ptr, size_t size) = 0;

    /**
     * @brief Frees previously allocated network visible memory.
     *
     * @param[in] ptr Handle of previously allocated memory.
     *
     * @return Status code containing outcome.
     *
     * @todo Implementation must ensure that the ptr parameter is not nullptr.
     */
    virtual Status net_free(void *ptr) = 0;

    /**
     * @brief Reports library usage of shared (LDS) space.
     *
     * @param[in,out] shared_bytes Number of LDS bytes used by library.
     *
     * @return Status code containing outcome.
     *
     * @todo Implementation must ensure that the ptr parameter is not nullptr.
     */
    virtual Status dynamic_shared(size_t *shared_bytes) = 0;

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
    __device__ void create_wg_state();

    /**
     * @brief Frees device-side library resources.
     *
     * @return void
     */
    __device__ void finalize_wg_state();

    /**
     * @brief blocks until CPU init is done
     *
     * @return void
     */
    __device__ void wait_wg_init_done();

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
    Status dump_stats();

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
    Status reset_stats();

    /**
     * @brief Abort the application.
     *
     * @param[in] status Exit code.
     *
     * @return void.
     *
     * @note This routine terminates the entire application.
     */
    virtual void global_exit(int status) = 0;

    /**
     * @brief High level stats that do not depend on backend type.
     */
    ROCStats globalStats;
    ROCHostStats globalHostStats;

    /**
     * @brief Total number of workgroups launched on device.
     *
     * @todo Change to size_t.
     */
    unsigned num_wg = 0;

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
    unsigned int *bufferTokens = nullptr;

    /**
     * @brief Number of processing elements running in job.
     *
     * @todo Change to size_t.
     */
    int num_pes = 0;

    /**
     * @brief Unique numeric identifier ranging from 0 (inclusive) to
     * num_pes (exclusive) [0 ... num_pes).
     *
     * @todo Change to size_t and set invalid entry to max size.
     */
    int my_pe = -1;

    /**
     * @brief indicate when init is done on the CPU. Non-blocking init is only
     * available with GPU-IB
     */
    uint8_t *done_init = nullptr;
    MPI_Comm thread_comm;

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
    BackendType type = BackendType::GPU_IB_BACKEND;

    /**
     * @brief Dumps derived class statistics.
     *
     * @return Status code containing outcome.
     */
    virtual Status dump_backend_stats() = 0;

    /**
     * @brief Resets derived class statistics.
     *
     * @return Status code containing outcome.
     */
    virtual Status reset_backend_stats() = 0;
};

/**
 * @brief Global handle used by the device to access the backend.
 */
extern __constant__ Backend *gpu_handle;

/*
 * The following forward declarations are used only by ROBackend.
 *
 * Please move these declarations into the RO_NET header file when this file
 * is split into pieces.
 */
class Transport;
struct ro_net_handle;
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
 public:
    /**
     * @copydoc Backend::Backend(unsigned)
     */
    explicit ROBackend(unsigned num_wg);

    /**
     * @copydoc Backend::~Backend()
     */
    virtual ~ROBackend();

    /**
     * @copydoc Backend::net_malloc(void**,size_t)
     */
    Status net_malloc(void **ptr, size_t size) override;

    /**
     * @copydoc Backend::net_free(void*)
     */
    Status net_free(void *ptr) override;

    /**
     * @copydoc Backend::dynamic_shared(size_t*)
     */
    Status dynamic_shared(size_t *shared_bytes) override;

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

/*
 * The following forward declarations are used only by GPUIBBackend.
 *
 * Please move these declarations into the GPUIB header file when this file
 * is split into pieces.
 */
class Connection;
class QueuePair;
struct ibv_mr;
struct hdp_reg_t;

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
    explicit GPUIBBackend(unsigned num_wg);

    /**
     * @copydoc Backend::~Backend()
     */
    virtual ~GPUIBBackend();

    /**
     * @copydoc Backend::net_malloc(void**,size_t)
     */
    Status net_malloc(void **ptr, size_t size) override;

    /**
     * @copydoc Backend::net_free(void*)
     */
    Status net_free(void *ptr) override;

    /**
     * @copydoc Backend::dynamic_shared(size_t*)
     */
    Status dynamic_shared(size_t *shared_bytes) override;

    /**
     * @brief The host-facing interface that will be used
     * by all contexts of the GPUIBBackend
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
     * @brief Constant number which holds maximum workgroup size.
     *
     * This member depends on device hardware. It is only used in this
     * class to help calculate the symmetric heap offset during
     * initialization.
     *
     * @todo Remove this member from this class. It belongs in a class
     * that specifically holds device hardware information. If this
     * device class existed, we could consolidate the various flavours of
     * the Instinct cards into their own groups and then set these
     * hard-coded fields by querying the rocm runtime during our library
     * initialization.
     *
     * @todo Change to size_t.
     */
    const int MAX_WG_SIZE = 1024;

    /**
     * @brief Named constant for a gibibyte.
     *
     * @todo Change to size_t.
     */
    static constexpr int gibibyte = 1 << 30;

    /**
     * @brief Default symmetric heap size (in bytes).
     *
     * The environment variable, ROC_SHMEM_HEAP_SIZE, may be used to
     * override this value.
     *
     * @todo Change to size_t.
     */
    int heap_size = gibibyte;

    /**
     * @brief ROC_SHMEM's copy of MPI_COMM_WORLD (for interoperability
     * with orthogonal MPI usage in an MPI+ROC_SHMEM program).
     *
     * @todo move this to the team class when we implement it
     */
    MPI_Comm team_world_comm;

 protected:
    /**
     * #brief spawn a new thread to perform the rest of initialization
     * specially the connection establishement
     */
    std::thread thread_spawn(GPUIBBackend *b);

    void thread_func_internal(GPUIBBackend *b);
    /**
     * @brief initialize MPI.
     * GPUIB relies on MPI just to exchange the connection information.
     *
     * todo: remove the dependency on MPI and make it generic to PMI-X or just
     * to OpenSHMEM to have support for both CPU and GPU
     */
    Status init_mpi_once();

    /**
     * @brief find and init the HDP information
     */
    Status initialize_hdp();

    /**
     * @brief init the network support
     */
    Status initialize_network();


    /**
     * @brief Allocate symmetric heap and exchange heap information between
     * all processing elements.
     *
     * Every processing element allocates a symmetric heap. (For details on
     * the symmetric heap, refer to the OpenSHMEM specification.)
     * The symmetric heap is allocated using fine-grained memory to allow
     * both host access and device access to the memory space.
     *
     * The symmetric heaps are visible to network by registering them as
     * InfiniBand memory regions. Every memory region has a remote key
     * which needs to be shared across the network (to access the memory
     * region).
     *
     * This method is responsible for allocating the symmetric heap and
     * exchanging the necessary information to access the heap from any
     * processing element.
     *
     * @return Status code containing outcome.
     *
     * @todo Create a symmetric heap class and hide the initialization code
     * in the class constructor. The current implementation could probably
     * use some helper functions to improve code reuse. It looks like there
     * is probably a template that exists to transfer data for both the
     * rkeys and the base addresses instead of having copy and pasted the
     * logic for both types of information.
     */
    Status allocate_heap_memory();

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
     * @brief Allocate and initialize barrier operation addresses on
     * symmetric heap.
     *
     * When this method completes, the barrier_sync member will be available
     * for use.
     */
    void roc_shmem_collective_init();

 public:
    /**
     * @brief Policy choice for two HDP implementations.
     *
     * The hdp_policy member is a policy class object. The policy class
     * is determined at compile time with the CMake configuration option,
     * "USE_HDP_MAP". This option chooses between an custom kernel module
     * (available internally at AMD) or the default ROCM HDP API.
     *
     * @todo Combine HDP related stuff together into a class with a
     * reasonable interface. The functionality does not need to exist in
     * multiple pieces in the Backend and QueuePair classes. The hdp_rkey,
     * hdp_addresses, and hdp_policy fields should all live in the class.
     */
    HdpPolicy *hdp_policy = nullptr;

    /**
     * @brief C-array of symmetric heap base pointers.
     *
     * A C-array of char* pointers corresponding to the heap base pointers
     * virtual address for each processing element that we can communicate
     * with.
     */
    char **heap_bases = nullptr;

    /**
     * @brief Info about MPI window on the symmetric GPU heap
     */
    WindowInfo *heap_window_info = nullptr;

    /**
     * @brief MPI window for HDP flushing
     */
    // MPI_Win hdp_win;

    /**
     * @brief Pointer to the top of the symmetric heap.
     */
    size_t current_heap_offset = 0;

    /**
     * @brief Scratchpad for the internal barrier algorithms.
     */
    int64_t *barrier_sync = nullptr;

    /**
     * @brief Compile-time configuration policy for intra-node shared memory
     * accesses.
     *
     * The configuration option "USE_IPC" can be enabled to allow shared
     * memory accesses to the symmetric heap from processing elements
     * co-located on the same node.
     */
    IpcImpl ipcImpl;

    /**
     * @brief Compile-time configuration policy for network (IB)
     *
     *
     * The configuration option "USE_SINGLE_NODE" can be enabled to not build
     * with network support.
     */
    NetworkImpl networkImpl;

 private:
    /**
     * @brief a helper thread to perform the initialization (non-blocking init)
     */
    std::thread async_thread;

    /**
     * @brief Holds a copy of the default context (see OpenSHMEM
     * specification).
     *
     * @todo Remove this member from the backend class. There is another
     * copy stored in ROC_SHMEM_CTX_DEFAULT.
     */
    GPUIBContext *default_ctx = nullptr;

    /**
     * @brief Holds a copy of the default context for host functions
     */
    GPUIBHostContext *default_host_ctx = nullptr;
};

#endif  // LIBRARY_SRC_BACKEND_HPP_
