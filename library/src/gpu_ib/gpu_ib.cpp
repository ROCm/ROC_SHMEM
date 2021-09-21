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

#include <endian.h>
#include <mpi.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>  // NOLINT(build/c++11)
#include <roc_shmem.hpp>

#include "context.hpp"
#include "backend.hpp"
#include "host.hpp"
#include "wg_state.hpp"

#include "queue_pair.hpp"

extern Context *ROC_SHMEM_HOST_CTX_DEFAULT;

Status
GPUIBBackend::net_free(void *ptr) {
    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::dump_backend_stats() {
    return networkImpl.dump_backend_stats(&globalStats);
}

Status
GPUIBBackend::reset_backend_stats() {
    return networkImpl.reset_backend_stats();
}

Status
GPUIBBackend::initialize_hdp() {
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&hdp_policy),
                        sizeof(HdpPolicy)));
    new (hdp_policy) HdpPolicy();
    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::allocate_heap_memory() {
    /*
     * Allocate host-side memory to hold symmetric heap base addresses for
     * all processing elements.
     */
    const size_t bases_size = num_pes * sizeof(void*);
    void **host_bases_cpy =
        reinterpret_cast<void**>(malloc(bases_size));
    if (host_bases_cpy == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    /*
     * Allocate device-side memory to hold symmetric heap base addresses for
     * all processing elements.
     */
    CHECK_HIP(hipMalloc(&heap_bases,
                        num_pes * sizeof(char*)));

    /*
     * Allocate fine-grained device-side memory for this processing
     * element's heap base.
     */
    void *base_heap;
    CHECK_HIP(hipExtMallocWithFlags(&base_heap,
                                    heap_size,
                                    hipDeviceMallocFinegrained));

    /*
     * Write this processing element's personal symmetric heap base to
     * the device-side memory using fine-grained memory accesses.
     */
    heap_bases[my_pe] = reinterpret_cast<char*>(base_heap);

    /*
     * Copy the device-side heap bases address array to the host-side heap
     * bases address array.
     */
    hipStream_t stream;
    hipStreamCreateWithFlags(&stream,
                             hipStreamNonBlocking);
    CHECK_HIP(hipMemcpyAsync(host_bases_cpy,
                             heap_bases,
                             bases_size,
                             hipMemcpyDeviceToHost,
                             stream));
    hipStreamSynchronize(stream);

    /*
     * Do all-to-all exchange of symmetric heap base address between the
     * processing elements.
     */
    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(void*),
                  MPI_CHAR,
                  host_bases_cpy,
                  sizeof(void*),
                  MPI_CHAR,
                  team_world_comm);

    /*
     * Copy the recently updated host-side heap base address array back to
     * the device-side memory.
     */
    CHECK_HIP(hipMemcpyAsync(heap_bases,
                             host_bases_cpy,
                             bases_size,
                             hipMemcpyHostToDevice,
                             stream));
    hipStreamSynchronize(stream);
    hipStreamDestroy(stream);

    /*
     * Create an MPI window on the allocated GPU heap, so that we can
     * offload the implementation of host-facing APIs on this memory
     * to the MPI library. Store the base of this window.
     */
    MPI_Win heap_win;
    MPI_Win_create(base_heap,
                   heap_size,
                   1,
                   MPI_INFO_NULL,
                   team_world_comm,
                   &heap_win);
    heap_window_info = new WindowInfo(heap_win,
                                      base_heap,
                                      heap_size);

    /*
     * Start a shared access epoch on windows of all ranks,
     * and let the library there is no need to check for
     * lock exclusivity during operations on this window
     * (MPI_MODE_NOCHECK).
     */
    MPI_Win_lock_all(MPI_MODE_NOCHECK, heap_window_info->get_win());

    /*
     * Free the host-side resources used to do the processing element
     * exchange of keys and addresses for the symmetric heap base.
     */
    free(host_bases_cpy);

    /*
     * Initialize the heap offset.
     * TODO(bpotter): this looks like it is set to zero twice
     * since it is also set to zero as a default member value.
     */
    current_heap_offset = 0;

    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::initialize_ipc() {
    ipcImpl.ipcHostInit(my_pe, heap_bases, thread_comm);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::initialize_network() {
    networkImpl.networkHostSetup(this);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::setup_default_host_ctx() {
    default_host_ctx = new GPUIBHostContext(*this, 0);
    ROC_SHMEM_HOST_CTX_DEFAULT = default_host_ctx;

    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::setup_default_ctx() {
    /*
     * Allocate device-side memory for default context and construct an
     * InfiniBand context in it.
     */
    CHECK_HIP(hipMalloc(&default_ctx, sizeof(GPUIBContext)));
    new (default_ctx) GPUIBContext(*this, 0);

    /*
     * Copy the symbol to ROC_SHMEM_CTX_DEFAULT.
     */
    int *symbol_address;
    CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&symbol_address),
                                  HIP_SYMBOL(ROC_SHMEM_CTX_DEFAULT)));


    hipStream_t stream;
    hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    CHECK_HIP(hipMemcpyAsync(symbol_address,
                             &default_ctx,
                             sizeof(default_ctx),
                             hipMemcpyDefault,
                             stream));
    hipStreamSynchronize(stream);
    hipStreamDestroy(stream);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::init_mpi_once() {
    static std::mutex init_mutex;
    const std::lock_guard<std::mutex> lock(init_mutex);

    int provided;
    int init_done = 0;
    if (MPI_Initialized(&init_done) == MPI_SUCCESS) {
        if (init_done) {
            return Status::ROC_SHMEM_SUCCESS;
        }
    }

    if (MPI_Init_thread(nullptr,
                        nullptr,
                        MPI_THREAD_MULTIPLE,
                        &provided)
                            != MPI_SUCCESS) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    return Status::ROC_SHMEM_SUCCESS;
}

std::thread
GPUIBBackend::thread_spawn(GPUIBBackend *b) {
    return std::thread (&GPUIBBackend::thread_func_internal, this, b);
}

void
GPUIBBackend::thread_func_internal(GPUIBBackend *b) {
    Status status;

    status = b->initialize_ipc();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    status = b->initialize_network();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    status = b->setup_default_ctx();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    *(b->done_init) = 1;
}

GPUIBBackend::GPUIBBackend(unsigned num_wgs)
    : Backend(num_wgs) {
    char *value;
    if ((value = getenv("ROC_SHMEM_HEAP_SIZE"))) {
        heap_size = atoi(value);
    }

    Status status;

    status = init_mpi_once();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    type = BackendType::GPU_IB_BACKEND;

    MPI_Comm_dup(MPI_COMM_WORLD, &team_world_comm);
    MPI_Comm_size(team_world_comm, &num_pes);
    MPI_Comm_rank(team_world_comm, &my_pe);

    status = allocate_heap_memory();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    /*
     * Initialize HDP here instead of in the async
     * thread since host-facing functions may use it.
     */
    status = initialize_hdp();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    /* Initialize the host interface */
    host_interface = new HostInterface(hdp_policy, team_world_comm);

    /*
     * Construct default host context independently of the
     * default device context (done in the async thread)
     * so that host operations can execute regardless of
     * device operations.
     */
    status = setup_default_host_ctx();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    roc_shmem_collective_init();

    MPI_Comm_dup(team_world_comm, &thread_comm);

    MPI_Barrier(team_world_comm);

    async_thread = thread_spawn(this);
}

void
GPUIBBackend::roc_shmem_collective_init() {
    /*
     * Grab a pointer to the top of the symmetric heap.
     */
    int64_t *ptr = reinterpret_cast<int64_t*>(heap_bases[my_pe]) +
                   current_heap_offset;

    /*
     * Increment the heap offset to create room for a barrier variable.
     */
    current_heap_offset += ROC_SHMEM_BARRIER_SYNC_SIZE;

    /*
     * Assign the barrier to the location at the previous top of the heap.
     */
    barrier_sync = ptr;

    /*
     * Initialize the barrier synchronization array with default values.
     */
    for (int i = 0; i < num_pes; i++) {
        barrier_sync[i] = ROC_SHMEM_SYNC_VALUE;
    }

    /*
     * Make sure that all processing elements have done this before
     * continuing.
     */
    MPI_Barrier(team_world_comm);
}

GPUIBBackend::~GPUIBBackend() {
    async_thread.join();

    hdp_policy->~HdpPolicy();
    CHECK_HIP(hipFree(hdp_policy));
    hdp_policy = nullptr;

    delete host_interface;
    host_interface = nullptr;

    /*
     * Close the access epoch, free the window,
     * free my heap, and free the array that
     * stores the bases of the heaps.
     */
    MPI_Win heap_win = heap_window_info->get_win();
    MPI_Win_unlock_all(heap_win);
    MPI_Win_free(&heap_win);
    delete heap_window_info;
    heap_window_info = nullptr;

    CHECK_HIP(hipFree(heap_bases[my_pe]));
    CHECK_HIP(hipFree(heap_bases));
    heap_bases = nullptr;

    MPI_Comm_free(&team_world_comm);

    CHECK_HIP(hipFree(default_ctx->rtn_gpu_handle));
    CHECK_HIP(hipFree(default_ctx));
    default_ctx = nullptr;

    networkImpl.networkHostFinalize();
}

Status
GPUIBBackend::net_malloc(void **ptr,
                         size_t size) {
    *ptr = reinterpret_cast<char*>(heap_bases[my_pe]) + current_heap_offset;

    current_heap_offset = current_heap_offset + (size / sizeof(char));

    if (current_heap_offset > heap_size) {
        *ptr = nullptr;
        printf("ERROR: SHeap allocation failed\n");
        return  Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    MPI_Barrier(team_world_comm);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
GPUIBBackend::dynamic_shared(size_t *shared_bytes) {
    uint32_t heap_usage = num_pes * sizeof(uint64_t);
    uint32_t network_usage = networkImpl.networkDynamicShared();
    uint32_t ipc_usage = ipcImpl.ipcDynamicShared();

    *shared_bytes = heap_usage +
                    network_usage +
                    ipc_usage +
                    sizeof(GPUIBContext) +
                    sizeof(WGState);

    return Status::ROC_SHMEM_SUCCESS;
}

__host__ void
GPUIBBackend::global_exit(int status) {
    MPI_Abort(team_world_comm, status);
}
