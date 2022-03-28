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

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <thread>

#include <roc_shmem.hpp>
#include "ro_net_internal.hpp"
#include "context_incl.hpp"
#include "backend_ro.hpp"
#include "backend_type.hpp"
#include "ro_net_team.hpp"
#include "transport.hpp"
#include "wg_state.hpp"
#include "atomic_return.hpp"

namespace rocshmem {

extern roc_shmem_ctx_t ROC_SHMEM_HOST_CTX_DEFAULT;

/***
 *
 * External Host-side API functions
 *
 ***/

ROBackend::~ROBackend()
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) backend_handle;

    ro_net_free_runtime(ro_net_gpu_handle);
}

Status
ROBackend::dynamic_shared(size_t *shared_bytes)
{
    auto max_num_teams {team_tracker.get_max_num_teams()};
    *shared_bytes = sizeof(ROContext) + sizeof(ro_net_wg_handle) +
        sizeof(WGState) + (max_num_teams * sizeof(WGTeamInfo));

    return Status::ROC_SHMEM_SUCCESS;
}

ROBackend::ROBackend(size_t num_wgs)
    : Backend(num_wgs)
{
    struct ro_net_handle *ro_net_gpu_handle;
    CHECK_HIP(hipHostMalloc((void**) &ro_net_gpu_handle,
                            sizeof(struct ro_net_handle *),
                            hipHostMallocCoherent));

    memset(ro_net_gpu_handle, 0, sizeof(ro_net_handle));

    backend_handle = ro_net_gpu_handle;
    type = BackendType::RO_BACKEND;

    char* value {nullptr};

    ro_net_gpu_handle->gpu_queue = true;
    if ((value = getenv("RO_NET_CPU_QUEUE")) != nullptr)
        ro_net_gpu_handle->gpu_queue = false;

    if (getenv("RO_NET_OPENSHMEM_TRANSPORT") != nullptr) {
#ifdef OPENSHMEM_TRANSPORT
        transport = new OpenSHMEMTransport();
#else
        std::cerr << "OpenSHMEM RO backend requested, but not built.  Please "
            "rebuild roc_shmem with OpenSHMEM transport support." << std::endl;
        exit(-1);
#endif
    } else {
        transport = new MPITransport();
    }

    CHECK_HIP(hipHostMalloc((void**) &ro_net_gpu_handle->hdp_policy,
                            sizeof(HdpPolicy), hipHostMallocCoherent));
    new (ro_net_gpu_handle->hdp_policy) HdpPolicy();

    if (!transport) {
        ro_net_free_runtime(ro_net_gpu_handle);
        exit(-static_cast<int>(Status::ROC_SHMEM_OOM_ERROR));
    }

    int num_threads = 1;

    if (num_threads > 0 &&
        ((num_wg < num_threads) || ((num_wg % num_threads) != 0))) {
        exit(-static_cast<int>(Status::ROC_SHMEM_INVALID_ARGUMENTS));
    }

    Status return_code;

    ro_net_gpu_handle->queue_size = DEFAULT_QUEUE_SIZE;
    if ((value = getenv("RO_NET_QUEUE_SIZE")) != nullptr) {
        ro_net_gpu_handle->queue_size = atoi(value);
        assert(ro_net_gpu_handle->queue_size != 0);
    }

    posix_memalign((void**)&elt, 64, sizeof(queue_element_t));
    if (!elt) {
        //net_free(ro_net_gpu_handle);
        exit(-static_cast<int>(Status::ROC_SHMEM_OOM_ERROR));
    }

    // allocate the resources for internal barriers
    unsigned int *barrier_ptr;
    CHECK_HIP(hipMalloc((void**) &barrier_ptr, sizeof(unsigned int )));
    *barrier_ptr=0;

    ROStats *profiler_ptr;
    CHECK_HIP(hipMalloc((void**) &profiler_ptr, sizeof(ROStats) * num_wg));
    new (profiler_ptr) ROStats();

    ro_net_gpu_handle->num_threads = num_threads;
    ro_net_gpu_handle->done_flag = 0;
    num_pes = transport->getNumPes();
    my_pe = transport->getMyPe();
    ro_net_gpu_handle->barrier_ptr = barrier_ptr;
    ro_net_gpu_handle->profiler = profiler_ptr;
#ifndef OPENSHMEM_TRANSPORT
    ro_net_gpu_handle->heap_window_info = heap.get_window_info();
    const auto& heap_bases = heap.get_heap_bases();
    ipcImpl.ipcHostInit(my_pe, heap_bases, transport->get_world_comm());

    init_g_ret(&heap,
               transport->get_world_comm(),
               num_wg,
               &ro_net_gpu_handle->g_ret);

    allocate_atomic_region(&ro_net_gpu_handle->atomic_ret, num_wg);

#endif

    if ((return_code = transport->initTransport(num_wg, ro_net_gpu_handle)) !=
        Status::ROC_SHMEM_SUCCESS) {
        //net_free(ro_net_gpu_handle);
        exit(-static_cast<int>(return_code));
    }

#ifdef OPENSHMEM_TRANSPORT
    std::cerr << "No host-facing support for OpenSHMEM RO backend" << std::endl;
    exit(-1);
#else
    {
        MPITransport *mpi_transport = static_cast<MPITransport *> (transport);
        host_interface = mpi_transport->host_interface;
    }
#endif

    default_host_ctx = new ROHostContext(*this, 0);
    ROC_SHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx;

    transport->default_host_ctx = default_host_ctx;

    /* Setup team world */
    TeamInfo *team_info_wrt_parent, *team_info_wrt_world;
    CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
    CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));

    new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);
    new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);

    MPI_Comm team_world_comm;
    MPI_Comm_dup(transport->get_world_comm(), &team_world_comm);

    ROTeam* team_world {nullptr};
    CHECK_HIP(hipMalloc(&team_world, sizeof(ROTeam)));
    new (team_world) ROTeam(*this,
                            team_info_wrt_parent,
                            team_info_wrt_world,
                            num_pes,
                            my_pe,
                            team_world_comm);
    team_tracker.set_team_world(team_world);

    ROC_SHMEM_TEAM_WORLD = (roc_shmem_team_t) team_world;
    /* Done setting up team world */

    queue_element_t **queues;
    CHECK_HIP(hipHostMalloc((void***)&queues,
                            sizeof(queue_element_t*) * num_wg,
                            hipHostMallocCoherent));

    ro_net_gpu_handle->queues = queues;

    queue_desc_t *queue_descs;
    if (ro_net_gpu_handle->gpu_queue) {
        ro_net_device_uc_malloc((void**) &queue_descs,
                                sizeof(queue_desc_t) * num_wg);
    } else {
        CHECK_HIP(hipHostMalloc((void**) &queue_descs,
                                sizeof(queue_desc_t) * num_wg,
                                hipHostMallocCoherent));
    }
    ro_net_gpu_handle->queue_descs = queue_descs;

    // Allocate circular buffer space for all queues.  Do all queues in a
    // single allocation since HIP currently doesn't handle a large number of
    // small allocations particularly well.
    if (ro_net_gpu_handle->gpu_queue) {
        ro_net_device_uc_malloc((void **) queues, num_wg *
                                sizeof(queue_element) *
                                ro_net_gpu_handle->queue_size);
    } else {
        CHECK_HIP(hipHostMalloc(queues,
                                num_wg * sizeof(queue_element_t) *
                                ro_net_gpu_handle->queue_size,
                                hipHostMallocCoherent));
    }
    memset(*queues, 0, num_wg * sizeof(queue_element_t) *
           ro_net_gpu_handle->queue_size);

    // Initialize queue descriptors
    for (int i = 0; i < num_wg; i++) {
        queues[i] = (*queues) + ro_net_gpu_handle->queue_size * i;
        queue_descs[i].read_idx = 0;
        queue_descs[i].write_idx = 0;
        // There is a status variable for each work-item in a work-group.  We
        // just overallocate for the maximum work-group size.
        int max_wg_size, gpu_dev;
        CHECK_HIP(hipGetDevice(&gpu_dev));
        CHECK_HIP(hipDeviceGetAttribute(&max_wg_size,
                                        hipDeviceAttributeMaxThreadsPerBlock,
                                        gpu_dev));
        // Status always goes in dGPU memory to prevent polling for completion
        // over PCIe
        ro_net_device_uc_malloc((void**) &queue_descs[i].status,
                                  max_wg_size * sizeof(char));

        memset(queue_descs[i].status, 0, max_wg_size * sizeof(char));
    }

    Context *ctx;
    CHECK_HIP(hipMalloc(&ctx, sizeof(ROContext)));
    new (ctx) ROContext(*this, 0);

    int * symbol_address;
    CHECK_HIP(hipGetSymbolAddress((void**)&symbol_address, HIP_SYMBOL(ROC_SHMEM_CTX_DEFAULT)));
    CHECK_HIP(hipMemcpy(symbol_address, &ctx, sizeof(ctx), hipMemcpyDefault));

    // Spawn threads to service the queues.
    for (int i = 0; i < num_threads; i++) {
        worker_threads.emplace_back(&ROBackend::ro_net_poll, this, i,
                                    num_threads);
    }

    *done_init = 1;
}

Status
ROBackend::team_destroy(roc_shmem_team_t team)
{
    ROTeam *team_obj = get_internal_ro_team(team);

    team_obj->~ROTeam();
    CHECK_HIP(hipFree(team_obj));

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ROBackend::create_new_team(Team *parent_team,
                           TeamInfo *team_info_wrt_parent,
                           TeamInfo *team_info_wrt_world,
                           int num_pes,
                           int my_pe_in_new_team,
                           MPI_Comm team_comm,
                           roc_shmem_team_t *new_team)
{
    transport->createNewTeam(this,
                             parent_team,
                             team_info_wrt_parent,
                             team_info_wrt_world,
                             num_pes,
                             my_pe_in_new_team,
                             team_comm,
                             new_team);
    return Status::ROC_SHMEM_SUCCESS;
}

/*
Status
ROBackend::net_malloc(void **ptr, size_t size)
{
    transport->allocateMemory(ptr, size);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
ROBackend::net_free(void * ptr)
{
    return transport->deallocateMemory(ptr);
}
*/
Status
ROBackend::reset_backend_stats()
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) backend_handle;

    for (int i = 0; i < num_wg; i++)
        ro_net_gpu_handle->profiler[i].resetStats();

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ROBackend::dump_backend_stats()
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) backend_handle;

    uint64_t total = 0;
    for (int i = 0; i < NUM_STATS; i++)
       total += globalStats.getStat(i);

    int gpu_frequency_khz = 27000;
    uint64_t us_wait_slot = 0;
    uint64_t us_pack = 0;
    uint64_t us_fence1 = 0;
    uint64_t us_fence2 = 0;
    uint64_t us_wait_host = 0;
    for (int i = 0; i < num_wg; i++) {
        // Average latency as perceived from a thread
        const ROStats &prof = ro_net_gpu_handle->profiler[i];
        us_wait_slot +=
            prof.getStat(WAITING_ON_SLOT) / (gpu_frequency_khz / 1000);
        us_pack += prof.getStat(PACK_QUEUE) / (gpu_frequency_khz / 1000);
        us_fence1 +=
            prof.getStat(THREAD_FENCE_1) / (gpu_frequency_khz / 1000);
        us_fence2 +=
            prof.getStat(THREAD_FENCE_2) / (gpu_frequency_khz / 1000);
        us_wait_host +=
            prof.getStat(WAITING_ON_HOST) / (gpu_frequency_khz / 1000);
    }

    const int FIELD_WIDTH = 20;
    const int FLOAT_PRECISION = 2;

    fprintf(stdout, "%*s%*s%*s%*s%*s\n",
            FIELD_WIDTH + 1, "Wait On Slot (us)",
            FIELD_WIDTH + 1, "Pack Queue (us)",
            FIELD_WIDTH + 1, "Fence 1 (us)",
            FIELD_WIDTH + 1, "Fence 2 (us)",
            FIELD_WIDTH + 1, "Wait Host (us)");

    fprintf(stdout,
                "%*.*f %*.*f %*.*f %*.*f %*.*f\n\n",
                FIELD_WIDTH, FLOAT_PRECISION, ((double) us_wait_slot) / total,
                FIELD_WIDTH, FLOAT_PRECISION, ((double) us_pack) / total,
                FIELD_WIDTH, FLOAT_PRECISION, ((double) us_fence1) / total,
                FIELD_WIDTH, FLOAT_PRECISION, ((double) us_fence2) / total,
                FIELD_WIDTH, FLOAT_PRECISION, ((double) us_wait_host) / total);

    fprintf(stdout, "PE %d: Queues %d Threads %d\n",
            my_pe, num_wg,
            ro_net_gpu_handle->num_threads);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ROBackend::ro_net_free_runtime(struct ro_net_handle * ro_net_gpu_handle)
{
    /*
     * Validate that a handle was passed that is not a nullptr.
     */
    assert(ro_net_gpu_handle);

    /*
     * Set this flag to denote that the runtime is being torn down.
     */
    ro_net_gpu_handle->done_flag = 1;

    /*
     * Tear down the worker threads.
     */
    for (auto &t : worker_threads) {
       t.join();
    }

    /*
     * Tear down the transport object.
     */
    if (transport) {
        while(!transport->readyForFinalize());
        transport->finalizeTransport();

        // TODO: For some reason this always seg faults.  I have no idea why.
        // Ignoring for now since its during tear-down anyway.
        delete transport;
    }

    /*
     * Tear down the HDP.
     */
    ro_net_gpu_handle->hdp_policy->~HdpPolicy();
    CHECK_HIP(hipHostFree((void*) ro_net_gpu_handle->hdp_policy));

    /*
     * Free the host-side single element used to do memory transfers from
     * device to host memory.
     */
    if (elt)
        free(elt);

    /*
     * Free the profiler statistics structure.
     */
    CHECK_HIP(hipFree(ro_net_gpu_handle->profiler));

    /*
     * Free the resources for internal barriers.
     */
    CHECK_HIP(hipFree(ro_net_gpu_handle->barrier_ptr));

    /*
     * Free the queue descriptors.
     */
    if (ro_net_gpu_handle->gpu_queue) {
        CHECK_HIP(hipFree(ro_net_gpu_handle->queue_descs));
    } else {
        CHECK_HIP(hipHostFree(ro_net_gpu_handle->queue_descs));
    }

    /*
     * Free the memory pointed to by the queues.
     */
    if (ro_net_gpu_handle->gpu_queue) {
        CHECK_HIP(hipFree(*ro_net_gpu_handle->queues));
    } else {
        CHECK_HIP(hipHostFree(*ro_net_gpu_handle->queues));
    }

    /*
     * Free the queue pointer itself.
     */
    CHECK_HIP(hipHostFree(ro_net_gpu_handle->queues));

    /*
     * Tear down team_world
     */
    auto* team_world {team_tracker.get_team_world()};
    team_world->~Team();
    CHECK_HIP(hipFree(team_world));

    /*
     * Free the default host context
     */
    delete default_host_ctx;

    /*
     * Free the gpu_handle.
     */
    CHECK_HIP(hipHostFree(ro_net_gpu_handle));

    return Status::ROC_SHMEM_SUCCESS;
}

bool
ROBackend::ro_net_process_queue(int queue_idx,
                                struct ro_net_handle *ro_net_gpu_handle,
                                bool *finalized)
{
    /*
     * Determine which indices to access in the queue.
     */
    queue_desc_t *queue_desc = &ro_net_gpu_handle->queue_descs[queue_idx];
    DPRINTF(("Queue Desc read_idx %zu\n", queue_desc->read_idx));
    uint64_t read_slot = queue_desc->read_idx %
        ro_net_gpu_handle->queue_size;

    /*
     * Check if next element from the device is ready.
     */
    const queue_element_t *next_element = nullptr;
    if (ro_net_gpu_handle->gpu_queue) {
        /*
         * Flush HDP read cache so we can see updates to the GPU Queue
         * descriptor.
         */
        ro_net_gpu_handle->hdp_policy->hdp_flush();

        /*
         * Copy the queue element from device memory to the host memory
         * elt buffer.
         */
        ::memcpy((void*)elt,
                 &ro_net_gpu_handle->queues[queue_idx][read_slot],
                 sizeof(queue_element_t));

        /*
         * Set our local variable to the next element.
         */
        next_element = reinterpret_cast<queue_element_t*>(elt);
    } else {
        /*
         * Set our local variable to the next element.
         */
        next_element = &ro_net_gpu_handle->queues[queue_idx][read_slot];
    }

    /*
     * By default, assume that no valid queue element exists. If a valid
     * queue element is found, we'll flip this return value to true.
     */
    bool valid = false;

    /*
     * Search for a single valid element and process that element through
     * the transport if one is found.
     */
    if (next_element->valid) {
        valid = true;

        DPRINTF(("Rank %d Processing read_slot %lu of queue %d \n",
                my_pe, read_slot, queue_idx));

        /*
         * Pass the queue element to the transport.
         * TODO: Who is responsible for freeing this memory?
         */
        transport->insertRequest(new queue_element_t(*next_element),
                                 queue_idx);

        /*
         * Toggle the queue flag back to invalid since the request was
         * just processed.
         */
        ro_net_gpu_handle->queues[queue_idx][read_slot].valid = 0;

        /*
         * Update the CPU's local read index.
         */
        queue_desc->read_idx++;
    }

    return valid;
}

void
ROBackend::ro_net_poll(int thread_id, int num_threads)
{
    /*
     * Cast the backend handle down into this struct to directly access
     * members.
     *
     * TODO: This seems error prone and needs to be changed.
     */
    ro_net_handle *ro_net_gpu_handle =
        reinterpret_cast<ro_net_handle*>(backend_handle);


    /*
     * This access assumes only one device exists and should not work
     * generally.
     */
    int gpu_dev = 0;
    CHECK_HIP(hipGetDevice(&gpu_dev));

    /*
     * Continue until the runtime is torn down.
     */
    while (!ro_net_gpu_handle->done_flag) {
        /*
         * Each worker thread is responsible for evaluating a queue index
         * at a time.
         */
        for (int i = thread_id; i < num_wg; i += num_threads) {
            /*
             * Drain up to 64 requests from this queue if they are ready.
             */
            int req_count = 0;

            /*
             * The finalize variable is not used.
             */
            bool finalize;

            /*
             * This variable will evaluate to "True" as long as valid queue
             * entries are found and processed. The loop ends when this
             * evalutes to "False" or we evaluate 64 entries.
             */
            bool processed_req;
            do {
                processed_req =
                    ro_net_process_queue(i, ro_net_gpu_handle, &finalize);
                req_count++;
            } while (processed_req && (req_count < 64));
        }
    }
}

void
ROBackend::ro_net_device_uc_malloc(void **ptr, size_t size)
{
    CHECK_HIP(hipExtMallocWithFlags(ptr, size, hipDeviceMallocFinegrained));
}

__host__ void
ROBackend::global_exit(int status)
{
    transport->global_exit(status);
}

}  // namespace rocshmem
