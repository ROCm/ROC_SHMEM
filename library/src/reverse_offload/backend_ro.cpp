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
#include "mpi_transport.hpp"
#include "wg_state.hpp"
#include "atomic_return.hpp"

namespace rocshmem {

extern roc_shmem_ctx_t ROC_SHMEM_HOST_CTX_DEFAULT;

Status
ro_net_get_dynamic_shared(size_t *shared_bytes) {
    TeamTracker tr {};
    auto max_num_teams {tr.get_max_num_teams()};

    *shared_bytes = sizeof(ROContext) + \
                    sizeof(ro_net_wg_handle) +
                    sizeof(WGState) + \
                    max_num_teams * sizeof(WGTeamInfo);

    return Status::ROC_SHMEM_SUCCESS;
}


ROBackend::ROBackend(size_t num_wgs)
    : profiler_proxy_(num_wgs),
      Backend(num_wgs) {
    type = BackendType::RO_BACKEND;

    // TODO: @Brandon move num_pes and my_pe init to base class
    num_pes = transport_.getNumPes();

    my_pe = transport_.getMyPe();

    char *value {nullptr};

    auto *bp {backend_proxy.get()};

    bp->gpu_queue = true;
    if ((value = getenv("RO_NET_CPU_QUEUE")) != nullptr) {
        bp->gpu_queue = false;
    }

    bp->queue_size = DEFAULT_QUEUE_SIZE;
    if ((value = getenv("RO_NET_QUEUE_SIZE")) != nullptr) {
        bp->queue_size = atoi(value);
        assert(bp->queue_size != 0);
    }

    /* Allocate pool of windows for RO_NET contexts */
    if ((value = getenv("RO_NET_MAX_NUM_CONTEXTS"))) {
        max_num_ctxs_ = atoi(value);
    } else {
        max_num_ctxs_ = num_wgs;
    }

    bp->max_num_ctxs = max_num_ctxs_;

    bp->hdp_policy = hdp_proxy_.get();

    bp->profiler = profiler_proxy_.get();

    bp->barrier_ptr = barrier_proxy_.get();

    bp->done_flag = 0;

    bp->heap_ptr = &heap;

    bp->heap_window_info = ro_context_pool_proxy_.get();

    initIPC();

    init_g_ret(&heap,
               transport_.get_world_comm(),
               num_wg,
               &bp->g_ret);

    allocate_atomic_region(&bp->atomic_ret,
                           num_wg);

    transport_.initTransport(num_wg,
                             &backend_proxy);

    host_interface = transport_.host_interface;

    default_host_ctx = std::make_unique<ROHostContext>(this, 0);

    ROC_SHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();

    team_tracker.set_team_world(team_world_proxy_.get());

    ROC_SHMEM_TEAM_WORLD = reinterpret_cast<roc_shmem_team_t>(team_world_proxy_.get());

    bp->win_pool_alloc_bitmask = win_pool_bitmask_proxy_.get();

    /* Done allocating pool of windows for RO_NET contexts */

    bp->queues = queue_proxy_.get();

    bp->queue_descs = queue_desc_proxy_.get();

    default_context_proxy_ = std::move(DefaultContextProxyT(this));

    int num_threads {1};
    if ((value = getenv("RO_NET_NUM_THREADS")) != nullptr) {
        num_threads = atoi(value);
    }

    if (num_threads > 0 &&
        ((num_wg < num_threads) || ((num_wg % num_threads) != 0))) {
        exit(-static_cast<int>(Status::ROC_SHMEM_INVALID_ARGUMENTS));
    }

    bp->num_threads = num_threads;

    // Spawn threads to service the queues.
    for (size_t i {0}; i < num_threads; i++) {
        worker_threads.emplace_back(&ROBackend::ro_net_poll,
                                    this,
                                    i,
                                    num_threads);
    }

    *done_init = 1;
}

ROBackend::~ROBackend() {
    ro_net_free_runtime();
}

Status
ROBackend::team_destroy(roc_shmem_team_t team) {
    ROTeam *team_obj {get_internal_ro_team(team)};

    team_obj->~ROTeam();
    //CHECK_HIP(hipFree(team_obj));

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ROBackend::create_new_team(Team *parent_team,
                           TeamInfo *team_info_wrt_parent,
                           TeamInfo *team_info_wrt_world,
                           int num_pes,
                           int my_pe_in_new_team,
                           MPI_Comm team_comm,
                           roc_shmem_team_t *new_team) {
    transport_.createNewTeam(this,
                             parent_team,
                             team_info_wrt_parent,
                             team_info_wrt_world,
                             num_pes,
                             my_pe_in_new_team,
                             team_comm,
                             new_team);
    return Status::ROC_SHMEM_SUCCESS;
}

void
ROBackend::ctx_create(int64_t options, void **ctx) {
    ROHostContext *new_ctx {nullptr};
    new_ctx = new ROHostContext(this, options);
    *ctx = new_ctx;
}

ROHostContext*
get_internal_ro_net_ctx(Context *ctx) {
    return reinterpret_cast<ROHostContext*>(ctx);
}

void
ROBackend::ctx_destroy(Context *ctx) {
    ROHostContext *ro_net_host_ctx {get_internal_ro_net_ctx(ctx)};
    delete ro_net_host_ctx;
}

Status
ROBackend::reset_backend_stats() {
    auto *bp {backend_proxy.get()};

    for (size_t i {0}; i < num_wg; i++) {
        bp->profiler[i].resetStats();
    }

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ROBackend::dump_backend_stats() {
    uint64_t total {0};
    for (int i = 0; i < NUM_STATS; i++) {
       total += globalStats.getStat(i);
    }

    /*
     * TODO: @Brandon
     * No idea where these numbers come from
     */
    constexpr uint64_t gpu_frequency_khz {27000};
    constexpr uint64_t gpu_frequency_mhz {gpu_frequency_khz / 1000};

    uint64_t us_wait_slot {0};
    uint64_t us_pack {0};
    uint64_t us_fence1 {0};
    uint64_t us_fence2 {0};
    uint64_t us_wait_host {0};

    auto *bp {backend_proxy.get()};

    for (size_t i {0}; i < num_wg; i++) {
        // Average latency as perceived from a thread
        const ROStats &prof {bp->profiler[i]};
        us_wait_slot += prof.getStat(WAITING_ON_SLOT) / gpu_frequency_mhz;
        us_pack += prof.getStat(PACK_QUEUE) / gpu_frequency_mhz;
        us_fence1 += prof.getStat(THREAD_FENCE_1) / gpu_frequency_mhz;
        us_fence2 += prof.getStat(THREAD_FENCE_2) / gpu_frequency_mhz;
        us_wait_host += prof.getStat(WAITING_ON_HOST) / gpu_frequency_mhz;
    }

    constexpr int FIELD_WIDTH {20};
    constexpr int FLOAT_PRECISION {2};

    printf("%*s%*s%*s%*s%*s\n",
           FIELD_WIDTH + 1, "Wait On Slot (us)",
           FIELD_WIDTH + 1, "Pack Queue (us)",
           FIELD_WIDTH + 1, "Fence 1 (us)",
           FIELD_WIDTH + 1, "Fence 2 (us)",
           FIELD_WIDTH + 1, "Wait Host (us)");

    printf("%*.*f %*.*f %*.*f %*.*f %*.*f\n\n",
           FIELD_WIDTH, FLOAT_PRECISION, ((double)us_wait_slot) / total,
           FIELD_WIDTH, FLOAT_PRECISION, ((double)us_pack) / total,
           FIELD_WIDTH, FLOAT_PRECISION, ((double)us_fence1) / total,
           FIELD_WIDTH, FLOAT_PRECISION, ((double)us_fence2) / total,
           FIELD_WIDTH, FLOAT_PRECISION, ((double)us_wait_host) / total);

    printf("PE %d: Queues %lu Threads %d\n",
           my_pe, num_wg, bp->num_threads);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
ROBackend::ro_net_free_runtime() {
    /*
     * Validate that a handle was passed that is not a nullptr.
     */
    auto *bp {backend_proxy.get()};
    assert(bp);

    /*
     * Set this flag to denote that the runtime is being torn down.
     */
    bp->done_flag = 1;

    /*
     * Tear down the worker threads.
     */
    for (auto &t : worker_threads) {
       t.join();
    }

    /*
     * Tear down the transport object.
     */
    while (!transport_.readyForFinalize()) {
        ;
    }
    transport_.finalizeTransport();

    /*
     * Free the profiler statistics structure.
     */
    //CHECK_HIP(hipFree(bp->profiler));

    /*
     * Tear down team_world
     */
    auto* team_world {team_tracker.get_team_world()};
    team_world->~Team();
    //CHECK_HIP(hipFree(team_world));

    /*
     * Free the gpu_handle.
     */
    //CHECK_HIP(hipHostFree(bp));

    return Status::ROC_SHMEM_SUCCESS;
}

bool
ROBackend::ro_net_process_queue(int queue_idx) {
    /*
     * Determine which indices to access in the queue.
     */
    auto *bp {backend_proxy.get()};
    queue_desc_t *queue_desc {&bp->queue_descs[queue_idx]};
    DPRINTF("Queue Desc read_idx %zu\n", queue_desc->read_idx);
    uint64_t read_slot {queue_desc->read_idx % bp->queue_size};

    /*
     * Check if next element from the device is ready.
     */
    const queue_element_t *next_element {nullptr};
    if (bp->gpu_queue) {
        /*
         * Flush HDP read cache so we can see updates to the GPU Queue
         * descriptor.
         */
        bp->hdp_policy->hdp_flush();

        /*
         * Copy the queue element from device memory to the host memory cache.
         */
        ::memcpy((void*)queue_element_cache_,
                 &bp->queues[queue_idx][read_slot],
                 sizeof(queue_element_t));

        /*
         * Set our local variable to the next element.
         */
        next_element = queue_element_cache_;
    } else {
        /*
         * Set our local variable to the next element.
         */
        next_element = &bp->queues[queue_idx][read_slot];
    }

    /*
     * By default, assume that no valid queue element exists. If a valid
     * queue element is found, we'll flip this return value to true.
     */
    bool valid {false};

    /*
     * Search for a single valid element and process that element through
     * the transport if one is found.
     */
    if (next_element->valid) {
        valid = true;

        DPRINTF("Rank %d Processing read_slot %lu of queue %d \n",
                my_pe, read_slot, queue_idx);

        /*
         * Pass the queue element to the transport.
         * TODO: Who is responsible for freeing this memory?
         */
        transport_.insertRequest(new queue_element_t(*next_element),
                                 queue_idx);

        /*
         * Toggle the queue flag back to invalid since the request was
         * just processed.
         */
        bp->queues[queue_idx][read_slot].valid = 0;

        /*
         * Update the CPU's local read index.
         */
        queue_desc->read_idx++;
    }

    return valid;
}

// TODO: change these int parameters into size_t
void
ROBackend::ro_net_poll(int thread_id, int num_threads) {
    /*
     * This access assumes only one device exists and should not work
     * generally.
     */
    int gpu_dev {-1};
    CHECK_HIP(hipGetDevice(&gpu_dev));

    auto *bp {backend_proxy.get()};

    /*
     * Continue until the runtime is torn down.
     */
    while (!bp->done_flag) {
        /*
         * Each worker thread is responsible for evaluating a queue index
         * at a time.
         */
        size_t tid = static_cast<size_t>(thread_id);
        for (size_t i {tid}; i < num_wg; i += num_threads) {
            /*
             * Drain up to 64 requests from this queue if they are ready.
             */
            int req_count {0};

            /*
             * This variable will evaluate to "True" as long as valid queue
             * entries are found and processed. The loop ends when this
             * evaluates to "False" or we evaluate 64 entries.
             */
            bool processed_req {false};
            do {
                processed_req = ro_net_process_queue(i);
                req_count++;
            } while (processed_req && (req_count < 64));
        }
    }
}

void
ROBackend::initIPC() {
    const auto& heap_bases {heap.get_heap_bases()};

    ipcImpl.ipcHostInit(transport_.getMyPe(),
                        heap_bases,
                        transport_.get_world_comm());
}

void
ROBackend::global_exit(int status) {
    transport_.global_exit(status);
}

}  // namespace rocshmem
