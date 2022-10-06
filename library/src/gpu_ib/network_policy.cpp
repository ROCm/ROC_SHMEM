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

#include "network_policy.hpp"

#include <mpi.h>

#include "../atomic_return.hpp"
#include "backend_ib.hpp"
#include "config.h"  // NOLINT(build/include_subdir)
#include "connection.hpp"
#include "context_incl.hpp"
#include "dynamic_connection.hpp"
#include "queue_pair.hpp"
#include "reliable_connection.hpp"
#include "wg_state.hpp"

namespace rocshmem {

Status
NetworkOnImpl::dump_backend_stats(ROCStats *globalStats) {
    /*
     * TODO(bpotter): Refactor this into the Stats class to remove the ifdef
     */
#ifdef PROFILE
    int statblocks = connection->total_number_connections();

    uint64_t cycles_ring_sq_db = 0;
    uint64_t cycles_update_wqe = 0;
    uint64_t cycles_poll_cq = 0;
    uint64_t cycles_next_cq = 0;
    uint64_t cycles_init = gpu_qps[statblocks - 1].profiler.getStat(INIT);
    uint64_t cycles_finalize =
        gpu_qps[statblocks - 1].profiler.getStat(FINALIZE);

    uint64_t total_quiet_count = 0;
    uint64_t total_db_count = 0;
    uint64_t total_wqe_count = 0;

    for (int i = 0; i < statblocks; i++) {
        cycles_ring_sq_db += gpu_qps[i].profiler.getStat(RING_SQ_DB);
        cycles_update_wqe += gpu_qps[i].profiler.getStat(UPDATE_WQE);
        cycles_poll_cq += gpu_qps[i].profiler.getStat(POLL_CQ);
        cycles_next_cq += gpu_qps[i].profiler.getStat(NEXT_CQ);
        total_quiet_count += gpu_qps[i].profiler.getStat(QUIET_COUNT);
        total_db_count += gpu_qps[i].profiler.getStat(DB_COUNT);
        total_wqe_count += gpu_qps[i].profiler.getStat(WQE_COUNT);
    }

    double us_ring_sq_db = cycles_ring_sq_db / gpu_clock_freq_mhz;
    double us_update_wqe = cycles_update_wqe / gpu_clock_freq_mhz;
    double us_poll_cq = cycles_poll_cq / gpu_clock_freq_mhz;
    double us_next_cq = cycles_next_cq / gpu_clock_freq_mhz;
    double us_init = cycles_init / gpu_clock_freq_mhz;
    double us_finalize = cycles_finalize / gpu_clock_freq_mhz;

    const int FIELD_WIDTH = 20;
    const int FLOAT_PRECISION = 2;

    printf("Counts: Internal Quiets %lu DB Rings %lu WQE Posts %lu\n",
           total_quiet_count,
           total_db_count,
           total_wqe_count);

    printf("\n%*s%*s%*s%*s%*s%*s\n",
           FIELD_WIDTH + 1, "Init (us)",
           FIELD_WIDTH + 1, "Finalize (us)",
           FIELD_WIDTH + 1, "Ring SQ DB (us)",
           FIELD_WIDTH + 1, "Update WQE (us)",
           FIELD_WIDTH + 1, "Poll CQ (us)",
           FIELD_WIDTH + 1, "Next CQ (us)");

    uint64_t totalFinalize = globalStats->getStat(NUM_FINALIZE);
    printf("%*.*f %*.*f %*.*f %*.*f %*.*f %*.*f\n",
           FIELD_WIDTH, FLOAT_PRECISION, us_init / totalFinalize,
           FIELD_WIDTH, FLOAT_PRECISION, us_finalize / totalFinalize,
           FIELD_WIDTH, FLOAT_PRECISION, us_ring_sq_db / total_db_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_update_wqe / total_wqe_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_poll_cq / total_quiet_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_next_cq / total_quiet_count);
#endif
    return Status::ROC_SHMEM_SUCCESS;
}

Status
NetworkOnImpl::reset_backend_stats() {
    int statblocks = connection->total_number_connections();

    for (size_t i = 0; i < statblocks; i++) {
        gpu_qps[i].profiler.resetStats();
    }

    return Status::ROC_SHMEM_SUCCESS;
}

Status
NetworkOnImpl::exchange_hdp_info(HdpPolicy *hdp_policy,
                                 MPI_Comm thread_comm) {
    /*
     * Using Connection class, register the host-side hdp flush address
     * with the InfiniBand network.
     */
    Status status;

// TODO(khamidou): commenting this out until ROCm4.5 fixed the issue
    status = connection->reg_mr(hdp_policy->get_hdp_flush_ptr(),
                                32,
                                &hdp_mr, false);

    if (status != Status::ROC_SHMEM_SUCCESS) {
        return status;
    }


    /*
     * Allocate device-side memory for the remote HDP keys.
     */
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&hdp_rkey),
                        num_pes * sizeof(uint32_t)));

    /*
     * Allocate device-side memory for the remote HDP addresses.
     */
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&hdp_address),
                        num_pes * sizeof(uintptr_t)));

    /*
     * Allocate host-side memory to exchange hdp keys using MPI_Allgather.
     */
    uint32_t *host_hdp_cpy =
        reinterpret_cast<uint32_t*>(malloc(num_pes * sizeof(uint32_t)));
    if (host_hdp_cpy == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    /*
     * Allocate host-side memory to exchange hdp addresses using
     * MPI_Allgather.
     */
    uint32_t **host_hdp_address_cpy =
        reinterpret_cast<uint32_t**>(malloc(num_pes * sizeof(uint32_t*)));
    if (host_hdp_address_cpy == nullptr) {
        free(host_hdp_cpy);
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    /*
     * This processing element writes its personal HDP key and HDP address
     * into the host-side arrays which were just allocated.
     */
    int my_rank = my_pe;
    host_hdp_cpy[my_rank] = htobe32(hdp_mr->rkey);
    host_hdp_address_cpy[my_rank] = hdp_policy->get_hdp_flush_ptr();

    /*
     * Do all-to-all exchange of our HDP key with other processing elements.
     */
    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  host_hdp_cpy,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  thread_comm);

    /*
     * Do all-to-all exchange of our HDP address with other processing
     * elements.
     */
    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(uintptr_t),
                  MPI_CHAR,
                  host_hdp_address_cpy,
                  sizeof(uint32_t *),
                  MPI_CHAR,
                  thread_comm);

    /*
     * Copy the recently exchanged HDP keys to device memory.
     */
    hipStream_t stream;
    CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CHECK_HIP(hipMemcpyAsync(hdp_rkey,
                             host_hdp_cpy,
                             num_pes * sizeof(uint32_t),
                             hipMemcpyHostToDevice,
                             stream));

    /*
     * Copy the recently exchanged HDP addresses to device memory.
     */
    CHECK_HIP(hipMemcpyAsync(hdp_address,
                             host_hdp_address_cpy,
                             num_pes * sizeof(uint32_t *),
                             hipMemcpyHostToDevice,
                             stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipStreamDestroy(stream));

    /*
     * Free the host-side resources used to exchange HDP resources
     * between processing elements.
     */
    free(host_hdp_cpy);
    free(host_hdp_address_cpy);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
NetworkOnImpl::setup_atomic_region() {

    /*
     * Allocate fine-grained device-side memory for the atomic return
     * region.
     */
    allocate_atomic_region(&atomic_ret, num_wg);

    /*
     * Register the atomic return region on the InfiniBand network.
     */
    Status status;
    status = connection->reg_mr(atomic_ret->atomic_base_ptr,
                                sizeof(uint64_t) * max_nb_atomic * num_wg,
                                &mr, false);

    if (status != Status::ROC_SHMEM_SUCCESS) {
        return status;
    }

    /*
     * Set member variable from class.
     */
    atomic_ret->atomic_lkey = htobe32(mr->lkey);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
NetworkOnImpl::heap_memory_rkey(char *local_heap_base,
                                size_t heap_size,
                                MPI_Comm thread_comm,
                                bool is_managed) {
    /*
     * Allocate host-side memory to hold remote keys for all processing
     * elements.
     */
    const size_t rkeys_size = sizeof(uint32_t) * num_pes;
    uint32_t *host_rkey_cpy =
        reinterpret_cast<uint32_t*>(malloc(rkeys_size));
    if (host_rkey_cpy == nullptr) {
        return Status::ROC_SHMEM_UNKNOWN_ERROR;
    }

    /*
     * Using the Connection class, register the symmetric heap with the
     * InfiniBand network.
     */
    void *base_heap = local_heap_base;
    Status status = connection->reg_mr(base_heap,
                                       heap_size,
                                       &heap_mr,
                                       is_managed);
    if (status != Status::ROC_SHMEM_SUCCESS) {
        free(host_rkey_cpy);
        printf("Error failed to register the heap\n");
        return status;
    }

    /*
     * Using the memory region from the prior heap memory registration,
     * allocate and initialize some device-side memory to hold the remote
     * keys for the symmetric heap base.
     *
     * Only the device-side memory entry for this processing element will be
     * updated with the key for the heap memory region.
     */
    connection->initialize_rkey_handle(&heap_rkey, heap_mr);

    /*
     * Copy the device-side heap base remote key array to the host-side
     * heap base remote key array.
     */
    hipStream_t stream;
    CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CHECK_HIP(hipMemcpyAsync(host_rkey_cpy,
                             heap_rkey,
                             rkeys_size,
                             hipMemcpyDeviceToHost,
                             stream));
    CHECK_HIP(hipStreamSynchronize(stream));

    /*
     * Do all-to-all exchange of symmetric heap base remote key between the
     * processing elements.
     */
    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  host_rkey_cpy,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  thread_comm);

    /*
     * Copy the recently updated host-side heap base remote key array back
     * to the device-side memory.
     */
    CHECK_HIP(hipMemcpyAsync(heap_rkey,
                             host_rkey_cpy,
                             rkeys_size,
                             hipMemcpyHostToDevice,
                             stream));
    CHECK_HIP(hipStreamSynchronize(stream));
    CHECK_HIP(hipStreamDestroy(stream));

    /*
     * Free the host-side resources used to do the processing element
     * exchange of keys and addresses for the symmetric heap base.
     */
    free(host_rkey_cpy);

    /*
     * Initialize this member variable to hold the InfiniBand memory
     * region's local key.
     */
    lkey = heap_mr->lkey;

    return Status::ROC_SHMEM_SUCCESS;
}

Status
NetworkOnImpl::setup_gpu_qps(GPUIBBackend *B) {
    /*
     * Determine how many connections are needed.
     * The number of connections depends on the connection type and the
     * number of workgroups.
     */
    int connections;
    connection->get_remote_conn(&connections);
    connections *= num_wg;

    /*
     * Allocate device-side memory for the queue pairs.
     */
    CHECK_HIP(hipMalloc(&gpu_qps, sizeof(QueuePair) * connections));

    /*
     * For every connection, initialize the QueuePair.
     */
    for (int i = 0; i < connections; i++) {
        new (&gpu_qps[i]) QueuePair(B);
        Status status;
        status = connection->init_gpu_qp_from_connection(&gpu_qps[i], i);

        if (status != Status::ROC_SHMEM_SUCCESS) {
            return status;
        }
    }

    return Status::ROC_SHMEM_SUCCESS;
}

void
NetworkOnImpl::roc_shmem_g_init(SymmetricHeap* heap_handle,
                                MPI_Comm thread_comm) {
    init_g_ret(heap_handle,
               thread_comm,
               num_wg,
               &g_ret);
}

__host__ void
NetworkOnImpl::networkHostSetup(GPUIBBackend *B) {
    num_pes = B->num_pes;
    my_pe = B->my_pe;
    num_wg = B->num_wg;

#ifdef USE_DC
    connection = new DynamicConnection(B);
#else
    connection = new ReliableConnection(B);
#endif

    Status status = connection->initialize(B->num_wg);
    assert(status == Status::ROC_SHMEM_SUCCESS);

    status = exchange_hdp_info(B->hdp_policy,
                               B->thread_comm);
    assert(status == Status::ROC_SHMEM_SUCCESS);

    const auto& heap_bases {B->heap.get_heap_bases()};
    status = heap_memory_rkey(heap_bases[my_pe],
                              B->heap.get_size(),
                              B->thread_comm,
                              B->heap.is_managed());
    assert(status == Status::ROC_SHMEM_SUCCESS);
    // The earliest we can allow the main thread to launch a kernel to
    // avoid potential deadlock
    network_init_done = true;

    status = setup_atomic_region();
    assert(status == Status::ROC_SHMEM_SUCCESS);

    status = connection->initialize_gpu_policy(&connection_policy,
                                               heap_rkey);
    assert(status == Status::ROC_SHMEM_SUCCESS);

    roc_shmem_g_init(&B->heap,
                     B->thread_comm);

    connection->post_wqes();

    status = setup_gpu_qps(B);
    assert(status == Status::ROC_SHMEM_SUCCESS);
}

__host__ void
NetworkOnImpl::networkHostFinalize() {
    CHECK_HIP(hipFree(hdp_rkey));
    hdp_rkey = nullptr;

    CHECK_HIP(hipFree(hdp_address));
    hdp_address = nullptr;

    CHECK_HIP(hipFree(atomic_ret));
    atomic_ret = nullptr;

    CHECK_HIP(hipFree(gpu_qps));
    gpu_qps = nullptr;

    CHECK_HIP(hipFree(connection_policy));
    connection_policy = nullptr;

    connection->free_rkey_handle(heap_rkey);

    auto status = connection->finalize();
    if (status == Status::ROC_SHMEM_SUCCESS) {
        delete connection;
        connection = nullptr;
    }
}
__host__ uint32_t
network_get_DynamicShared(int num_pes) {
#ifndef USE_SINGLE_NODE
    int remote_conn = 1;
#ifndef USE_DC
    remote_conn = num_pes;
#endif
    return remote_conn * sizeof(QueuePair);
#else
    return 0;
#endif
}

__host__ void
NetworkOnImpl::networkHostInit(GPUIBContext *ctx,
                               int buffer_id) {
    int remote_conn = getNumQueuePairs();

    CHECK_HIP(hipMalloc(&ctx->device_qp_proxy,
                        remote_conn * sizeof(QueuePair)));

    for (int i = 0; i < getNumQueuePairs(); i++) {
        /*
         * RC gpu_qp is actually [NUM_PE][NUM_WG] qps but is flattend.
         * Each num_pe entry contains num_wg QPs connected to that PE.
         * For RC, we need to iterate gpu_qp[i][buffer_id] to collect a
         * single QP for each connected PE in order to build context.
         * For DC, NUM_PE = 1 so can just use buffer_id directly.
         */
        int offset = num_wg * i + buffer_id;
        new (ctx->getQueuePair(i)) QueuePair(gpu_qps[offset]);

        auto *qp = ctx->getQueuePair(i);
        qp->global_qp = &gpu_qps[offset];
        qp->num_cqs = getNumQueuePairs();
        qp->atomic_ret.atomic_base_ptr =
            &atomic_ret->atomic_base_ptr[max_nb_atomic * buffer_id];
        qp->base_heap = ctx->base_heap;
    }
    ctx->g_ret = g_ret;
}

__device__ void
NetworkOnImpl::networkGpuInit(GPUIBContext *ctx,
                              int buffer_id) {
    for (int i = 0; i < getNumQueuePairs(); i++) {
        int offset = num_wg * i + buffer_id;

        auto *qp = ctx->getQueuePair(i);
        new (qp) QueuePair(gpu_qps[offset]);

        qp->global_qp = &gpu_qps[offset];
        qp->num_cqs = getNumQueuePairs();
        qp->atomic_ret.atomic_base_ptr =
                &atomic_ret->atomic_base_ptr[max_nb_atomic * buffer_id];
        qp->base_heap = ctx->base_heap;
    }
    ctx->g_ret = g_ret;
}

__device__ __host__ QueuePair*
NetworkOnImpl::getQueuePair(QueuePair *qp_handle,
                            int pe) {
#ifdef USE_DC
    return qp_handle;
#else
    return &qp_handle[pe];
#endif
}

__device__ __host__ int
NetworkOnImpl::getNumQueuePairs() {
#ifdef USE_DC
    return 1;
#else
    return num_pes;
#endif
}

}  // namespace rocshmem
