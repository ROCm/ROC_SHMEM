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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <endian.h>

#include <roc_shmem.hpp>
#include <mpi.h>

#include "dynamic_connection.hpp"
#include "reliable_connection.hpp"
#include "context.hpp"
#include "backend.hpp"
#include "queue_pair.hpp"
#include "wg_state.hpp"

roc_shmem_status_t
GPUIBBackend::net_free(void *ptr)
{
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::dump_backend_stats()
{
    /*
     * TODO: Refactor this into the Stats class to remove the ifdef.
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

    uint64_t total_rtn_quiet_count = 0;
    uint64_t total_rtn_db_count = 0;
    uint64_t total_rtn_wqe_count = 0;

    for (int i = 0; i < statblocks; i++) {
        cycles_ring_sq_db += gpu_qps[i].profiler.getStat(RING_SQ_DB);
        cycles_update_wqe += gpu_qps[i].profiler.getStat(UPDATE_WQE);
        cycles_poll_cq += gpu_qps[i].profiler.getStat(POLL_CQ);
        cycles_next_cq += gpu_qps[i].profiler.getStat(NEXT_CQ);
        total_rtn_quiet_count += gpu_qps[i].profiler.getStat(RTN_QUIET_COUNT);
        total_rtn_db_count += gpu_qps[i].profiler.getStat(RTN_DB_COUNT);
        total_rtn_wqe_count += gpu_qps[i].profiler.getStat(RTN_WQE_COUNT);
    }

    double us_ring_sq_db = (double) cycles_ring_sq_db / gpu_clock_freq_mhz;
    double us_update_wqe = (double) cycles_update_wqe / gpu_clock_freq_mhz;
    double us_poll_cq = (double) cycles_poll_cq / gpu_clock_freq_mhz;
    double us_next_cq = (double) cycles_next_cq / gpu_clock_freq_mhz;
    double us_init = (double) cycles_init / gpu_clock_freq_mhz;
    double us_finalize = (double) cycles_finalize / gpu_clock_freq_mhz;

    const int FIELD_WIDTH = 20;
    const int FLOAT_PRECISION = 2;

    printf("RTN Counts: Internal Quiets %lu DB Rings %lu WQE Posts "
           "%lu\n", total_rtn_quiet_count, total_rtn_db_count,
           total_rtn_wqe_count);

    printf("\n%*s%*s%*s%*s%*s%*s\n", FIELD_WIDTH + 1, "Init (us)",
           FIELD_WIDTH + 1, "Finalize (us)",
           FIELD_WIDTH + 1, "Ring SQ DB (us)",
           FIELD_WIDTH + 1, "Update WQE (us)",
           FIELD_WIDTH + 1, "Poll CQ (us)",
           FIELD_WIDTH + 1, "Next CQ (us)");

    uint64_t totalFinalize = globalStats.getStat(NUM_FINALIZE);
    printf("%*.*f %*.*f %*.*f %*.*f %*.*f %*.*f\n",
           FIELD_WIDTH, FLOAT_PRECISION, us_init / totalFinalize,
           FIELD_WIDTH, FLOAT_PRECISION, us_finalize / totalFinalize,
           FIELD_WIDTH, FLOAT_PRECISION, us_ring_sq_db / total_rtn_db_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_update_wqe / total_rtn_wqe_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_poll_cq / total_rtn_quiet_count,
           FIELD_WIDTH, FLOAT_PRECISION, us_next_cq / total_rtn_quiet_count);
#endif
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::reset_backend_stats()
{
    int statblocks = connection->total_number_connections();

    for (int i = 0; i < statblocks; i++)
        gpu_qps[i].profiler.resetStats();

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::exchange_hdp_info()
{
    CHECK_HIP(hipMalloc((void**) &hdp_policy, sizeof(HdpPolicy)));
    new (hdp_policy) HdpPolicy();

    ibv_mr *hdp_mr = nullptr;
    roc_shmem_status_t status;
    status = connection->reg_mr(hdp_policy->get_hdp_flush_addr(), 32, &hdp_mr);
    if (status != ROC_SHMEM_SUCCESS) {
        return status;
    }

    // exchange the hdp info for a fence implementation
    CHECK_HIP(hipMalloc((void**)&hdp_rkey,
                        sizeof(uint32_t) * num_pes));

    CHECK_HIP(hipMalloc((void**)&hdp_address,
                         sizeof(uintptr_t) * num_pes));

    uint32_t *host_hdp_cpy =
        (uint32_t*) malloc(sizeof(uint32_t) * num_pes);
    if (host_hdp_cpy == nullptr) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    uint32_t **host_hdp_address_cpy =
        (uint32_t**) malloc(sizeof(uint32_t*) * num_pes);
    if (host_hdp_address_cpy == nullptr) {
        free(host_hdp_cpy);
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    int my_rank = my_pe;
    host_hdp_cpy[my_rank] = htobe32(hdp_mr->rkey);
    host_hdp_address_cpy[my_rank] = hdp_policy->get_hdp_flush_addr();

    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  host_hdp_cpy,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(uintptr_t),
                  MPI_CHAR,
                  host_hdp_address_cpy,
                  sizeof(uint32_t *),
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    CHECK_HIP(hipMemcpy(hdp_rkey,
                        host_hdp_cpy,
                        sizeof(uint32_t) * num_pes,
                        hipMemcpyHostToDevice));

    CHECK_HIP(hipMemcpy(hdp_address,
                        host_hdp_address_cpy,
                        sizeof(uint32_t *) * num_pes,
                        hipMemcpyHostToDevice));

    free(host_hdp_cpy);
    free(host_hdp_address_cpy);

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::setup_atomic_region()
{
    CHECK_HIP(hipMalloc((void**)&atomic_ret, sizeof(rtn_atomic_ret_t)));

    CHECK_HIP(hipExtMallocWithFlags((void**)&atomic_ret->atomic_base_ptr,
                                    sizeof(uint64_t) * max_nb_atomic * num_wg,
                                    hipDeviceMallocFinegrained));

    memset(atomic_ret->atomic_base_ptr, 0,
           sizeof(uint64_t) * max_nb_atomic * num_wg);

    struct ibv_mr *mr = nullptr;
    roc_shmem_status_t status;
    status = connection->reg_mr(atomic_ret->atomic_base_ptr,
                                sizeof(uint64_t) * max_nb_atomic * num_wg,
                                &mr);

    if (status != ROC_SHMEM_SUCCESS) {
        return status;
    }

    atomic_ret->atomic_lkey = htobe32(mr->lkey);

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::allocate_heap_memory()
{
    // allocate and register heap memory
    uint64_t *host_bases_cpy =
        (uint64_t*) malloc(sizeof(uint64_t*) * num_pes);
    if (host_bases_cpy == nullptr) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    uint32_t *host_rkey_cpy =
        (uint32_t*) malloc(sizeof(uint32_t) * num_pes);
    if (host_rkey_cpy == nullptr) {
        free(host_bases_cpy);
        return ROC_SHMEM_UNKNOWN_ERROR;
    }

    CHECK_HIP(hipMalloc(&heap_bases, sizeof(char*) * num_pes));

    void *base_heap;
    CHECK_HIP(hipExtMallocWithFlags(&base_heap,
                                    heap_size,
                                    hipDeviceMallocFinegrained));
    roc_shmem_status_t status;
    status = connection->reg_mr(base_heap, heap_size, &heap_mr);
    if (status != ROC_SHMEM_SUCCESS) {
        free(host_bases_cpy);
        free(host_rkey_cpy);
        return status;
    }

    // TODO: I'm not sure if this is needed)
    connection->initialize_rkey_handle(&heap_rkey, heap_mr);

    heap_bases[my_pe] = (char*) base_heap;

    CHECK_HIP(hipMemcpy(host_bases_cpy,
                        heap_bases,
                        sizeof(uint64_t*) * num_pes,
                        hipMemcpyDeviceToHost));

    CHECK_HIP(hipMemcpy(host_rkey_cpy,
                        heap_rkey,
                        sizeof(uint32_t*) * num_pes,
                        hipMemcpyDeviceToHost));

    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(uint64_t),
                  MPI_CHAR,
                  host_bases_cpy,
                  sizeof(uint64_t),
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  host_rkey_cpy,
                  sizeof(uint32_t),
                  MPI_CHAR,
                  MPI_COMM_WORLD);

    CHECK_HIP(hipMemcpy(heap_bases,
                        host_bases_cpy,
                        sizeof(uint64_t*) * num_pes,
                        hipMemcpyHostToDevice));

    CHECK_HIP(hipMemcpy(heap_rkey,
                        host_rkey_cpy,
                        sizeof(uint32_t*) * num_pes,
                        hipMemcpyHostToDevice));

    free(host_bases_cpy);
    free(host_rkey_cpy);

    current_heap_offset = 0;
    lkey = heap_mr->lkey;

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::initialize_ipc()
{
    ipcImpl.ipcHostInit(my_pe, heap_bases);
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::setup_gpu_qps()
{
    int connections;
    connection->get_remote_conn(connections);
    connections *= num_wg;

    CHECK_HIP(hipMalloc(&gpu_qps, sizeof(QueuePair) * connections));

    for (int i = 0; i < connections; i++) {
        new (&gpu_qps[i]) QueuePair(this);

        roc_shmem_status_t status;
        status = connection->init_gpu_qp_from_connection(gpu_qps[i], i);

        if (status != ROC_SHMEM_SUCCESS) {
            return status;
        }
    }

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::setup_default_ctx()
{

    Context *ctx;
    CHECK_HIP(hipMalloc(&ctx, sizeof(GPUIBContext)));
    new (ctx) GPUIBContext(*this, 0);

    hipMemcpyToSymbol(HIP_SYMBOL(SHMEM_CTX_DEFAULT), &ctx,
                      sizeof(ctx), 0, hipMemcpyHostToDevice);

    return ROC_SHMEM_SUCCESS;
}

GPUIBBackend::GPUIBBackend(unsigned num_wgs)
    : Backend(num_wgs)
{
    char * value;
    if ((value = getenv("ROC_SHMEM_HEAP_SIZE"))) {
        heap_size = atoi(value);
    }

    roc_shmem_status_t status;

#ifdef USE_DC
    connection = new DynamicConnection(this);
#else
    connection = new ReliableConnection(this);
#endif

    status = connection->init_mpi_once();
    assert(status == ROC_SHMEM_SUCCESS);

    type = GPU_IB_BACKEND;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe);

    status = connection->initialize(num_wg);
    assert(status == ROC_SHMEM_SUCCESS);

    status = exchange_hdp_info();
    assert(status == ROC_SHMEM_SUCCESS);

    status = allocate_heap_memory();
    assert(status == ROC_SHMEM_SUCCESS);

    status = initialize_ipc();
    assert(status == ROC_SHMEM_SUCCESS);

    status = setup_atomic_region();
    assert(status == ROC_SHMEM_SUCCESS);

    status = connection->initialize_gpu_policy(&connection_policy, heap_rkey);
    assert(status == ROC_SHMEM_SUCCESS);

    roc_shmem_collective_init();
    roc_shmem_g_init();

    connection->post_wqes();

    status = setup_gpu_qps();
    assert(status == ROC_SHMEM_SUCCESS);

    status = setup_default_ctx();
    assert(status == ROC_SHMEM_SUCCESS);
}

void
GPUIBBackend::roc_shmem_collective_init()
{
    int64_t *ptr = (int64_t*) heap_bases[my_pe] +
        current_heap_offset;

    current_heap_offset += SHMEM_BARRIER_SYNC_SIZE;

    barrier_sync = ptr;

    for (int i = 0; i < num_pes; i++) {
        barrier_sync[i] = SHMEM_SYNC_VALUE;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void
GPUIBBackend::roc_shmem_g_init()
{
    char *ptr = (char*) heap_bases[my_pe] +
        current_heap_offset;

    current_heap_offset = current_heap_offset +
        sizeof(int64_t)* MAX_WG_SIZE * num_wg;

    g_ret = (char*) ptr;

    MPI_Barrier(MPI_COMM_WORLD);
}

GPUIBBackend::~GPUIBBackend()
{
    auto status = connection->finalize();
    if (status) {
        delete connection;
    }
    hdp_policy->~HdpPolicy();
    CHECK_HIP(hipFree(hdp_policy));
    CHECK_HIP(hipFree(atomic_ret));
    CHECK_HIP(hipFree(connection_policy));
}

roc_shmem_status_t
GPUIBBackend::net_malloc(void **ptr, size_t size)
{
    *ptr = (char*)  heap_bases[my_pe] + current_heap_offset;

    current_heap_offset = current_heap_offset + (size / sizeof(char));

    MPI_Barrier(MPI_COMM_WORLD);
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
GPUIBBackend::dynamic_shared(size_t *shared_bytes)
{
    uint32_t heap_usage = sizeof(uint64_t) * num_pes;
    uint32_t rtn_usage = 0;
    uint32_t ipc_usage = 0;

    int remote_conn;
    connection->get_remote_conn(remote_conn);
    rtn_usage = sizeof(QueuePair) * remote_conn;
    ipc_usage = ipcImpl.ipcDynamicShared();

    *shared_bytes = heap_usage + rtn_usage + ipc_usage + sizeof(GPUIBContext)
        + sizeof(WGState);

    return ROC_SHMEM_SUCCESS;
}
