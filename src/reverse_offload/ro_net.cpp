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
#include <smmintrin.h>
#include <immintrin.h>
#include <thread>

#include <roc_shmem.hpp>
#include "ro_net_internal.hpp"
#include "backend.hpp"
#include "transport.hpp"

/***
 *
 * External Host-side API functions
 *
 ***/
roc_shmem_status_t
ROBackend::finalize()
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) backend_handle;

    ro_net_free_runtime(ro_net_gpu_handle);

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
ROBackend::dynamic_shared(size_t *shared_bytes)
{
    *shared_bytes = sizeof(ROContext) + sizeof(ro_net_wg_handle);
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
ROBackend::pre_init()
{
    struct ro_net_handle *ro_net_gpu_handle;
    hipHostMalloc_assert((void**) &ro_net_gpu_handle,
                         sizeof(struct ro_net_handle *));

    memset(ro_net_gpu_handle, 0, sizeof(ro_net_handle));

    backend_handle = ro_net_gpu_handle;
    type = RO_BACKEND;

    #ifdef MPI_TRANSPORT
    transport = new MPITransport();
    #endif

    #ifdef OPENSHMEM_TRANSPORT
    transport = new OpenSHMEMTransport();
    #endif

    // DGPU:
    // Need to allocate hdp flush registers.  We will go ahead and map all
    // the devices on the node.
    #if defined GPU_HEAP || defined GPU_QUEUE
    ro_net_gpu_handle->hdp_regs = hdp_map_all();
    #endif

    if (!transport) {
        ro_net_free_runtime(ro_net_gpu_handle);
        return ROC_SHMEM_OOM_ERROR;
    }

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
ROBackend::init(int num_queues)
{
    int num_threads = 1;

    ro_net_handle *ro_net_gpu_handle =
        reinterpret_cast<ro_net_handle*>(backend_handle);

    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess)
        return ROC_SHMEM_UNKNOWN_ERROR;

    if (count == 0) {
        std::cerr << "No GPU found!" << std::endl;
        exit(-1);
    }

    if (count > 1) {
        std::cerr << "More than one GPU on this node.  RO_NET currently only "
            << "supports one GPU per node and will use device 0" << std::endl;
    }

    // Take advantage of the fact that only so many WGs can be scheduled on
    // the HW to limit the number of queues.  For now, I'm being ultra
    // conservative and allocating for worst case.
    int num_cus;
    if (hipDeviceGetAttribute(&num_cus,
        hipDeviceAttributeMultiprocessorCount, 0)) {
        return ROC_SHMEM_UNKNOWN_ERROR;
    }
    // Even though we can only have 32 WGs on a CU max, we reserve queues
    // for 40 because we statically bind them to WV slots, and we don't
    // know which of the 40 slots will be used by the 32 WGs.
    int max_num_queues = num_cus * 40;
    if (num_queues > max_num_queues) {
        std::cerr << "User requested more queues than can concurrently be "
            << "used by the hardware.  Overriding to " << max_num_queues
            << " queues." << std::endl;
        num_queues = max_num_queues;
    } else if (!num_queues) {
        num_queues = max_num_queues;
    }

    if (num_threads > 0 &&
        ((num_queues < num_threads) || ((num_queues % num_threads) != 0))) {
        return ROC_SHMEM_INVALID_ARGUMENTS;
    }

    roc_shmem_status_t return_code;

    char *value;
    ro_net_gpu_handle->queue_size = DEFAULT_QUEUE_SIZE;
    if ((value = getenv("RO_NET_QUEUE_SIZE")) != NULL) {
        ro_net_gpu_handle->queue_size = atoi(value);
        assert(ro_net_gpu_handle->queue_size != 0);
    }

    posix_memalign((void**)&elt, 64, sizeof(queue_element_t));
    if (!elt) {
        net_free(ro_net_gpu_handle);
        return ROC_SHMEM_OOM_ERROR;
    }

    // allocate the resources for internal barriers
    unsigned int *barrier_ptr;
    hipMalloc_assert((void**) &barrier_ptr, sizeof(unsigned int ));
    *barrier_ptr=0;

    ROStats *profiler_ptr;
    hipMalloc_assert((void**) &profiler_ptr, sizeof(ROStats) * num_queues);
    new (profiler_ptr) ROStats();

    ro_net_gpu_handle->num_threads = num_threads;
    ro_net_gpu_handle->num_queues = num_queues;
    ro_net_gpu_handle->done_flag = 0;
    num_pes = transport->getNumPes();
    my_pe = transport->getMyPe();
    ro_net_gpu_handle->barrier_ptr = barrier_ptr;
    ro_net_gpu_handle->profiler = profiler_ptr;

    if ((return_code = transport->initTransport(num_queues, ro_net_gpu_handle)) !=
        ROC_SHMEM_SUCCESS) {
        net_free(ro_net_gpu_handle);
        return return_code;
    }

    queue_element_t **queues;
    hipHostMalloc_assert((void***)&queues,
                         sizeof(queue_element_t*) * num_queues);

    ro_net_gpu_handle->queues = queues;

    unsigned int *queue_tokens;
    hipMalloc_assert((void**) &queue_tokens,
                     sizeof(unsigned int) * num_queues);
    ro_net_gpu_handle->queueTokens = queue_tokens;

    for (int i = 0; i < num_queues; i++) {
        ro_net_gpu_handle->queueTokens[i] = 1;
    }

    queue_desc_t *queue_descs;
    #ifdef GPU_QUEUE
    ro_net_device_uc_malloc((void**) &queue_descs,
                              sizeof(queue_desc_t) * num_queues);
    #else
    hipHostMalloc_assert((void**) &queue_descs,
                         sizeof(queue_desc_t) * num_queues);
    #endif
    ro_net_gpu_handle->queue_descs = queue_descs;

    // Allocate circular buffer space for all queues.  Do all queues in a
    // single allocation since HIP currently doesn't handle a large number of
    // small allocations particularly well.
    #ifdef GPU_QUEUE
    ro_net_device_uc_malloc((void **) queues, num_queues *
                            sizeof(queue_element) *
                            ro_net_gpu_handle->queue_size);
    #else
    hipHostMalloc_assert(queues,
                         num_queues * sizeof(queue_element_t) *
                         ro_net_gpu_handle->queue_size);
    #endif
    memset(*queues, 0, num_queues * sizeof(queue_element_t) *
           ro_net_gpu_handle->queue_size);

    // Initialize queue descriptors
    for (int i = 0; i < num_queues; i++) {
        queues[i] = (*queues) + ro_net_gpu_handle->queue_size * i;
        queue_descs[i].read_idx = 0;
        queue_descs[i].write_idx = 0;
        // There is a status variable for each work-item in a work-group.  We
        // just overallocate for the maximum work-group size.
        int max_wg_size, gpu_dev;
        hipGetDevice_assert(&gpu_dev);
        hipDeviceGetAttribute(&max_wg_size,
            hipDeviceAttributeMaxThreadsPerBlock, gpu_dev);
        // Status always goes in dGPU memory to prevent polling for completion
        // over PCIe
        ro_net_device_uc_malloc((void**) &queue_descs[i].status,
                                  max_wg_size * sizeof(char));
    }

    // Spawn threads to service the queues.
    for (int i = 0; i < num_threads; i++) {
        worker_threads.emplace_back(&ROBackend::ro_net_poll, this, i,
                                    num_threads);
    }
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
ROBackend::net_malloc(void **ptr, size_t size)
{
    transport->allocateMemory(ptr, size);
    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
ROBackend::net_free(void * ptr)
{
    return transport->deallocateMemory(ptr);
}

roc_shmem_status_t
ROBackend::reset_backend_stats()
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) backend_handle;

    for (int i = 0; i < ro_net_gpu_handle->num_queues; i++)
        ro_net_gpu_handle->profiler[i].resetStats();

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
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
    for (int i = 0; i < ro_net_gpu_handle->num_queues; i++) {
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
            my_pe, ro_net_gpu_handle->num_queues,
            ro_net_gpu_handle->num_threads);

    return ROC_SHMEM_SUCCESS;
}

roc_shmem_status_t
ROBackend::ro_net_free_runtime(struct ro_net_handle * ro_net_gpu_handle)
{
    assert(ro_net_gpu_handle);

    ro_net_gpu_handle->done_flag = 1;
    for (auto &t : worker_threads) {
       t.join();
    }

    if (transport) {
        while(!transport->readyForFinalize());
        transport->finalizeTransport();
        // TODO: For some reason this always seg faults.  I have no idea why.
        // Ignoring for now since its during tear-down anyway.
        delete transport;
    }

    #if defined GPU_HEAP || defined GPU_QUEUE
    hipHostFree((void*) ro_net_gpu_handle->hdp_regs);
    #endif

    if (elt)
        free(elt);

    hipFree_assert(ro_net_gpu_handle->profiler);
    hipFree_assert(ro_net_gpu_handle->barrier_ptr);
    hipFree_assert(ro_net_gpu_handle->queueTokens);
    #ifdef GPU_QUEUE
    hipFree_assert(ro_net_gpu_handle->queue_descs);
    #else
    hipHostFree_assert(ro_net_gpu_handle->queue_descs);
    #endif

    for (int i = 0; i < ro_net_gpu_handle->num_queues; i++) {
        #ifdef GPU_QUEUE
        hipFree_assert(ro_net_gpu_handle->queues[i]);
        #else
        hipHostFree(ro_net_gpu_handle->queues[i]);
        #endif
    }

    hipHostFree_assert(ro_net_gpu_handle->queues);

    hipHostFree_assert(ro_net_gpu_handle);

    return ROC_SHMEM_SUCCESS;
}

bool
ROBackend::ro_net_process_queue(int queue_idx,
                                struct ro_net_handle *ro_net_gpu_handle,
                                bool *finalized)
{
    // Check if next element from the GPU is ready
    queue_desc_t *queue_desc = &ro_net_gpu_handle->queue_descs[queue_idx];
    DPRINTF(("Queue Desc read_idx %zu\n", queue_desc->read_idx));
    uint64_t read_slot = queue_desc->read_idx %
        ro_net_gpu_handle->queue_size;

    #ifdef GPU_QUEUE
    // Need to flush HDP read cache so we can see updates to the GPU Queue
    // descriptor
    hdp_read_inv(ro_nset_gpu_handle->hdp_regs);
    memcpy((void*)elt, &ro_net_gpu_handle->queues[queue_idx][read_slot],
           sizeof(queue_element_t));
    // Don't allow updates to the temporary element buffer
    const queue_element_t *next_element =
        reinterpret_cast<queue_element_t*>(elt);
    #else
    const queue_element_t *next_element =
        &ro_net_gpu_handle->queues[queue_idx][read_slot];
    #endif

    bool valid = false;
    if (next_element->valid) {
        valid = true;
        DPRINTF(("Rank %d Processing read_slot %lu of queue %d \n",
                my_pe, read_slot, queue_idx));

        transport->insertRequest(next_element, queue_idx);

        ro_net_gpu_handle->queues[queue_idx][read_slot].valid = 0;
        // Update the CPU's local read index
        queue_desc->read_idx++;
    }

    return valid;
}

/* Service thread routine that spins on a number of queues until the host
   calls net_finalize.  */
void
ROBackend::ro_net_poll(int thread_id, int num_threads)
{
    ro_net_handle *ro_net_gpu_handle =
        reinterpret_cast<ro_net_handle*>(backend_handle);
    int gpu_dev =0;
    hipGetDevice_assert(&gpu_dev);
    while (!ro_net_gpu_handle->done_flag) {
        for (int i = thread_id; i < ro_net_gpu_handle->num_queues;
             i += num_threads) {
            // Drain up to 64 requests from this queue if they are ready
            int req_count = 0;
            bool finalize;
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
    hipExtMallocWithFlags_assert(ptr, size, hipDeviceMallocFinegrained);
}
