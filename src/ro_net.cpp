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

#include <ro_net.hpp>
#include <ro_net_internal.hpp>
#include <transport.hpp>

char *elt = nullptr;
Transport *transport = nullptr;
bool RO_NET_DEBUG = false;

/***
 *
 * External Host-side API functions
 *
 ***/

int
ro_net_my_pe()
{
    return transport->getMyPe();
}

int
ro_net_n_pes()
{
    return transport->getNumPes();
}

ro_net_status_t
ro_net_finalize(ro_net_handle_t * ro_net_gpu_handle_p)
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) ro_net_gpu_handle_p;

    ro_net_free_runtime(ro_net_gpu_handle);

    return RO_NET_SUCCESS;
}

ro_net_status_t
ro_net_pre_init(ro_net_handle_t** ro_net_gpu_handle_ptr_p)
{
    struct ro_net_handle *ro_net_gpu_handle;
    hipHostMalloc_assert((void**) &ro_net_gpu_handle,
                         sizeof(struct ro_net_handle *));

    memset(ro_net_gpu_handle, 0, sizeof(ro_net_handle));
    *ro_net_gpu_handle_ptr_p = (ro_net_handle_t*) ro_net_gpu_handle;

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
        return RO_NET_OOM_ERROR;
    }

    if (getenv("RO_NET_DEBUG") != NULL)
        RO_NET_DEBUG = true;

    return RO_NET_SUCCESS;
}

ro_net_status_t
ro_net_init(ro_net_handle_t** ro_net_gpu_handle_ptr_p,
            int num_threads, int num_queues)
{
    struct ro_net_handle *ro_net_gpu_handle =
        (ro_net_handle *) *ro_net_gpu_handle_ptr_p;

    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess)
        return RO_NET_UNKNOWN_ERROR;

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
        return RO_NET_UNKNOWN_ERROR;
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
        return RO_NET_INVALID_ARGUMENTS;
    }

    ro_net_status_t return_code;

    char *value;
    ro_net_gpu_handle->queue_size = DEFAULT_QUEUE_SIZE;
    if ((value = getenv("RO_NET_QUEUE_SIZE")) != NULL) {
        ro_net_gpu_handle->queue_size = atoi(value);
        assert(ro_net_gpu_handle->queue_size != 0);
    }

    posix_memalign((void**)&elt, 64, sizeof(queue_element_t));
    if (!elt) {
        ro_net_free(ro_net_gpu_handle);
        return RO_NET_OOM_ERROR;
    }

    // allocate the resources for internal barriers
    unsigned int *barrier_ptr;
    hipMalloc_assert((void**) &barrier_ptr, sizeof(unsigned int ));
    *barrier_ptr=0;

    profiler_t *profiler_ptr;
    hipMalloc_assert((void**) &profiler_ptr,
                     sizeof(profiler_t)*num_queues);
    memset(profiler_ptr, 0,  sizeof(profiler_t)*num_queues);

    ro_net_gpu_handle->num_threads = num_threads;
    ro_net_gpu_handle->num_queues = num_queues;
    ro_net_gpu_handle->done_flag = 0;
    ro_net_gpu_handle->num_pes = transport->getNumPes();
    ro_net_gpu_handle->my_pe = transport->getMyPe();
    ro_net_gpu_handle->barrier_ptr = barrier_ptr;
    ro_net_gpu_handle->profiler = profiler_ptr;

    if ((return_code = transport->initTransport(num_queues, ro_net_gpu_handle)) !=
        RO_NET_SUCCESS) {
        ro_net_free(ro_net_gpu_handle);
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
    ro_net_device_uc_malloc((void **) queues, num_queues * sizeof(queue_element) *
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
    // If num_threads == 0, then there are no service threads and the user
    // needs to call ro_net_forward manually.
    ro_net_gpu_handle->worker_threads =  nullptr;
    if (num_threads > 0) {
        ro_net_gpu_handle->worker_threads =
            (pthread_t*) malloc(sizeof(pthread_t) * num_threads);

        for (int i = 0; i < num_threads; i++) {
            pthread_args_t *args = (pthread_args_t *) malloc(
                sizeof(pthread_args_t) * num_threads);
            args->thread_id = i;
            args->num_threads = num_threads;
            args->ro_net_gpu_handle = ro_net_gpu_handle;
            pthread_create(&ro_net_gpu_handle->worker_threads[i],
                           NULL, ro_net_poll, args);
        }
    }
    return RO_NET_SUCCESS;
}

// Host visible progress engine for 0 service thread mode.  Returns when
// each WG calls net_finalize.
ro_net_status_t
ro_net_forward(ro_net_handle_t *ro_net_gpu_handle_p, int num_wgs)
{
    struct ro_net_handle * ro_net_gpu_handle =
        (struct ro_net_handle *) ro_net_gpu_handle_p;

    assert(ro_net_gpu_handle->num_threads == 0);
    int finalize_count = 0;

    int gpu_dev = 0;
    hipGetDevice_assert(&gpu_dev);

    while (true) {
        for (int i = 0; i < ro_net_gpu_handle->num_queues; i++) {
            DPRINTF(("Processing Queue %d (%d finalized WGs) (token = %d)\n",
                    i, finalize_count, ro_net_gpu_handle->queueTokens[i]));

            bool finalize = false;
            ro_net_process_queue(i, ro_net_gpu_handle,
                                   &finalize);

            if (finalize)
                finalize_count++;

            if (finalize_count == num_wgs)
                return RO_NET_SUCCESS;
        }
    }

    // Wait for all outstanding requests to drain before returning
    while (transport->numOutstandingRequests()) {
    }
}

void *
ro_net_malloc(size_t size)
{
    void *ptr;
    transport->allocateMemory(&ptr, size);
    return ptr;
}

ro_net_status_t
ro_net_free(void * ptr)
{
    return transport->deallocateMemory(ptr);
}

/***
 *
 * Internal Host-side functions
 *
 ***/
inline void
load_elmt (__m256i* src, char* reg)
{
   __asm__ __volatile__ (
        "VMOVDQA %1,%%ymm1 ;"
        "VMOVDQA %%ymm1,%0 ;"
        : "=rm" (*reg)
        : "m" (*src)
        : "ymm1", "memory"
    );
}

ro_net_status_t
ro_net_reset_stats(ro_net_handle_t * ro_net_gpu_handle_p)
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) ro_net_gpu_handle_p;

    for (int i = 0; i < ro_net_gpu_handle->num_queues; i++) {
        queue_desc_t *q_desc = &ro_net_gpu_handle->queue_descs[i];
        memset(&q_desc->host_stats, 0, sizeof(host_stats_t));
        profiler_t *prof = &ro_net_gpu_handle->profiler[i];
        memset(prof, 0, sizeof(profiler_t));
    }

    return RO_NET_SUCCESS;
}

ro_net_status_t
ro_net_dump_stats(ro_net_handle_t * ro_net_gpu_handle_p)
{
    struct ro_net_handle *ro_net_gpu_handle =
        (struct ro_net_handle *) ro_net_gpu_handle_p;
    uint64_t numPutTotal = 0;
    uint64_t numGetTotal = 0;
    uint64_t numPutNbiTotal = 0;
    uint64_t numGetNbiTotal = 0;
    uint64_t numQuietTotal = 0;
    uint64_t numFinalizeTotal = 0;
    uint64_t total = 0;
    // Raw counts of different types of operations
    for (int i = 0; i < ro_net_gpu_handle->num_queues; i++) {
        queue_desc_t *q_desc = &ro_net_gpu_handle->queue_descs[i];
        numPutTotal += q_desc->host_stats.numPut;
        numPutNbiTotal += q_desc->host_stats.numPutNbi;
        numGetTotal += q_desc->host_stats.numGet;
        numGetNbiTotal += q_desc->host_stats.numGetNbi;
        numQuietTotal += q_desc->host_stats.numQuiet;
        numFinalizeTotal += q_desc->host_stats.numFinalize;
    }

    total += numPutTotal + numPutNbiTotal + numGetTotal + numGetNbiTotal +
        numQuietTotal + numFinalizeTotal;

#ifdef PROFILE
    int gpu_frequency_khz = 27000;
    uint64_t us_wait_slot = 0;
    uint64_t us_pack = 0;
    uint64_t us_fence1 = 0;
    uint64_t us_fence2 = 0;
    uint64_t us_wait_host = 0;
    for (int i = 0; i < ro_net_gpu_handle->num_queues; i++) {
        // Average latency as perceived from a thread
        profiler_t *prof = &ro_net_gpu_handle->profiler[i];
        us_wait_slot += prof->waitingOnSlot / (gpu_frequency_khz / 1000);
        us_pack += prof->packQueue / (gpu_frequency_khz / 1000);
        us_fence1 += prof->threadFence1 / (gpu_frequency_khz / 1000);
        us_fence2 += prof->threadFence2 / (gpu_frequency_khz / 1000);
        us_wait_host += prof->waitingOnHost / (gpu_frequency_khz / 1000);
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
#endif

    fprintf(stdout, "PE %d: Queues %d Threads %d\n",
        ro_net_gpu_handle->my_pe, ro_net_gpu_handle->num_queues,
        ro_net_gpu_handle->num_threads);
    fprintf(stdout, "PE %d: Puts %lu/%lu Gets %lu/%lu Quiets %lu Finalizes "
        "%lu Total %lu\n",
        ro_net_gpu_handle->my_pe, numPutTotal, numPutNbiTotal,
        numGetTotal, numGetNbiTotal, numQuietTotal, numFinalizeTotal, total);

    return RO_NET_SUCCESS;
}

ro_net_status_t
ro_net_free_runtime(struct ro_net_handle * ro_net_gpu_handle)
{
    assert(ro_net_gpu_handle);

    ro_net_gpu_handle->done_flag = 1;
    for (int i = 0; i < ro_net_gpu_handle->num_threads; i++) {
       pthread_join(ro_net_gpu_handle->worker_threads[i], NULL);
    }
    free(ro_net_gpu_handle->worker_threads);

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

    return RO_NET_SUCCESS;
}

bool
ro_net_process_queue(int queue_idx,
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
    hdp_read_inv(ro_net_gpu_handle->hdp_regs);
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
                ro_net_gpu_handle->my_pe, read_slot, queue_idx));

        transport->insertRequest(next_element, queue_idx);

        ro_net_gpu_handle->queues[queue_idx][read_slot].valid = 0;
        // Update the CPU's local read index
        queue_desc->read_idx++;
    }

    return valid;
}

/* Service thread routine that spins on a number of queues until the host
   calls net_finalize.  */
void *
ro_net_poll(void * pthread_args)
{
    pthread_args_t *args = (pthread_args_t *) pthread_args;
    int gpu_dev =0;
    hipGetDevice_assert(&gpu_dev);
    while (!args->ro_net_gpu_handle->done_flag) {
        for (int i = args->thread_id;
             i < args->ro_net_gpu_handle->num_queues;
             i += args->num_threads) {
            // Drain up to 64 requests from this queue if they are ready
            int req_count = 0;
            bool finalize;
            bool processed_req;
            do {
                processed_req =
                    ro_net_process_queue(i, args->ro_net_gpu_handle,
                                           &finalize);
                req_count++;
            } while (processed_req && (req_count < 64));
        }
    }
    free(args);
    pthread_exit(NULL);
}

void
ro_net_device_uc_malloc(void **ptr, size_t size)
{
    #ifdef UC_DEVICE_ALLOCATOR
    hipExtMallocWithFlags_assert(ptr, size, hipDeviceMallocFinegrained);
    #else
    hipMalloc_assert(ptr, size);
    #endif
}
