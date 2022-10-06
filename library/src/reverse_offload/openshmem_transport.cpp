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

#include "openshmem_transport.hpp"
#include "shmem.h"

namespace rocshmem {

#define NET_CHECK(cmd) {\
    if (cmd != 0) {\
        fprintf(stderr, "Unrecoverable error: SHMEM Failure\n");\
        exit(1);\
    }\
}

OpenSHMEMTransport::OpenSHMEMTransport()
    : Transport() {
    // TODO: Provide context support
    int provided {};
    shmem_init_thread(SHMEM_THREAD_MULTIPLE, &provided);
    if (provided != SHMEM_THREAD_MULTIPLE) {
        fprintf(stderr, "Warning requested multi-thread level is not "
                        "supported \n");
    }
    num_pes = shmem_n_pes();
    my_pe = shmem_my_pe();
}

Status
OpenSHMEMTransport::initTransport(int num_queues) {
    ctx_vec.resize(num_queues);
    for (int i = 0; i < ctx_vec.size(); i++) {
        NET_CHECK(shmem_ctx_create(SHMEM_CTX_SERIALIZED,
                                   ctx_vec.data() + i));
    }
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::finalizeTransport() {
    shmem_finalize();
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::allocateMemory(void **ptr,
                                   size_t size) {
    if ((*ptr = shmem_malloc(size)) == nullptr) {
        return ROC_SHMEM_OOM_ERROR;
    }
    CHECK_HIP(hipHostRegister(*ptr, size, 0));
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::deallocateMemory(void *ptr) {
    CHECK_HIP(hipHostUnregister(ptr));
    shmem_free(ptr);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::barrier(int wg_id) {
    shmem_barrier_all();
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::reduction(void *dst,
                              void *src,
                              int size,
                              int pe,
                              int wg_id,
                              int start,
                              int logPstride,
                              int sizePE,
                              void *pWrk,
                              long *pSync,
                              RO_NET_Op op) {
    assert(op == RO_NET_SUM);
    shmem_float_sum_to_all((float*)dst,
                           (float*)src,
                           size,
                           pe,
                           logPstride,
                           sizePE,
                           (float*)pWrk,
                           pSync);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::broadcast(void *dst,
                              void *src,
                              int size,
                              int pe,
                              int wg_id,
                              int start,
                              int logPstride,
                              int sizePE,
                              int root,
                              long *pSync) {
    shmem_broadcast((float*)dst,
                    (float*)src,
                    size,
                    root,
                    pe,
                    logPstride,
                    sizePE,
                    pSync);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::putMem(void *dst,
                           void *src,
                           int size,
                           int pe,
                           int wg_id) {
    assert(wg_id < ctx_vec.size());
    shmem_ctx_putmem_nbi(ctx_vec[wg_id], dst, src, size, pe);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::getMem(void *dst,
                           void *src,
                           int size,
                           int pe,
                           int wg_id) {
    assert(wg_id < ctx_vec.size());
    shmem_ctx_getmem_nbi(ctx_vec[wg_id], dst, src, size, pe);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::amoFOP(void *dst,
                           void *src,
                           int64_t val,
                           int pe,
                           int wg_id,
                           int threadId,
                           bool blocking,
                           ROC_SHMEM_OP op) {
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::progress(int wg_id,
                             BackendProxyT *proxy) {
    // TODO: Might want to delay a quiet for a while to make sure we get
    // messages from other contexts injected before we block the service
    // thread.
    if (proxy->needs_quiet[wg_id] ||
        proxy->needs_blocking[wg_id]) {
        assert(wg_id < ctx_vec.size());
        shmem_ctx_quiet(ctx_vec[wg_id]);
        proxy->needs_quiet[wg_id] = false;
        proxy->needs_blocking[wg_id] = false;
        proxy->queue_descs[wg_id].status = 1;

        if (handle->gpu_queue) {
            SFENCE();
            proxy->hdp_policy->hdp_flush();
        }
    }
    return Status::ROC_SHMEM_SUCCESS;
}

Status
OpenSHMEMTransport::quiet(int wg_id) {
    return Status::ROC_SHMEM_SUCCESS;
}

int
OpenSHMEMTransport::numOutstandingRequests() {
    for (auto ctx : ctx_vec)
        shmem_ctx_quiet(ctx);
    return 0;
}

}  // namespace rocshmem
