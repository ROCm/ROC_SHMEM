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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_OPENSHMEM_TRANSPORT_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_OPENSHMEM_TRANSPORT_HPP

#include "transport.hpp"

namespace rocshmem {

class OpenSHMEMTransport : public Transport {
  public:
    OpenSHMEMTransport();

    Status
    initTransport(int num_queues,
                  BackendProxyT *proxy) override;

    Status
    finalizeTransport() override;

    Status
    allocateMemory(void **ptr, size_t size) override;

    Status
    deallocateMemory(void *ptr) override;

    Status
    barrier(int wg_id) override;

    Status
    reduction(void *dst,
              void *src,
              int size,
              int pe,
              int wg_id,
              int start,
              int logPstride,
              int sizePE,
              void *pWrk,
              long *pSync,
              RO_NET_Op op) override;

    Status
    broadcast(void *dst,
              void *src,
              int size,
              int pe,
              int wg_id,
              int start,
              int logPstride,
              int sizePE,
              int PE_root,
              long* pSync) override;

    Status
    putMem(void *dst,
           void *src,
           int size,
           int pe,
           int wg_id) override;

    Status
    getMem(void *dst,
           void *src,
           int size,
           int pe,
           int wg_id) override;

    Status
    amoFOP(void *dst,
           void *src,
           int64_t val,
           int pe,
           int wg_id,
           int threadId,
           bool blocking,
           ROC_SHMEM_OP op) override;

    Status
    amoFCAS(void *dst,
            void *src,
            int64_t val,
            int pe,
            int wg_id,
            int threadId,
            bool blocking,
            int64_t cond) override;

    Status
    quiet(int wg_id) override;

    Status
    progress() override;

    virtual int
    numOutstandingRequests() override;

    virtual MPI_Comm
    get_world_comm() override {
    }

  private:
    std::vector<shmem_ctx_t> ctx_vec;
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_OPENSHMEM_TRANSPORT_HPP
