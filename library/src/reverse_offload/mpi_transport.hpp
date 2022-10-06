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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_MPI_TRANSPORT_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_MPI_TRANSPORT_HPP

#include <map>
#include <mutex>
#include <queue>
#include <vector>

#include "transport.hpp"

namespace rocshmem {

class HostInterface;

class MPITransport : public Transport {
  public:
    MPITransport();

    virtual ~MPITransport();

    Status
    initTransport(int num_queues,
                  BackendProxyT *proxy) override;

    Status
    finalizeTransport() override;

    Status
    allocateMemory(void **ptr,
                   size_t size) override {
        return Status::ROC_SHMEM_SUCCESS;
    };

    Status
    deallocateMemory(void *ptr) override {
        return Status::ROC_SHMEM_SUCCESS;
    };

    Status
    createNewTeam(ROBackend *backend,
                  Team *parent_team,
                  TeamInfo *team_info_wrt_parent,
                  TeamInfo *team_info_wrt_world,
                  int num_pes,
                  int my_pe_in_new_team,
                  MPI_Comm team_comm,
                  roc_shmem_team_t *new_team) override;

    Status
    barrier(int wg_id,
            int threadId,
            bool blocking,
            MPI_Comm team) override;

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
              ROC_SHMEM_OP op,
              ro_net_types type,
              int threadId,
              bool blocking) override;

    Status
    team_reduction(void *dst,
                   void *src,
                   int size,
                   int wg_id,
                   MPI_Comm team,
                   ROC_SHMEM_OP op,
                   ro_net_types type,
                   int threadId,
                   bool blocking) override;

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
              long *pSync,
              ro_net_types type,
              int threadId,
              bool blocking) override;

    Status
    team_broadcast(void *dst,
                   void *src,
                   int size,
                   int wg_id,
                   MPI_Comm team,
                   int PE_root,
                   ro_net_types type,
                   int threadId,
                   bool blocking) override;

    Status
    alltoall(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking) override;

    Status
    alltoall_broadcast(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    alltoall_mpi(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    alltoall_gcen(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    alltoall_gcen2(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    fcollect(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking) override;

    Status
    fcollect_broadcast(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    fcollect_mpi(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    fcollect_gcen(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    fcollect_gcen2(void *dst,
             void *src,
             int size,
             int wg_id,
             MPI_Comm team,
             void *ata_buffptr,
             ro_net_types type,
             int threadId,
             bool blocking);

    Status
    putMem(void *dst,
           void *src,
           int size,
           int pe,
           int wg_id,
           int threadId,
           bool blocking,
           bool inline_data = false) override;

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
    getMem(void *dst,
           void *src,
           int size,
           int pe,
           int wg_id,
           int threadId,
           bool blocking) override;

    Status
    quiet(int wg_id,
          int threadId) override;

    Status
    progress() override;

    virtual int
    numOutstandingRequests() override;

    virtual void
    insertRequest(const queue_element_t *element,
                  int queue_id) override;

    virtual bool
    readyForFinalize() override {
        return !transport_up;
    }

    MPI_Comm ro_net_comm_world {};

    virtual MPI_Comm
    get_world_comm() override {
        return ro_net_comm_world;
    }

    HostInterface *host_interface {nullptr};

    void
    global_exit(int status) override;

    MPI_Op
    get_mpi_op(ROC_SHMEM_OP op);

  private:
    struct CommKey
    {
        CommKey(int _start,
                int _logPstride,
                int _size)
            : start(_start),
              logPstride(_logPstride),
              size(_size) {
        }

        bool
        operator< (const CommKey& key) const {
            return start < key.start ||
                   (start == key.start && logPstride < key.logPstride) ||
                   (start == key.start && logPstride == key.logPstride &&
                    size < key.size);
        }

        int start {-1};

        int logPstride {-1};

        int size {-1};
    };

    struct RequestProperties
    {
        RequestProperties(int _threadId,
                          int _wgId,
                          bool _blocking,
                          void *_src,
                          bool _inline_data)
            : threadId(_threadId),
              wgId(_wgId),
              blocking(_blocking),
              src(_src),
              inline_data(_inline_data) {
        }

        RequestProperties(int _threadId,
                          int _wgId,
                          bool _blocking)
            : threadId(_threadId),
              wgId(_wgId),
              blocking(_blocking),
              src(nullptr),
              inline_data(false) {
        }

        int threadId {-1};
        int wgId {-1};
        bool blocking {};
        void *src {nullptr};
        bool inline_data {};
    };

    MPI_Comm
    createComm(int start,
               int logPstride,
               int size);

    void
    threadProgressEngine();

    void
    submitRequestsToMPI();

    // Unordered vector of in-flight MPI Requests. Can complete out of order.
    std::vector<RequestProperties> req_prop_vec {};

    std::vector<MPI_Request> req_vec {};

    std::vector<std::vector<int> > waiting_quiet {};

    std::vector<int> outstanding {};

    std::map<CommKey, MPI_Comm> comm_map {};

    std::queue<const queue_element_t *> q {};

    std::queue<int> q_wgid {};

    std::mutex queue_mutex {};

    volatile int hostBarrierDone {false};

    volatile bool transport_up {false};

    BackendProxyT *backend_proxy {nullptr};

    std::thread *progress_thread {nullptr};

    int *indices {nullptr};

    const int INDICES_SIZE {128};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_MPI_TRANSPORT_HPP
