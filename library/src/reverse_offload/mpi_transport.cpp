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

#include "mpi_transport.hpp"

#include "backend_ro.hpp"
#include "host.hpp"
#include "ro_net_team.hpp"

namespace rocshmem {

#define NET_CHECK(cmd) {\
    if (cmd != MPI_SUCCESS) {\
        fprintf(stderr, "Unrecoverable error: MPI Failure\n");\
        exit(1);\
    }\
}

MPITransport::MPITransport()
    : Transport{} {
    indices = new int[INDICES_SIZE];

    int init_done {};
    NET_CHECK(MPI_Initialized(&init_done));

    int provided {};
    if (!init_done) {
        NET_CHECK(MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided));
        if (provided != MPI_THREAD_MULTIPLE) {
            fprintf(stderr, "Warning requested multi-thread level is not "
                        "supported \n");
        }
    }

    NET_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &ro_net_comm_world));
    NET_CHECK(MPI_Comm_size(ro_net_comm_world, &num_pes));
    NET_CHECK(MPI_Comm_rank(ro_net_comm_world, &my_pe));
}

MPITransport::~MPITransport() {
    delete [] indices;
}

void
MPITransport::threadProgressEngine() {
    auto *bp {backend_proxy->get()};

    transport_up = true;
    while (!(bp->done_flag)) {
        submitRequestsToMPI();
        progress();
    }
    transport_up = false;
}

void
MPITransport::insertRequest(const queue_element_t *element,
                            int queue_id) {
    std::unique_lock<std::mutex> mlock(queue_mutex);
    q.push(element);
    q_wgid.push(queue_id);
}

void
MPITransport::submitRequestsToMPI() {
    if (q.empty())
        return;

    std::unique_lock<std::mutex> mlock(queue_mutex);

    const queue_element_t *next_element {q.front()};
    int queue_idx {q_wgid.front()};
    q.pop();
    q_wgid.pop();
    mlock.unlock();

    switch (next_element->type) {
        case RO_NET_PUT:
            putMem(next_element->dst,
                   next_element->src,
                   next_element->size,
                   next_element->PE,
                   queue_idx,
                   next_element->threadId,
                   true);
            DPRINTF("Received PUT dst %p src %p size %d pe %d\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE);
            break;
        case RO_NET_P: {
            // No equivalent inline OP for MPI.
            // Allocate a temp buffer for value.
            // TODO: @Brandon this is a memory leak - fix it
            void *source_buffer {malloc(next_element->size)};

            ::memcpy(source_buffer,
                     &next_element->src,
                     next_element->size);

            putMem(next_element->dst,
                   source_buffer,
                   next_element->size,
                   next_element->PE,
                   queue_idx,
                   next_element->threadId,
                   true,
                   true);
            DPRINTF("Received P dst %p value %p pe %d\n",
                    next_element->dst,
                    next_element->src,
                    next_element->PE);
            break;
        }
        case RO_NET_GET:
            getMem(next_element->dst,
                   next_element->src,
                   next_element->size,
                   next_element->PE,
                   queue_idx,
                   next_element->threadId,
                   true);
            DPRINTF("Received GET dst %p src %p size %d pe %d\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE);
            break;
        case RO_NET_PUT_NBI:
            putMem(next_element->dst,
                   next_element->src,
                   next_element->size,
                   next_element->PE,
                   queue_idx,
                   next_element->threadId,
                   false);
            DPRINTF("Received PUT NBI dst %p src %p size %d pe %d\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE);

            break;
        case RO_NET_GET_NBI:
            getMem(next_element->dst,
                   next_element->src,
                   next_element->size,
                   next_element->PE,
                   queue_idx,
                   next_element->threadId,
                   false);
            DPRINTF("Received GET NBI dst %p src %p size %d pe %d\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE);
            break;
        case RO_NET_AMO_FOP:
            amoFOP(next_element->dst,
                   next_element->src,
                   next_element->size,
                   next_element->PE,
                   queue_idx,
                   next_element->threadId,
                   true,
                   (ROC_SHMEM_OP)next_element->op);
            DPRINTF("Received AMO dst %p src %p Val %d pe %d\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE);
            break;
        case RO_NET_AMO_FCAS:
            amoFCAS(next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE,
                    queue_idx,
                    next_element->threadId,
                    true,
                    (int64_t)next_element->pWrk);
            DPRINTF("Received F_CSWAP dst %p src %p Val %d pe %d cond %ld\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE,
                    (int64_t)next_element->pWrk);
            break;
        case RO_NET_TEAM_TO_ALL:
            team_reduction(next_element->dst,
                           next_element->src,
                           next_element->size,
                           queue_idx,
                           next_element->team_comm,
                           (ROC_SHMEM_OP)next_element->op,
                           (ro_net_types)next_element->datatype,
                           next_element->threadId,
                           true);
            DPRINTF("Received FLOAT_SUM_TEAM_TO_ALL dst %p src %p size %d team %d\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->team_comm);
            break;
        case RO_NET_TO_ALL:
            reduction(next_element->dst,
                      next_element->src,
                      next_element->size,
                      next_element->PE,
                      queue_idx,
                      next_element->PE,
                      next_element->logPE_stride,
                      next_element->PE_size,
                      next_element->pWrk,
                      next_element->pSync,
                      (ROC_SHMEM_OP)next_element->op,
                      (ro_net_types)next_element->datatype,
                      next_element->threadId,
                      true);
            DPRINTF("Received FLOAT_SUM_TO_ALL dst %p src %p size %d "
                    "PE_start %d, logPE_stride %d, PE_size %d, pWrk %p, pSync %p\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE,
                    next_element->logPE_stride,
                    next_element->PE_size,
                    next_element->pWrk,
                    next_element->pSync);
            break;
        case RO_NET_TEAM_BROADCAST:
            team_broadcast(next_element->dst,
                           next_element->src,
                           next_element->size,
                           queue_idx,
                           next_element->team_comm,
                           next_element->PE_root,
                           (ro_net_types)next_element->datatype,
                           next_element->threadId,
                           true);
            DPRINTF("Received TEAM_BROADCAST dst %p src %p size %d "
                    "team %d, PE_root %d \n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->team_comm,
                    next_element->PE_root);
            break;
        case RO_NET_BROADCAST:
            broadcast(next_element->dst,
                      next_element->src,
                      next_element->size,
                      next_element->PE, queue_idx,
                      next_element->PE,
                      next_element->logPE_stride,
                      next_element->PE_size,
                      next_element->PE_root,
                      next_element->pSync,
                      (ro_net_types)next_element->datatype,
                      next_element->threadId,
                      true);
            DPRINTF("Received BROADCAST dst %p src %p size %d PE_start %d, "
                    "logPE_stride %d, PE_size %d, PE_root %d, pSync %p\n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->PE,
                    next_element->logPE_stride,
                    next_element->PE_size,
                    next_element->PE_root,
                    next_element->pSync);
            break;
        case RO_NET_ALLTOALL:
            alltoall(next_element->dst,
                     next_element->src,
                     next_element->size, queue_idx,
                     next_element->team_comm,
                     next_element->pWrk,
                     (ro_net_types) next_element->datatype,
                     next_element->threadId, true);

            DPRINTF(("Received ALLTOALL  dst %p src %p size %d team %d \n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->team_comm));

            break;
        case RO_NET_FCOLLECT:
            fcollect(next_element->dst,
                     next_element->src,
                     next_element->size, queue_idx,
                     next_element->team_comm,
                     next_element->pWrk,
                     (ro_net_types) next_element->datatype,
                     next_element->threadId, true);

            DPRINTF(("Received FCOLLECT  dst %p src %p size %d team %d \n",
                    next_element->dst,
                    next_element->src,
                    next_element->size,
                    next_element->team_comm));

            break;
        case RO_NET_BARRIER_ALL:
            barrier(queue_idx, next_element->threadId, true,
                    ro_net_comm_world);
            DPRINTF("Received Barrier_all\n");
            break;
        case RO_NET_SYNC:
            barrier(queue_idx, next_element->threadId, true,
            	next_element->team_comm);
            DPRINTF(("Received Sync\n"));
            break;
        case RO_NET_FENCE:
        case RO_NET_QUIET:
            quiet(queue_idx, next_element->threadId);
            DPRINTF("Received FENCE/QUIET\n");
            break;
        case RO_NET_FINALIZE:
            quiet(queue_idx, next_element->threadId);
            DPRINTF("Received Finalize\n");
            break;
        default:
            fprintf(stderr, "Invalid GPU Packet received, exiting....\n");
            exit(1);
            break;
    }

    delete next_element;
}

Status
MPITransport::initTransport(int num_queues,
                            BackendProxyT *proxy) {
    waiting_quiet.resize(num_queues, std::vector<int>());
    outstanding.resize(num_queues, 0);
    transport_up = false;

    backend_proxy = proxy;
    auto *bp {backend_proxy->get()};

    host_interface = new HostInterface(bp->hdp_policy,
                                       ro_net_comm_world,
                                       bp->heap_ptr);
    progress_thread =
        new std::thread(&MPITransport::threadProgressEngine, this);
    while (!transport_up) {
        ;
    }
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::finalizeTransport() {
    progress_thread->join();
    delete progress_thread;
    delete host_interface;
    return Status::ROC_SHMEM_SUCCESS;
}

roc_shmem_team_t
get_external_team(ROTeam *team) {
    return reinterpret_cast<roc_shmem_team_t>(team);
}

Status
MPITransport::createNewTeam(ROBackend *backend,
                            Team *parent_team,
                            TeamInfo *team_info_wrt_parent,
                            TeamInfo *team_info_wrt_world,
                            int num_pes,
                            int my_pe_in_new_team,
                            MPI_Comm team_comm,
                            roc_shmem_team_t *new_team) {
    ROTeam *new_team_obj {nullptr};

    CHECK_HIP(hipMalloc(&new_team_obj, sizeof(ROTeam)));

    new (new_team_obj) ROTeam(backend,
                              team_info_wrt_parent,
                              team_info_wrt_world,
                              num_pes,
                              my_pe_in_new_team,
                              team_comm);

    *new_team = get_external_team(new_team_obj);

    return Status::ROC_SHMEM_SUCCESS;
}

MPI_Comm
MPITransport::createComm(int start,
                         int stride,
                         int size) {
    CommKey key(start, stride, size);
    auto it {comm_map.find(key)};
    if (it != comm_map.end()) {
        DPRINTF("Using cached communicator\n");
        return it->second;
    }

    int world_size {};
    NET_CHECK(MPI_Comm_size(ro_net_comm_world, &world_size));

    MPI_Comm comm {};
    if (start == 0 && stride == 1 && size == world_size) {
        NET_CHECK(MPI_Comm_dup(ro_net_comm_world, &comm));
    } else {
        MPI_Group world_group {};
        NET_CHECK(MPI_Comm_group(ro_net_comm_world, &world_group));

        int group_ranks[size];
        group_ranks[0] = start;
        for (int i {1}; i < size; i++) {
            group_ranks[i] = group_ranks[i - 1] + stride;
        }

        MPI_Group new_group {};
        NET_CHECK(MPI_Group_incl(world_group, size, group_ranks, &new_group));
        NET_CHECK(MPI_Comm_create_group(ro_net_comm_world, new_group, 0, &comm));
    }

    comm_map.insert(std::pair<CommKey, MPI_Comm>(key, comm));
    DPRINTF("Creating new communicator\n");

    return comm;
}

void
MPITransport::global_exit(int status) {
    MPI_Abort(ro_net_comm_world, status);
}

Status
MPITransport::barrier(int wg_id,
                      int threadId,
                      bool blocking,
                      MPI_Comm team) {
    MPI_Request request {};
    NET_CHECK(MPI_Ibarrier(team, &request));

    req_prop_vec.emplace_back(threadId, wg_id, blocking);
    req_vec.push_back(request);
    outstanding[wg_id]++;

    return Status::ROC_SHMEM_SUCCESS;
}

MPI_Op
MPITransport::get_mpi_op(ROC_SHMEM_OP op) {
    switch(op) {
        case ROC_SHMEM_SUM:
            return MPI_SUM;
        case ROC_SHMEM_MAX:
            return MPI_MAX;
        case ROC_SHMEM_MIN:
            return MPI_MIN;
        case ROC_SHMEM_PROD:
            return MPI_PROD;
        case ROC_SHMEM_AND:
            return MPI_BAND;
        case ROC_SHMEM_OR:
            return MPI_BOR;
        case ROC_SHMEM_XOR:
            return MPI_BXOR;
        default:
            fprintf(stderr, "Unknown ROC_SHMEM op MPI conversion %d\n", op);
            exit(1);
    }
}

static MPI_Datatype
convertType(ro_net_types type) {
    switch(type) {
        case RO_NET_FLOAT:
            return MPI_FLOAT;
        case RO_NET_DOUBLE:
            return MPI_DOUBLE;
        case RO_NET_INT:
            return MPI_INT;
        case RO_NET_LONG:
            return MPI_LONG;
        case RO_NET_LONG_LONG:
            return MPI_LONG_LONG;
        case RO_NET_SHORT:
            return MPI_SHORT;
        case RO_NET_LONG_DOUBLE:
            return MPI_LONG_DOUBLE;
        default:
            fprintf(stderr, "Unknown ROC_SHMEM type MPI conversion %d\n", type);
            exit(1);
    }
}

Status
MPITransport::reduction(void *dst,
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
                        bool blocking) {
    MPI_Request request {};
    MPI_Op mpi_op {get_mpi_op(op)};
    MPI_Datatype mpi_type {convertType(type)};
    MPI_Comm comm {createComm(start, 1 << logPstride, sizePE)};

    if (dst == src) {
        NET_CHECK(MPI_Iallreduce(MPI_IN_PLACE,
                                 dst,
                                 size,
                                 mpi_type,
                                 mpi_op,
                                 comm,
                                 &request));
    } else {
        NET_CHECK(MPI_Iallreduce(src,
                                 dst,
                                 size,
                                 mpi_type,
                                 mpi_op,
                                 comm,
                                 &request));
    }

    req_prop_vec.emplace_back(threadId, wg_id, blocking);
    req_vec.push_back(request);
    outstanding[wg_id]++;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::broadcast(void *dst,
                        void *src,
                        int size,
                        int pe,
                        int wg_id,
                        int start,
                        int logPstride,
                        int sizePE,
                        int root,
                        long *pSync,
                        ro_net_types type,
                        int threadId,
                        bool blocking) {

    MPI_Comm comm {createComm(start, 1 << logPstride, sizePE)};

    int new_rank {};
    MPI_Comm_rank(comm, &new_rank);

    void *data {nullptr};
    if (new_rank == root) {
        data = src;
    } else {
        data = dst;
    }

    MPI_Request request {};
    MPI_Datatype mpi_type {convertType(type)};
    NET_CHECK(MPI_Ibcast(data, size, mpi_type, root, comm, &request));

    req_prop_vec.emplace_back(threadId, wg_id, blocking);
    req_vec.push_back(request);

    outstanding[wg_id]++;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::team_reduction(void *dst,
                             void *src,
                             int size,
                             int wg_id,
                             MPI_Comm team,
                             ROC_SHMEM_OP op,
                             ro_net_types type,
                             int threadId,
                             bool blocking) {
    MPI_Request request {};

    MPI_Op mpi_op {get_mpi_op(op)};
    MPI_Datatype mpi_type {convertType(type)};
    MPI_Comm comm {team};

    if (dst == src) {
        NET_CHECK(MPI_Iallreduce(MPI_IN_PLACE,
                                 dst,
                                 size, mpi_type, mpi_op,
                                 comm, &request));
    } else {
        NET_CHECK(MPI_Iallreduce(src,
                                 dst,
                                 size,
                                 mpi_type,
                                 mpi_op,
                                 comm,
                                 &request));
    }

    req_prop_vec.emplace_back(threadId, wg_id, blocking);
    req_vec.push_back(request);
    outstanding[wg_id]++;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::team_broadcast(void *dst,
                             void *src,
                             int size,
                             int wg_id,
                             MPI_Comm team,
                             int root,
                             ro_net_types type,
                             int threadId,
                             bool blocking) {

    MPI_Comm comm {team};
    int new_rank {};
    MPI_Comm_rank(comm, &new_rank);
    void *data {nullptr};
    if (new_rank == root) {
        data = src;
    } else {
        data = dst;
    }

    MPI_Datatype mpi_type {convertType(type)};
    MPI_Request request {};
    NET_CHECK(MPI_Ibcast(data,
                         size,
                         mpi_type,
                         root,
                         comm,
                         &request));

    req_prop_vec.emplace_back(threadId, wg_id, blocking);
    req_vec.push_back(request);
    outstanding[wg_id]++;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::alltoall(void *dst,
                       void *src,
                       int size,
                       int wg_id,
                       MPI_Comm team,
                       void *ata_buffptr,
                       ro_net_types type,
                       int threadId,
                       bool blocking) {
    int pe_size, type_size;
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Datatype mpi_type = convertType(type);
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));

    // Currently GPU-centric algo only supports multiples of square root
    int num_clust = sqrt(pe_size);
    int clust_size = (pe_size + num_clust - 1) / num_clust;
    // TODO: Allow any size of cluster
    // GPU-centric algorithm is most optimal under these conditions
    if((pe_size >= 8 || type_size * size < 2048) &&
        num_clust * clust_size == pe_size) {
        return alltoall_gcen(dst, src, size, wg_id, team,
            ata_buffptr, type, threadId, blocking);
    }
    // For some reason MPI crashes for > 512 messages
    else if(size <= 512) {
        return alltoall_mpi(dst, src, size, wg_id, team,
            ata_buffptr, type, threadId, blocking);
    }
    else {
        return alltoall_broadcast(dst, src, size, wg_id, team, 
            ata_buffptr, type, threadId, blocking);
    }
}

Status
MPITransport::alltoall_broadcast(void *dst,
                                 void *src,
                                 int size,
                                 int wg_id,
                                 MPI_Comm team,
                                 void *ata_buffptr,
                                 ro_net_types type,
                                 int threadId,
                                 bool blocking) {
    // Broadcast implementation of alltoall
    auto *bp {backend_proxy->get()};
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Group grp, world_grp;
    NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
    NET_CHECK(MPI_Comm_group(comm, &grp));

    int grp_size;
    NET_CHECK(MPI_Group_size(grp, &grp_size));

    int *ranks = (int*)malloc(grp_size * sizeof(int));
    int *world_ranks = (int*)malloc(grp_size * sizeof(int));

    for (int i = 0; i < grp_size; i++)
        ranks[i] = i;
    // Convert comm ranks to global ranks for rput
    NET_CHECK(MPI_Group_translate_ranks(grp, grp_size, ranks,
        world_grp, world_ranks));

    int type_size;
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));
    MPI_Request *pe_req = (MPI_Request*)malloc(
        sizeof(MPI_Request) * pe_size);
    // Put data to all PEs
    for(int i = 0; i < pe_size; ++i) {
        int src_offset = i * type_size * size;
        int dst_offset = new_rank * type_size * size;
        NET_CHECK(MPI_Rput((char *)src + src_offset, size,
            mpi_type, world_ranks[i], bp->heap_window_info[wg_id]->
            get_offset((char *)dst + dst_offset),
            size, mpi_type, bp->heap_window_info[wg_id]->get_win(),
            &pe_req[i]));
    }
    NET_CHECK(MPI_Waitall(pe_size, pe_req, MPI_STATUSES_IGNORE));
    NET_CHECK(MPI_Win_flush_all(bp->heap_window_info[wg_id]->get_win()));

    free(ranks);
    free(world_ranks);
    free(pe_req);
    // Now wait for completion
    barrier(wg_id, threadId, blocking, comm);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::alltoall_mpi(void *dst,
                           void *src,
                           int size,
                           int wg_id,
                           MPI_Comm team,
                           void *ata_buffptr,
                           ro_net_types type,
                           int threadId,
                           bool blocking) {
    // MPI's implementation of alltoall
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    NET_CHECK(MPI_Alltoall(src, size, mpi_type, dst,
                  size, mpi_type, comm));
    quiet(wg_id, threadId);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::alltoall_gcen(void *dst,
                            void *src,
                            int size,
                            int wg_id,
                            MPI_Comm team,
                            void *ata_buffptr,
                            ro_net_types type,
                            int threadId,
                            bool blocking) {
    // GPU-centric implementation of alltoall
    auto *bp {backend_proxy->get()};
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Group grp, world_grp;
    NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
    NET_CHECK(MPI_Comm_group(comm, &grp));

    int grp_size;
    NET_CHECK(MPI_Group_size(grp, &grp_size));

    int *ranks = (int*)malloc(grp_size * sizeof(int));
    int *world_ranks = (int*)malloc(grp_size * sizeof(int));

    for (int i = 0; i < grp_size; i++)
        ranks[i] = i;
    // Convert comm ranks to global ranks for rput
    NET_CHECK(MPI_Group_translate_ranks(grp, grp_size, ranks,
        world_grp, world_ranks));

    int type_size;
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));

    // Works when number of PEs divisible by root(PE_size)
    int num_clust = sqrt(pe_size);
    int clust_size = (pe_size + num_clust - 1) / num_clust;
    // TODO: Allow any size of cluster
    assert(num_clust * clust_size == pe_size);
    int clust_id = new_rank / clust_size;

    if(MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
        fprintf(stderr, "Alltoall size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
        exit(-1);
    }

    MPI_Request *clust_req = new MPI_Request[pe_size];

    // Step 1: Send data to PEs in cluster
    for(int i = 0; i < pe_size; ++i) {
        int src_offset = (new_rank % clust_size + (i / clust_size) *
                         clust_size) * type_size * size;
        int dst_offset = i * type_size * size;
        NET_CHECK(MPI_Rget((void*)((char *)ata_buffptr + dst_offset), size,
            mpi_type, world_ranks[clust_id * clust_size + (i % clust_size)],
            bp->heap_window_info[wg_id]->get_offset((char*)src + src_offset),
            size, mpi_type, bp->heap_window_info[wg_id]->get_win(),
            &clust_req[i]));
    }

    NET_CHECK(MPI_Waitall(pe_size, clust_req, MPI_STATUSES_IGNORE));

    // Step 2: Send final data to PEs outside cluster
    for(int i = 0; i < num_clust; ++i) {
        int src_offset = i * type_size * size * clust_size;
        int dst_offset = clust_id * type_size * size * clust_size;
        NET_CHECK(MPI_Put((void*)((char *)ata_buffptr + src_offset), size *
            clust_size, mpi_type, world_ranks[(new_rank % clust_size) + i *
            clust_size], bp->heap_window_info[wg_id]->get_offset(dst) +
            dst_offset, size * clust_size, mpi_type,
            bp->heap_window_info[wg_id]->get_win()));

        // Since MPI makes puts as complete as soon as the local buffer is free,
        // we need a flush to satisfy quiet.
        NET_CHECK(MPI_Win_flush(world_ranks[(new_rank % clust_size) + i *
                  clust_size], bp->heap_window_info[wg_id]->get_win()));
    }

    int stride = world_ranks[1] - world_ranks[0];
    MPI_Comm comm_cluster = createComm(world_ranks[clust_id * clust_size],
                            stride, clust_size);
    MPI_Comm comm_ring = createComm(world_ranks[new_rank % clust_size],
                         stride * clust_size, num_clust);
    // Now wait for completion
    barrier(wg_id, threadId, false, comm_cluster);
    barrier(wg_id, threadId, blocking, comm_ring);

    free(ranks);
    free(world_ranks);
    free(clust_req);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::alltoall_gcen2(void *dst,
                            void *src,
                            int size,
                            int wg_id,
                            MPI_Comm team,
                            void *ata_buffptr,
                            ro_net_types type,
                            int threadId,
                            bool blocking) {
    // GPU-centric alltoall with in-place blocking synchronization
    auto *bp {backend_proxy->get()};
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Group grp, world_grp;
    NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
    NET_CHECK(MPI_Comm_group(comm, &grp));

    int grp_size;
    NET_CHECK(MPI_Group_size(grp, &grp_size));

    int *ranks = (int*)malloc(grp_size * sizeof(int));
    int *world_ranks = (int*)malloc(grp_size * sizeof(int));

    for (int i = 0; i < grp_size; i++)
        ranks[i] = i;
    // Convert comm ranks to global ranks for rput
    NET_CHECK(MPI_Group_translate_ranks(grp, grp_size, ranks,
        world_grp, world_ranks));

    int type_size;
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));

    // Works when number of PEs divisible by root(PE_size)
    int num_clust = sqrt(pe_size);
    int clust_size = (pe_size + num_clust - 1) / num_clust;
    // TODO: Allow any size of cluster
    assert(num_clust * clust_size == pe_size);
    int clust_id = new_rank / clust_size;

    if(MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
        fprintf(stderr, "Alltoall size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
        exit(-1);
    }

    MPI_Request *clust_req = new MPI_Request[pe_size];

    // Step 1: Send data to PEs in cluster
    for(int i = 0; i < pe_size; ++i) {
        int src_offset = (new_rank % clust_size + (i / clust_size) *
                         clust_size) * type_size * size;
        int dst_offset = i * type_size * size;
        NET_CHECK(MPI_Rget((void*)((char *)ata_buffptr + dst_offset), size,
            mpi_type, world_ranks[clust_id * clust_size + (i % clust_size)],
            bp->heap_window_info[wg_id]->get_offset((char*)src + src_offset),
            size, mpi_type, bp->heap_window_info[wg_id]->get_win(),
            &clust_req[i]));
    }

    NET_CHECK(MPI_Waitall(pe_size, clust_req, MPI_STATUSES_IGNORE));

    // Now wait
    int stride = world_ranks[1] - world_ranks[0];
    MPI_Comm comm_cluster = createComm(world_ranks[clust_id * clust_size],
                            stride, clust_size);
    MPI_Barrier(comm_cluster);

    // Step 2: Send final data to PEs outside cluster
    for(int i = 0; i < num_clust; ++i) {
        int src_offset = i * type_size * size * clust_size;
        int dst_offset = clust_id * type_size * size * clust_size;
        NET_CHECK(MPI_Put((void*)((char *)ata_buffptr + src_offset),
            size * clust_size, mpi_type, world_ranks[(new_rank % clust_size) +
            i * clust_size], bp->heap_window_info[wg_id]->get_offset(dst) +
            dst_offset, size * clust_size, mpi_type,
            bp->heap_window_info[wg_id]->get_win()));

        // Since MPI makes puts as complete as soon as the local buffer is free,
        // we need a flush to satisfy quiet.
        NET_CHECK(MPI_Win_flush(world_ranks[(new_rank % clust_size) +
                    i * clust_size], bp->heap_window_info[wg_id]->get_win()));
    }

    MPI_Comm comm_ring = createComm(world_ranks[new_rank % clust_size],
                         stride * clust_size, num_clust);
    // Now wait for completion
    barrier(wg_id, threadId, blocking, comm_ring);

    free(ranks);
    free(world_ranks);
    free(clust_req);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::fcollect(void *dst, void *src, int size, int wg_id,
                        MPI_Comm team, void *ata_buffptr,
                        ro_net_types type, int threadId, bool blocking)
{
    int pe_size, type_size;
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Datatype mpi_type = convertType(type);
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));

    // Currently GPU-centric algo only supports multiples of square root
    // TODO: Allow any size of cluster
    int num_clust = sqrt(pe_size);
    int clust_size = (pe_size + num_clust - 1) / num_clust;

    // In most cases the MPI implementation is optimal
    // But it crashes for > 512 messages
    if(size <= 512) {
        return fcollect_mpi(dst, src, size, wg_id, team,
                            ata_buffptr, type, threadId, blocking);
    }
    else if(num_clust * clust_size == pe_size) {
        return fcollect_gcen(dst, src, size, wg_id, team,
                             ata_buffptr, type, threadId, blocking);
    }
    else {
        return fcollect_broadcast(dst, src, size, wg_id, team,
                                  ata_buffptr, type, threadId, blocking);
    }
}

Status
MPITransport::fcollect_broadcast(void *dst, void *src, int size, int wg_id,
                        MPI_Comm team, void *ata_buffptr,
                        ro_net_types type, int threadId, bool blocking)
{
    // Broadcast implementation of fcollect
    auto *bp {backend_proxy->get()};
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Group grp, world_grp;
    NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
    NET_CHECK(MPI_Comm_group(comm, &grp));

    int grp_size;
    NET_CHECK(MPI_Group_size(grp, &grp_size));

    int *ranks = (int*)malloc(grp_size * sizeof(int));
    int *world_ranks = (int*)malloc(grp_size * sizeof(int));

    for (int i = 0; i < grp_size; i++)
        ranks[i] = i;
    // Convert comm ranks to global ranks for rput
    NET_CHECK(MPI_Group_translate_ranks(grp, grp_size, ranks,
        world_grp, world_ranks));

    int type_size;
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));

    MPI_Request *pe_req = (MPI_Request*)malloc(
        sizeof(MPI_Request) * pe_size);

    // Put data to all PEs
    for(int i = 0; i < pe_size; ++i) {
        int dst_offset = new_rank * type_size * size;
        NET_CHECK(MPI_Rput((char *)src, size,
            mpi_type, world_ranks[i], bp->heap_window_info[wg_id]->
            get_offset((char *)dst + dst_offset),
            size, mpi_type, bp->heap_window_info[wg_id]->get_win(),
            &pe_req[i]));
    }
    NET_CHECK(MPI_Waitall(pe_size, pe_req, MPI_STATUSES_IGNORE));
    NET_CHECK(MPI_Win_flush_all(bp->heap_window_info[wg_id]->get_win()));

    free(ranks);
    free(world_ranks);
    free(pe_req);
    // Now wait for completion
    barrier(wg_id, threadId, blocking, comm);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::fcollect_mpi(void *dst, void *src, int size, int wg_id,
                        MPI_Comm team, void *ata_buffptr,
                        ro_net_types type, int threadId, bool blocking)
{
    // MPI's implementation of fcollect
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));
    NET_CHECK(MPI_Allgather(src, size, mpi_type, dst,
                            size, mpi_type, comm));
    quiet(wg_id, threadId);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::fcollect_gcen(void *dst, void *src, int size, int wg_id,
                        MPI_Comm team, void *ata_buffptr,
                        ro_net_types type, int threadId, bool blocking)
{
    // GPU-centric implementation of fcollect
    auto *bp {backend_proxy->get()};
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Group grp, world_grp;
    NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
    NET_CHECK(MPI_Comm_group(comm, &grp));

    int grp_size;
    NET_CHECK(MPI_Group_size(grp, &grp_size));

    int *ranks = (int*)malloc(grp_size * sizeof(int));
    int *world_ranks = (int*)malloc(grp_size * sizeof(int));

    for (int i = 0; i < grp_size; i++)
        ranks[i] = i;
    // Convert comm ranks to global ranks for rput
    NET_CHECK(MPI_Group_translate_ranks(grp, grp_size, ranks,
        world_grp, world_ranks));

    int type_size;
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));

    // Works when number of PEs divisible by root(PE_size)
    int num_clust = sqrt(pe_size);
    int clust_size = (pe_size + num_clust - 1) / num_clust;
    // TODO: Allow any size of cluster
    assert(num_clust * clust_size == pe_size);
    int clust_id = new_rank / clust_size;

    if(MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
        fprintf(stderr, "Fcollect size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
        exit(-1);
    }

    MPI_Request *clust_req = new MPI_Request[pe_size];

    // Step 1: Send data to PEs in cluster
    for(int i = 0; i < clust_size; ++i) {
        int dst_offset = i * type_size * size;
        NET_CHECK(MPI_Rget((void*)((char *)ata_buffptr + dst_offset), size,
            mpi_type, world_ranks[clust_id * clust_size + (i % clust_size)],
            bp->heap_window_info[wg_id]->get_offset(src), size,
            mpi_type, bp->heap_window_info[wg_id]->get_win(), &clust_req[i]));
    }

    NET_CHECK(MPI_Waitall(clust_size, clust_req, MPI_STATUSES_IGNORE));

    // Step 2: Send final data to PEs outside cluster
    for(int i = 0; i < num_clust; ++i) {
        int src_offset = i * type_size * size * clust_size;
        int dst_offset = clust_id * type_size * size * clust_size;
        NET_CHECK(MPI_Put(ata_buffptr, size * clust_size, mpi_type,
            world_ranks[(new_rank % clust_size) + i * clust_size],
            bp->heap_window_info[wg_id]->get_offset((char *)dst + dst_offset),
            size * clust_size, mpi_type,
            bp->heap_window_info[wg_id]->get_win()));

        // Since MPI makes puts as complete as soon as the local buffer is free,
        // we need a flush to satisfy quiet.
        NET_CHECK(MPI_Win_flush(world_ranks[(new_rank % clust_size) +
                  i * clust_size], bp->heap_window_info[wg_id]->get_win()));
    }

    int stride = world_ranks[1] - world_ranks[0];
    MPI_Comm comm_cluster = createComm(world_ranks[clust_id * clust_size],
                            stride, clust_size);
    MPI_Comm comm_ring = createComm(world_ranks[new_rank % clust_size],
                          stride * clust_size, num_clust);
    // Now wait for completion
    barrier(wg_id, threadId, false, comm_cluster);
    barrier(wg_id, threadId, blocking, comm_ring);

    free(ranks);
    free(world_ranks);
    free(clust_req);
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::fcollect_gcen2(void *dst, void *src, int size, int wg_id,
                        MPI_Comm team, void *ata_buffptr,
                        ro_net_types type, int threadId, bool blocking)
{
    // GPU-centric implementation with in-place, blocking synchronization
    auto *bp {backend_proxy->get()};
    MPI_Request request;
    int new_rank, pe_size;

    MPI_Datatype mpi_type = convertType(type);
    MPI_Comm comm = team;
    NET_CHECK(MPI_Comm_rank(comm, &new_rank));
    NET_CHECK(MPI_Comm_size(comm, &pe_size));

    MPI_Group grp, world_grp;
    NET_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_grp));
    NET_CHECK(MPI_Comm_group(comm, &grp));

    int grp_size;
    NET_CHECK(MPI_Group_size(grp, &grp_size));

    int *ranks = (int*)malloc(grp_size * sizeof(int));
    int *world_ranks = (int*)malloc(grp_size * sizeof(int));

    for (int i = 0; i < grp_size; i++)
        ranks[i] = i;
    // Convert comm ranks to global ranks for rput
    NET_CHECK(MPI_Group_translate_ranks(grp, grp_size, ranks,
        world_grp, world_ranks));

    int type_size;
    NET_CHECK(MPI_Type_size(mpi_type, &type_size));

    // Works when number of PEs divisible by root(PE_size)
    int num_clust = sqrt(pe_size);
    int clust_size = (pe_size + num_clust - 1) / num_clust;
    // TODO: Allow any size of cluster
    assert(num_clust * clust_size == pe_size);
    int clust_id = new_rank / clust_size;

    if(MAX_ATA_BUFF_SIZE < type_size * size * pe_size) {
        fprintf(stderr, "Fcollect size %d exceeds max MAX_ATA_BUFF_SIZE %d\n",
            type_size * size * pe_size, MAX_ATA_BUFF_SIZE);
        exit(-1);
    }

    MPI_Request *clust_req = new MPI_Request[pe_size];

    // Step 1: Send data to PEs in cluster
    for(int i = 0; i < clust_size; ++i) {
        int dst_offset = i * type_size * size;
        NET_CHECK(MPI_Rget((void*)((char *)ata_buffptr + dst_offset), size,
            mpi_type, world_ranks[clust_id * clust_size + (i % clust_size)],
            bp->heap_window_info[wg_id]->get_offset(src), size,
            mpi_type, bp->heap_window_info[wg_id]->get_win(), &clust_req[i]));
    }

    NET_CHECK(MPI_Waitall(clust_size, clust_req, MPI_STATUSES_IGNORE));

    int stride = world_ranks[1] - world_ranks[0];
    MPI_Comm comm_cluster = createComm(world_ranks[clust_id * clust_size],
                                       stride, clust_size);
    MPI_Barrier(comm_cluster);

    // Step 2: Send final data to PEs outside cluster
    for(int i = 0; i < num_clust; ++i) {
        int src_offset = i * type_size * size * clust_size;
        int dst_offset = clust_id * type_size * size * clust_size;
        NET_CHECK(MPI_Put(ata_buffptr, size * clust_size, mpi_type,
            world_ranks[(new_rank % clust_size) + i * clust_size],
            bp->heap_window_info[wg_id]->get_offset((char *)dst + dst_offset),
            size * clust_size, mpi_type,
            bp->heap_window_info[wg_id]->get_win()));

        // Since MPI makes puts as complete as soon as the local buffer is free,
        // we need a flush to satisfy quiet.
        NET_CHECK(MPI_Win_flush(world_ranks[(new_rank % clust_size) +
                  i * clust_size], bp->heap_window_info[wg_id]->get_win()));
    }

    MPI_Comm comm_ring = createComm(world_ranks[new_rank % clust_size],
                                    stride * clust_size, num_clust);
    // Now wait for completion
    barrier(wg_id, threadId, blocking, comm_ring);

    free(ranks);
    free(world_ranks);
    free(clust_req);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::putMem(void *dst,
                     void *src,
                     int size,
                     int pe,
                     int wg_id,
                     int threadId,
                     bool blocking,
                     bool inline_data) {
    auto *bp {backend_proxy->get()};

    if (!bp->gpu_queue) {
        // Need to flush HDP read cache so that the NIC can see data to push
        // out to the network.  If we have the network buffers allocated
        // on the host or we've already flushed for the command queue on the
        // GPU then we can ignore this step.
        bp->hdp_policy->hdp_flush();
    }

    MPI_Request request {};
    NET_CHECK(MPI_Rput(src,
                       size,
                       MPI_CHAR,
                       pe,
                       bp->heap_window_info[wg_id]->get_offset(dst),
                       size,
                       MPI_CHAR,
                       bp->heap_window_info[wg_id]->get_win(),
                       &request));

    // Since MPI makes puts as complete as soon as the local buffer is free,
    // we need a flush to satisfy quiet.  Put it here as a hack for now even
    // though it should be in the progress loop.
    NET_CHECK(MPI_Win_flush_all(bp->heap_window_info[wg_id]->get_win()));

    req_prop_vec.emplace_back(threadId, wg_id, blocking, src, inline_data);
    req_vec.push_back(request);
    outstanding[wg_id]++;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::amoFOP(void *dst,
                     void *src,
                     int64_t val,
                     int pe,
                     int wg_id,
                     int threadId,
                     bool blocking,
                     ROC_SHMEM_OP op) {
    auto *bp {backend_proxy->get()};

    if (!bp->gpu_queue) {
        // Need to flush HDP read cache so that the NIC can see data to push
        // out to the network.  If we have the network buffers allocated
        // on the host or we've already flushed for the command queue on the
        // GPU then we can ignore this step.
        bp->hdp_policy->hdp_flush();
    }

    NET_CHECK(MPI_Fetch_and_op((void*)&val,
                               src,
                               MPI_INT64_T,
                               pe,
                               bp->heap_window_info[wg_id]->get_offset(dst),
                               get_mpi_op(op),
                               bp->heap_window_info[wg_id]->get_win()));

    // Since MPI makes puts as complete as soon as the local buffer is free,
    // we need a flush to satisfy quiet.  Put it here as a hack for now even
    // though it should be in the progress loop.
    NET_CHECK(MPI_Win_flush_local(pe, bp->heap_window_info[wg_id]->get_win()));

    bp->queue_descs[wg_id].status[threadId] = 1;
    if (bp->gpu_queue) {
        SFENCE();
        bp->hdp_policy->hdp_flush();
    }

    //MPI_Request request;
    //req_prop_vec.emplace_back(threadId, wg_id, blocking, src, inline_data);
    //req_vec.push_back(request);
    //outstanding[wg_id]++;
    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::amoFCAS(void *dst,
                      void *src,
                      int64_t val,
                      int pe,
                      int wg_id,
                      int threadId,
                      bool blocking,
                      int64_t cond) {
    auto *bp {backend_proxy->get()};

    if (!bp->gpu_queue) {
        // Need to flush HDP read cache so that the NIC can see data to push
        // out to the network.  If we have the network buffers allocated
        // on the host or we've already flushed for the command queue on the
        // GPU then we can ignore this step.
        bp->hdp_policy->hdp_flush();
    }

    NET_CHECK(MPI_Compare_and_swap((const void*)&val,
                                   (const void*)&cond,
                                   src,
                                   MPI_INT64_T,
                                   pe,
                                   bp->heap_window_info[wg_id]->get_offset(dst),
                                   bp->heap_window_info[wg_id]->get_win()));

    // Since MPI makes puts as complete as soon as the local buffer is free,
    // we need a flush to satisfy quiet.  Put it here as a hack for now even
    // though it should be in the progress loop.
    NET_CHECK(MPI_Win_flush_local(pe, bp->heap_window_info[wg_id]->get_win()));

    bp->queue_descs[wg_id].status[threadId] = 1;
    if (bp->gpu_queue) {
        SFENCE();
        bp->hdp_policy->hdp_flush();
    }

    return Status::ROC_SHMEM_SUCCESS;
}


Status
MPITransport::getMem(void *dst,
                     void *src,
                     int size,
                     int pe,
                     int wg_id,
                     int threadId,
                     bool blocking) {
    auto *bp {backend_proxy->get()};

    outstanding[wg_id]++;

    MPI_Request request {};
    NET_CHECK(MPI_Rget(dst,
                       size,
                       MPI_CHAR,
                       pe,
                       bp->heap_window_info[wg_id]->get_offset(src),
                       size,
                       MPI_CHAR,
                       bp->heap_window_info[wg_id]->get_win(),
                       &request));

   req_prop_vec.emplace_back(threadId, wg_id, blocking);
   req_vec.push_back(request);

    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::progress() {
    MPI_Status status {};
    int flag {0};
    auto *bp {backend_proxy->get()};

    DPRINTF("Entering progress engine\n");

    // Enter the progress engine if there aren't any pending requests
    if (req_vec.size() == 0) {
        DPRINTF("Probing MPI\n");

        NET_CHECK(MPI_Iprobe(num_pes - 1,
                             1000,
                             ro_net_comm_world,
                             &flag,
                             &status));
    } else {
        DPRINTF("Testing all outstanding requests (%zu\n)",
                req_vec.size());

        // Check completion of any outstanding requests. We check on either
        // the first INDICES_SIZE requests or the size of the request vector
        auto req_vec_size {static_cast<int>(req_vec.size())};
        int incount {(req_vec_size < INDICES_SIZE) ?
                     req_vec_size :
                     INDICES_SIZE};
        int outcount {};
        NET_CHECK(MPI_Testsome(incount,
                               req_vec.data(),
                               &outcount,
                               indices,
                               MPI_STATUSES_IGNORE));

        // If any request completed remove it from outstanding request vector
        for (int i {0}; i < outcount; i++) {
            int indx {indices[i]};
            int wg_id {req_prop_vec[indx].wgId};
            int threadId {req_prop_vec[indx].threadId};

            if (wg_id != -1) {
                outstanding[wg_id]--;
                DPRINTF("Finished op for wg_id %d at threadId %d "
                        "(%d requests outstanding)\n",
                        wg_id, threadId, outstanding[wg_id]);
            } else {
                DPRINTF("Finished host barrier\n");
                hostBarrierDone = 1;
            }

            if (req_prop_vec[indx].blocking) {
                if (wg_id != -1) {
                    bp->queue_descs[wg_id].status[threadId] = 1;
                }
                if (bp->gpu_queue) {
                    SFENCE();
                    bp->hdp_policy->hdp_flush();
                }
            }

            if (req_prop_vec[indx].inline_data) {
                free(req_prop_vec[indx].src);
            }

            // If the GPU has requested a quiet, notify it of completion when
            // all outstanding requests are complete.
            if (!outstanding[wg_id] && !waiting_quiet[wg_id].empty()) {
                for (const auto threadId : waiting_quiet[wg_id]) {
                    DPRINTF("Finished Quiet for wg_id %d at threadId %d\n",
                            wg_id, threadId);
                    bp->queue_descs[wg_id].status[threadId] = 1;
                }

                waiting_quiet[wg_id].clear();

                if (bp->gpu_queue) {
                    SFENCE();
                    bp->hdp_policy->hdp_flush();
                }
            }
        }

        // Remove the MPI Request and the RequestProperty tracking entry
        sort(indices, indices + outcount, std::greater<int>());
        for (int i {0}; i < outcount; i++) {
            int indx {indices[i]};
            req_vec.erase(req_vec.begin() + indx);
            req_prop_vec.erase(req_prop_vec.begin() + indx);
        }
    }

    return Status::ROC_SHMEM_SUCCESS;
}

Status
MPITransport::quiet(int wg_id, int threadId) {
    auto *bp {backend_proxy->get()};

    if (!outstanding[wg_id]) {
        DPRINTF("Finished Quiet immediately for wg_id %d at threadId %d\n",
                wg_id, threadId);
        bp->queue_descs[wg_id].status[threadId] = 1;
    } else {
        waiting_quiet[wg_id].emplace_back(threadId);
    }
    return Status::ROC_SHMEM_SUCCESS;
}

int
MPITransport::numOutstandingRequests() {
    return req_vec.size() + q.size();
}

}  // namespace rocshmem
