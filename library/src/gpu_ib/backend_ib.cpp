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

#include "src/gpu_ib/backend_ib.hpp"

#include <endian.h>
#include <mpi.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>  // NOLINT(build/c++11)

#include "include/roc_shmem.hpp"
#include "src/backend_type.hpp"
#include "src/context_incl.hpp"
#include "src/gpu_ib/gpu_ib_team.hpp"
#include "src/gpu_ib/queue_pair.hpp"
#include "src/host/host.hpp"

namespace rocshmem {

#define NET_CHECK(cmd)                                       \
  {                                                          \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort();                                               \
    }                                                        \
  }

extern roc_shmem_ctx_t ROC_SHMEM_HOST_CTX_DEFAULT;

roc_shmem_team_t get_external_team(GPUIBTeam *team) {
  return reinterpret_cast<roc_shmem_team_t>(team);
}

int get_ls_non_zero_bit(char *bitmask, int mask_length) {
  int position = -1;

  for (int bit_i = 0; bit_i < mask_length; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;
    if (bitmask[byte_i] & (1 << (bit_i % CHAR_BIT))) {
      position = bit_i;
      break;
    }
  }

  return position;
}

GPUIBBackend::GPUIBBackend(MPI_Comm comm) : Backend() {
  if (auto num_blocks_cstr = getenv("ROC_SHMEM_NUM_BLOCKS")) {
    std::stringstream sstream(num_blocks_cstr);
    sstream >> num_blocks_;
  }

  init_mpi_once(comm);

  type = BackendType::GPU_IB_BACKEND;

  NET_CHECK(MPI_Comm_dup(backend_comm, &gpu_ib_comm_world));
  NET_CHECK(MPI_Comm_size(gpu_ib_comm_world, &num_pes));
  NET_CHECK(MPI_Comm_rank(gpu_ib_comm_world, &my_pe));

  /* Initialize the host interface */
  host_interface =
      new HostInterface(hdp_proxy_.get(), gpu_ib_comm_world, &heap);

  /*
   * Construct default host context independently of the
   * default device context (done in the async thread)
   * so that host operations can execute regardless of
   * device operations.
   */
  setup_default_host_ctx();

  setup_team_world();

  roc_shmem_collective_init();

  teams_init();

  // MPI_Comm_dup(gpu_ib_comm_world, &thread_comm);
  thread_comm = gpu_ib_comm_world;

  NET_CHECK(MPI_Barrier(gpu_ib_comm_world));

  // commenting out the async  thread as there is some issues with ROCm
  // this makes the CPU init blocking
  // async_thread_ = thread_spawn(this);
  thread_func_internal(this);
}

__device__ void GPUIBBackend::create_ctx(int64_t options, roc_shmem_ctx_t *ctx) {
  GPUIBContext *ctx_;
  int idx = (atomicAdd(&ctx_counter, 1)) % num_blocks_;
  ctx_ = &ctx_array[idx];
  ctx_->dev_mtx_.shareable_ = static_cast<bool>(options >> 3);
  ctx->ctx_opaque = ctx_;
}

void GPUIBBackend::ctx_create(int64_t options, void **ctx) {
  GPUIBHostContext *new_ctx = nullptr;

  new_ctx = new GPUIBHostContext(this, options);

  *ctx = new_ctx;
}

GPUIBHostContext *get_internal_gpu_ib_ctx(Context *ctx) {
  return reinterpret_cast<GPUIBHostContext *>(ctx);
}

void GPUIBBackend::ctx_destroy(Context *ctx) {
  GPUIBHostContext *gpu_ib_host_ctx = get_internal_gpu_ib_ctx(ctx);
  delete gpu_ib_host_ctx;
}

GPUIBBackend::~GPUIBBackend() {
  // need to get thi sback once ROCm is fixed
  // async_thread_.join();

  /**
   * Destroy teams infrastructure
   * and team world
   */
  teams_destroy();
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  CHECK_HIP(hipFree(team_world));

  delete default_host_ctx_;

  NET_CHECK(MPI_Comm_free(&gpu_ib_comm_world));

  CHECK_HIP(hipFree(default_ctx_->device_qp_proxy));
  CHECK_HIP(hipFree(default_ctx_));
  default_ctx_ = nullptr;

  delete host_interface;
  host_interface = nullptr;

  networkImpl.networkHostFinalize();
}

__host__ void GPUIBBackend::global_exit(int status) {
  MPI_Abort(gpu_ib_comm_world, status);
}

void GPUIBBackend::create_new_team([[maybe_unused]] Team *parent_team,
                                     TeamInfo *team_info_wrt_parent,
                                     TeamInfo *team_info_wrt_world, int num_pes,
                                     int my_pe_in_new_team, MPI_Comm team_comm,
                                     roc_shmem_team_t *new_team) {
  /**
   * Read the bit mask and find out a common index into
   * the pool of available work arrays.
   */
  NET_CHECK(MPI_Allreduce(pool_bitmask_, reduced_bitmask_, bitmask_size_,
                          MPI_CHAR, MPI_BAND, team_comm));

  /* Pick the least significant non-zero bit (logical layout) in the reduced
   * bitmask */
  auto max_num_teams{team_tracker.get_max_num_teams()};
  int common_index = get_ls_non_zero_bit(reduced_bitmask_, max_num_teams);
  if (common_index < 0) {
    /* No team available */
    abort();
  }

  /* Mark the team as taken (by unsetting the bit in the pool bitmask) */
  int byte = common_index / CHAR_BIT;
  pool_bitmask_[byte] &= ~(1 << (common_index % CHAR_BIT));

  /**
   * Allocate device-side memory for team_world and
   * construct a GPU_IB team in it
   */
  GPUIBTeam *new_team_obj;
  CHECK_HIP(hipMalloc(&new_team_obj, sizeof(GPUIBTeam)));
  new (new_team_obj)
      GPUIBTeam(this, team_info_wrt_parent, team_info_wrt_world, num_pes,
                my_pe_in_new_team, team_comm, common_index);

  *new_team = get_external_team(new_team_obj);
}

void GPUIBBackend::team_destroy(roc_shmem_team_t team) {
  GPUIBTeam *team_obj = get_internal_gpu_ib_team(team);

  /* Mark the pool as available */
  int bit = team_obj->pool_index_;
  int byte_i = bit / CHAR_BIT;
  pool_bitmask_[byte_i] |= 1 << (bit % CHAR_BIT);

  team_obj->~GPUIBTeam();
  CHECK_HIP(hipFree(team_obj));
}

void GPUIBBackend::dump_backend_stats() {
  networkImpl.dump_backend_stats(&globalStats);
}

void GPUIBBackend::reset_backend_stats() {
  networkImpl.reset_backend_stats();
}

void GPUIBBackend::initialize_ipc() {
  ipcImpl.ipcHostInit(my_pe, heap.get_heap_bases(), thread_comm);
}

void GPUIBBackend::initialize_network() {
  networkImpl.networkHostSetup(this);
}

void GPUIBBackend::setup_default_host_ctx() {
  default_host_ctx_ = new GPUIBHostContext(this, 0);
  ROC_SHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx_;
}

void GPUIBBackend::setup_ctxs() {
  /*
   * Allocate device-side memory for all context and construct an
   * InfiniBand context in it.
   */
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(GPUIBContext) * num_blocks_));
  for (int i = 0; i < num_blocks_; i++)
    new (&ctx_array[i]) GPUIBContext(this, false, i);
}

void GPUIBBackend::setup_default_ctx() {
  /*
   * Allocate device-side memory for default context and construct an
   * InfiniBand context in it.
   */
  CHECK_HIP(hipMalloc(&default_ctx_, sizeof(GPUIBContext)));
  new (default_ctx_) GPUIBContext(this, true, 0);

  /*
   * Set the ROC_SHMEM_CTX_DEFAULT in constant memory.
   */
  int *symbol_address;
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void **>(&symbol_address),
                                HIP_SYMBOL(ROC_SHMEM_CTX_DEFAULT)));

  roc_shmem_ctx_t ctx_default_host{default_ctx_, nullptr};

  hipStream_t stream;
  CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  CHECK_HIP(hipMemcpyAsync(symbol_address, &ctx_default_host,
                           sizeof(roc_shmem_ctx_t), hipMemcpyDefault, stream));
  CHECK_HIP(hipStreamSynchronize(stream));
  CHECK_HIP(hipStreamDestroy(stream));
}

void GPUIBBackend::setup_team_world() {
  TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

  /**
   * Allocate device-side memory for team_world and construct a
   * GPU_IB team in it.
   */
  CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
  CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));

  new (team_info_wrt_parent) TeamInfo(nullptr, 0, 1, num_pes);
  new (team_info_wrt_world) TeamInfo(nullptr, 0, 1, num_pes);

  MPI_Comm team_world_comm;
  NET_CHECK(MPI_Comm_dup(gpu_ib_comm_world, &team_world_comm));

  GPUIBTeam *team_world{nullptr};
  CHECK_HIP(hipMalloc(&team_world, sizeof(GPUIBTeam)));
  new (team_world) GPUIBTeam(this, team_info_wrt_parent, team_info_wrt_world,
                             num_pes, my_pe, team_world_comm, 0);
  team_tracker.set_team_world(team_world);

  /**
   * Copy the address to ROC_SHMEM_TEAM_WORLD.
   */
  ROC_SHMEM_TEAM_WORLD = reinterpret_cast<roc_shmem_team_t>(team_world);
}

void GPUIBBackend::init_mpi_once(MPI_Comm comm) {
  static std::mutex init_mutex;
  const std::lock_guard<std::mutex> lock(init_mutex);

  int init_done = 0;
  NET_CHECK(MPI_Initialized(&init_done));
  if (init_done == 0) {
    int provided;
    NET_CHECK(
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
  }
  if (comm == MPI_COMM_NULL) {
    NET_CHECK(MPI_Comm_dup(MPI_COMM_WORLD, &backend_comm));
  } else {
    NET_CHECK(MPI_Comm_dup(comm, &backend_comm));
  }
}

std::thread GPUIBBackend::thread_spawn(GPUIBBackend *b) {
  return std::thread(&GPUIBBackend::thread_func_internal, this, b);
}

void GPUIBBackend::thread_func_internal(GPUIBBackend *b) {
  CHECK_HIP(hipSetDevice(hip_dev_id));

  b->initialize_ipc();
  b->initialize_network();
  b->setup_ctxs();
  b->setup_default_ctx();
  *(b->done_init) = 1;
}

void GPUIBBackend::teams_init() {
  /**
   * Allocate pools for the teams sync and work arrary from the SHEAP.
   */
  auto max_num_teams{team_tracker.get_max_num_teams()};
  barrier_pSync_pool = reinterpret_cast<long *>(roc_shmem_malloc(
      sizeof(long) * ROC_SHMEM_BARRIER_SYNC_SIZE * max_num_teams));
  reduce_pSync_pool = reinterpret_cast<long *>(roc_shmem_malloc(
      sizeof(long) * ROC_SHMEM_REDUCE_SYNC_SIZE * max_num_teams));
  bcast_pSync_pool = reinterpret_cast<long *>(roc_shmem_malloc(
      sizeof(long) * ROC_SHMEM_BCAST_SYNC_SIZE * max_num_teams));
  alltoall_pSync_pool = reinterpret_cast<long *>(roc_shmem_malloc(
      sizeof(long) * ROC_SHMEM_ALLTOALL_SYNC_SIZE * max_num_teams));

  /* Accommodating for largest possible data type for pWrk */
  pWrk_pool = roc_shmem_malloc(
      sizeof(double) * ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE * max_num_teams);
  pAta_pool = roc_shmem_malloc(sizeof(double) * ROC_SHMEM_ATA_MAX_WRKDATA_SIZE *
                               max_num_teams);

  /**
   * Initialize the sync arrays in the pool with default values.
   */
  long *barrier_pSync, *reduce_pSync, *bcast_pSync, *alltoall_pSync;
  for (int team_i = 0; team_i < max_num_teams; team_i++) {
    barrier_pSync = reinterpret_cast<long *>(
        &barrier_pSync_pool[team_i * ROC_SHMEM_BARRIER_SYNC_SIZE]);
    reduce_pSync = reinterpret_cast<long *>(
        &reduce_pSync_pool[team_i * ROC_SHMEM_REDUCE_SYNC_SIZE]);
    bcast_pSync = reinterpret_cast<long *>(
        &bcast_pSync_pool[team_i * ROC_SHMEM_BCAST_SYNC_SIZE]);
    alltoall_pSync = reinterpret_cast<long *>(
        &alltoall_pSync_pool[team_i * ROC_SHMEM_ALLTOALL_SYNC_SIZE]);

    for (int i = 0; i < ROC_SHMEM_BARRIER_SYNC_SIZE; i++) {
      barrier_pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
    for (int i = 0; i < ROC_SHMEM_REDUCE_SYNC_SIZE; i++) {
      reduce_pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
    for (int i = 0; i < ROC_SHMEM_BCAST_SYNC_SIZE; i++) {
      bcast_pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
    for (int i = 0; i < ROC_SHMEM_ALLTOALL_SYNC_SIZE; i++) {
      alltoall_pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
  }

  /**
   * Initialize bit mask
   *
   * Logical:
   * MSB..........................................................................LSB
   * Physical: MSB...1st least significant 8 bits...LSB  MSB...2nd least
   * signifant 8 bits...LSB
   *
   * Description shows only a 2-byte long mask but idea extends to any arbitrary
   * size.
   */
  bitmask_size_ = (max_num_teams % CHAR_BIT) ? (max_num_teams / CHAR_BIT + 1)
                                             : (max_num_teams / CHAR_BIT);
  pool_bitmask_ = reinterpret_cast<char *>(malloc(bitmask_size_));
  reduced_bitmask_ = reinterpret_cast<char *>(malloc(bitmask_size_));

  memset(pool_bitmask_, 0, bitmask_size_);
  memset(reduced_bitmask_, 0, bitmask_size_);
  /* Set all to available except the 0th one (reserved for TEAM_WORLD) */
  for (int bit_i = 1; bit_i < max_num_teams; bit_i++) {
    int byte_i = bit_i / CHAR_BIT;

    pool_bitmask_[byte_i] |= 1 << (bit_i % CHAR_BIT);
  }

  /**
   * Make sure that all processing elements have done this before
   * continuing.
   */
  NET_CHECK(MPI_Barrier(gpu_ib_comm_world));
}

void GPUIBBackend::teams_destroy() {
  roc_shmem_free(barrier_pSync_pool);
  roc_shmem_free(reduce_pSync_pool);
  roc_shmem_free(bcast_pSync_pool);
  roc_shmem_free(alltoall_pSync_pool);
  roc_shmem_free(pWrk_pool);
  roc_shmem_free(pAta_pool);

  free(pool_bitmask_);
  free(reduced_bitmask_);
}

void GPUIBBackend::roc_shmem_collective_init() {
  /*
   * Allocate heap space for barrier_sync
   */
  size_t one_sync_size_bytes{sizeof(*barrier_sync)};
  size_t sync_size_bytes{one_sync_size_bytes * ROC_SHMEM_BARRIER_SYNC_SIZE};
  heap.malloc(reinterpret_cast<void **>(&barrier_sync), sync_size_bytes);

  /*
   * Initialize the barrier synchronization array with default values.
   */
  for (int i = 0; i < num_pes; i++) {
    barrier_sync[i] = ROC_SHMEM_SYNC_VALUE;
  }

  /*
   * Make sure that all processing elements have done this before
   * continuing.
   */
  NET_CHECK(MPI_Barrier(gpu_ib_comm_world));
}

}  // namespace rocshmem
