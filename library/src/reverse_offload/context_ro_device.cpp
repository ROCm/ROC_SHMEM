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

#include "src/reverse_offload/context_ro_device.hpp"

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_device_functions.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "config.h"  // NOLINT(build/include_subdir)
#include "include/roc_shmem.hpp"
#include "src/backend_type.hpp"
#include "src/hdp_policy.hpp"
#include "src/reverse_offload/backend_proxy.hpp"
#include "src/reverse_offload/backend_ro.hpp"
#include "src/reverse_offload/ro_net_team.hpp"
#include "src/sync/abql_block_mutex.hpp"

namespace rocshmem {

__host__ ROContext::ROContext(Backend *b, size_t block_id)
    : Context(b, false) {
  ROBackend *backend{static_cast<ROBackend *>(b)};
  if (block_id == -1) {
    block_handle = backend->default_block_handle_proxy_.get();
  } else {
    auto block_base{backend->block_handle_proxy_.get()};
    block_handle = &block_base[block_id];
  }
  ro_net_win_id = block_id % backend->ro_window_proxy_->MAX_NUM_WINDOWS;
}

__device__ void ROContext::threadfence_system() {
  int thread_id = get_flat_block_id();
  if (thread_id % WF_SIZE == lowerID()) {
    block_handle->hdp.flushCoherency();
  }
  __threadfence_system();
}

__device__ void ROContext::putmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset,
                     const_cast<void *>(source), nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_PUT, dest, const_cast<void *>(source), nelems,
                        pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                        ro_net_win_id, block_handle, true);
  }
}

__device__ void ROContext::getmem(void *dest, const void *source, size_t nelems,
                                  int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_GET, dest, const_cast<void *>(source), nelems,
                        pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                        ro_net_win_id, block_handle, true);
  }
}

__device__ void ROContext::putmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[local_pe] + L_offset,
                     const_cast<void *>(source), nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_PUT_NBI, dest, const_cast<void *>(source),
                        nelems, pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                        ro_net_win_id, block_handle, false);
  }
}

__device__ void ROContext::getmem_nbi(void *dest, const void *source,
                                      size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    bool must_send_message = wf_coal_.coalesce(pe, source, dest, &nelems);
    if (!must_send_message) {
      return;
    }
    build_queue_element(RO_NET_GET_NBI, dest, const_cast<void *>(source),
                        nelems, pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                        ro_net_win_id, block_handle, false);
  }
}

__device__ void ROContext::fence() {
  build_queue_element(RO_NET_FENCE, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, MPI_COMM_NULL, ro_net_win_id, block_handle, true);
}

__device__ void ROContext::fence(int pe) {
  // TODO(khamidou): need to check if per pe has any special handling
  build_queue_element(RO_NET_FENCE, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, MPI_COMM_NULL, ro_net_win_id, block_handle, true);
}

__device__ void ROContext::quiet() {
  build_queue_element(RO_NET_QUIET, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                      nullptr, MPI_COMM_NULL, ro_net_win_id, block_handle, true);
}

__device__ void *ROContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    void *dst = const_cast<void *>(dest);
    uint64_t L_offset =
        reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[my_pe];
    ret = ipcImpl_.ipc_bases[pe] + L_offset;
  }
  return ret;
}

__device__ void ROContext::barrier_all() {
  if (is_thread_zero_in_block()) {
    build_queue_element(RO_NET_BARRIER_ALL, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, MPI_COMM_NULL, ro_net_win_id,
                        block_handle, true);
  }
  __syncthreads();
}

__device__ void ROContext::sync_all() {
  if (is_thread_zero_in_block()) {
    build_queue_element(RO_NET_BARRIER_ALL, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, MPI_COMM_NULL, ro_net_win_id,
                        block_handle, true);
  }
  __syncthreads();
}

__device__ void ROContext::sync(roc_shmem_team_t team) {
  ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);
  if (is_thread_zero_in_block()) {
    build_queue_element(RO_NET_SYNC, nullptr, nullptr, 0, 0, 0, 0, 0, nullptr,
                        nullptr, team_obj->mpi_comm, ro_net_win_id, block_handle,
                        true);
  }
  __syncthreads();
}

__device__ void ROContext::ctx_destroy() {
  if (is_thread_zero_in_block()) {
    ROBackend *backend{static_cast<ROBackend *>(device_backend_proxy)};
    BackendProxyT &backend_proxy{backend->backend_proxy};
    auto *proxy{backend_proxy.get()};

    build_queue_element(RO_NET_FINALIZE, nullptr, nullptr, 0, 0, 0, 0, 0,
                        nullptr, nullptr, MPI_COMM_NULL, ro_net_win_id,
                        block_handle, true);

    int buffer_id = ro_net_win_id;
    backend->queue_.descriptor(buffer_id)->write_index = block_handle->write_index;

    ROStats &global_handle = proxy->profiler[buffer_id];
    global_handle.accumulateStats(block_handle->profiler);
  }

  __syncthreads();
}

__device__ void ROContext::putmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset,
                        const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_PUT, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, true);
    }
  }
  __syncthreads();
}

__device__ void ROContext::getmem_wg(void *dest, const void *source,
                                     size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_GET, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, true);
    }
  }
  __syncthreads();
}

__device__ void ROContext::putmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wg(ipcImpl_.ipc_bases[local_pe] + L_offset,
                        const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_PUT_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
  __syncthreads();
}

__device__ void ROContext::getmem_nbi_wg(void *dest, const void *source,
                                         size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wg(dest, ipcImpl_.ipc_bases[local_pe] + L_offset, nelems);
  } else {
    if (is_thread_zero_in_block()) {
      build_queue_element(RO_NET_GET_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
  __syncthreads();
}

__device__ void ROContext::putmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset,
                          const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_PUT, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, true);
    }
  }
}

__device__ void ROContext::getmem_wave(void *dest, const void *source,
                                       size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset,
                          nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_GET, dest, const_cast<void *>(source), nelems,
                          pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, true);
    }
  }
}

__device__ void ROContext::putmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    uint64_t L_offset =
        reinterpret_cast<char *>(dest) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wave(ipcImpl_.ipc_bases[local_pe] + L_offset,
                          const_cast<void *>(source), nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_PUT_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
}

__device__ void ROContext::getmem_nbi_wave(void *dest, const void *source,
                                           size_t nelems, int pe) {
  if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
    int local_pe = pe % ipcImpl_.shm_size;
    const char *src_typed = reinterpret_cast<const char *>(source);
    uint64_t L_offset =
        const_cast<char *>(src_typed) - ipcImpl_.ipc_bases[my_pe];
    ipcImpl_.ipcCopy_wave(dest, ipcImpl_.ipc_bases[local_pe] + L_offset,
                          nelems);
  } else {
    if (is_thread_zero_in_wave()) {
      build_queue_element(RO_NET_GET_NBI, dest, const_cast<void *>(source),
                          nelems, pe, 0, 0, 0, nullptr, nullptr, MPI_COMM_NULL,
                          ro_net_win_id, block_handle, false);
    }
  }
}

__device__ uint64_t number_active_lanes() {
  return __popcll(__ballot(1));
}

__device__ uint64_t active_logical_lane_id() {
  uint64_t ballot{__ballot(1)};
  uint64_t my_physical_lane_id{__lane_id()};
  uint64_t all_ones_mask = -1;
  uint64_t lane_mask{all_ones_mask << my_physical_lane_id};
  uint64_t inverted_mask{~lane_mask};
  uint64_t lower_active_lanes{ballot & inverted_mask};
  uint64_t my_logical_lane_id{__popcll(lower_active_lanes)};
  return my_logical_lane_id;
}

__device__ uint64_t broadcast_up(uint64_t value) {
  for (unsigned i{0}; i < __AMDGCN_WAVEFRONT_SIZE; i++) {
    uint64_t temp{__shfl_up(value, i)};
    if (temp) {
      value = temp;
    }
  }
  return value;
}

__device__ bool enough_space(BlockHandle *h, uint64_t required) {
  return (h->queue_size - (h->write_index - h->read_index)) >= required;
}

__device__ void refresh_volatile(BlockHandle *handle) {
  __asm__ volatile(
    "global_load_dwordx2 %0 %1 off glc slc\n "
    "s_waitcnt vmcnt(0)"
    : "=v"(handle->read_index)
    : "v"(handle->host_read_index));
}

__device__ void acquire_lock(BlockHandle *handle) {
  while(atomicCAS((uint64_t *)&handle->lock, 0, 1) == 1) ;
}

__device__ void release_lock(BlockHandle *handle) {
  handle->lock = 0;
}

__device__ void wait_until_space_available(BlockHandle *handle, uint64_t required) {
  while (!enough_space(handle, required)) {
    refresh_volatile(handle);
  }
}

__device__ uint64_t next_write_slot(BlockHandle *handle) {
  auto num_active_lanes{number_active_lanes()};
  uint64_t write_slot{0};
  auto my_active_lane_id {active_logical_lane_id()};
  bool is_lowest_active_lane {my_active_lane_id == 0};
  if (is_lowest_active_lane) {
    acquire_lock(handle);
    wait_until_space_available(handle, num_active_lanes);
    write_slot = handle->write_index;
    handle->write_index += num_active_lanes;
    __threadfence();
    release_lock(handle);
    __threadfence();
  }
  write_slot = broadcast_up(write_slot);
  write_slot += my_active_lane_id;
  return write_slot % handle->queue_size;
}

__device__ void build_queue_element(
    ro_net_cmds type, void *dst, void *src, size_t size, int pe,
    int logPE_stride, int PE_size, int PE_root, void *pWrk, long *pSync,
    MPI_Comm team_comm, int ro_net_win_id, BlockHandle *handle,
    bool blocking, ROC_SHMEM_OP op, ro_net_types datatype) {
  auto write_slot{next_write_slot(handle)};

  handle->queue[write_slot].type = type;
  handle->queue[write_slot].PE = pe;
  handle->queue[write_slot].ol1.size = size;
  handle->queue[write_slot].dst = dst;
  handle->queue[write_slot].ro_net_win_id = ro_net_win_id;

  if (type == RO_NET_P) {
    memcpy(&handle->queue[write_slot].src, src, size);
  } else {
    handle->queue[write_slot].src = src;
  }

  auto threadId {get_flat_id()};
  handle->queue[write_slot].threadId = threadId;

  if (type == RO_NET_AMO_FOP) {
    handle->queue[write_slot].op = op;
    handle->queue[write_slot].datatype = datatype;
  }
  if (type == RO_NET_AMO_FCAS) {
    handle->queue[write_slot].ol2.pWrk = pWrk;
    handle->queue[write_slot].datatype = datatype;
  }
  if (type == RO_NET_TO_ALL) {
    handle->queue[write_slot].logPE_stride = logPE_stride;
    handle->queue[write_slot].PE_size = PE_size;
    handle->queue[write_slot].ol2.pWrk = pWrk;
    handle->queue[write_slot].pSync = pSync;
    handle->queue[write_slot].op = op;
    handle->queue[write_slot].datatype = datatype;
  }
  if (type == RO_NET_TEAM_TO_ALL) {
    handle->queue[write_slot].op = op;
    handle->queue[write_slot].datatype = datatype;
    handle->queue[write_slot].team_comm = team_comm;
  }
  if (type == RO_NET_BROADCAST) {
    handle->queue[write_slot].logPE_stride = logPE_stride;
    handle->queue[write_slot].PE_size = PE_size;
    handle->queue[write_slot].pSync = pSync;
    handle->queue[write_slot].PE_root = PE_root;
    handle->queue[write_slot].datatype = datatype;
  }
  if (type == RO_NET_TEAM_BROADCAST) {
    handle->queue[write_slot].PE_root = PE_root;
    handle->queue[write_slot].datatype = datatype;
    handle->queue[write_slot].team_comm = team_comm;
  }
  if (type == RO_NET_ALLTOALL) {
    handle->queue[write_slot].datatype = datatype;
    handle->queue[write_slot].team_comm = team_comm;
    handle->queue[write_slot].ol2.pWrk = pWrk;
  }
  if (type == RO_NET_FCOLLECT) {
    handle->queue[write_slot].datatype = datatype;
    handle->queue[write_slot].team_comm = team_comm;
    handle->queue[write_slot].ol2.pWrk = pWrk;
  }
  if (type == RO_NET_SYNC) {
    handle->queue[write_slot].team_comm = team_comm;
  }

  // Make sure queue element data is visible to CPU
  __threadfence();

  // Make data as ready and make visible to CPU
  handle->queue[write_slot].notify_cpu.valid = 1;
  __threadfence();

  // Blocking requires the CPU to complete the operation.
  if (blocking) {
    int net_status = 0;
    do {
      __asm__ volatile(
          "s_sleep 32\n"
          "global_load_sbyte %0 %1 off glc slc\n "
          "s_waitcnt vmcnt(0)"
          : "=v"(net_status)
          : "v"(&handle->status[threadId]));
    } while (net_status == 0);

    handle->status[threadId] = 0;
    __threadfence();
  }
}

__device__ uint64_t *ROContext::get_unused_atomic() {
  auto index{atomicAdd(&block_handle->atomic_ret.atomic_counter, 1)};
  index = index % max_nb_atomic;
  auto atomic_base_ptr{block_handle->atomic_ret.atomic_base_ptr};
  return &atomic_base_ptr[index];
}

}  // namespace rocshmem
