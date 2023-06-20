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

#ifndef LIBRARY_SRC_GPU_IB_CONTEXT_IB_DEVICE_HPP_
#define LIBRARY_SRC_GPU_IB_CONTEXT_IB_DEVICE_HPP_

#include "src/context.hpp"
#include "src/gpu_ib/memory_builder_policy.hpp"
#include "src/gpu_ib/network_policy.hpp"

namespace rocshmem {

class QueuePair;

class GPUIBContext : public Context {
 public:
  __host__ GPUIBContext(Backend *b, bool option, int idx);

  __device__ __host__ QueuePair *getQueuePair(int pe);

  __device__ __host__ int getNumQueuePairs();

  __device__ __host__ int getNumDest();

  __device__ __attribute__((noinline)) void threadfence_system();

  __device__ void ctx_create();

  __device__ void ctx_destroy();

  __device__ void putmem(void *dest, const void *source, size_t nelems, int pe);

  __device__ void getmem(void *dest, const void *source, size_t nelems, int pe);

  __device__ void putmem_nbi(void *dest, const void *source, size_t nelems,
                             int pe);

  __device__ void getmem_nbi(void *dest, const void *source, size_t size,
                             int pe);

  __device__ void fence();

  __device__ void fence(int pe);

  __device__ void quiet();

  __device__ void *shmem_ptr(const void *dest, int pe);

  __device__ void barrier_all();

  __device__ void sync_all();

  __device__ void sync(roc_shmem_team_t team);

  template <typename T>
  __device__ void amo_add(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_set(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_swap(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_and(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_and(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_or(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_or(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_xor(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_xor(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ T amo_fetch_add(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ void p(T *dest, T value, int pe);

  template <typename T>
  __device__ T g(const T *source, int pe);

  template <typename T, ROC_SHMEM_OP Op>
  __device__ void to_all(T *dest, const T *source, int nreduce, int PE_start,
                         int logPE_stride, int PE_size, T *pWrk,
                         long *pSync);  // NOLINT(runtime/int)

  template <typename T, ROC_SHMEM_OP Op>
  __device__ void to_all(roc_shmem_team_t team, T *dest, const T *source,
                         int nreduce);

  template <typename T>
  __device__ void put(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void broadcast(roc_shmem_team_t team, T *dest, const T *source,
                            int nelems, int pe_root);

  template <typename T>
  __device__ void broadcast(T *dest, const T *source, int nelems, int pe_root,
                            int pe_start, int log_pe_stride, int pe_size,
                            long *p_sync);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void alltoall(roc_shmem_team_t team, T *dest, const T *source,
                           int nelems);

  template <typename T>
  __device__ void alltoall_broadcast(roc_shmem_team_t team, T *dest,
                                     const T *source, int nelems);

  template <typename T>
  __device__ void alltoall_brucks(roc_shmem_team_t team, T *dest,
                                  const T *source, int nelems);

  template <typename T>
  __device__ void alltoall_gcen(roc_shmem_team_t team, T *dest, const T *source,
                                int nelems);

  template <typename T>
  __device__ void alltoall_gcen2(roc_shmem_team_t team, T *dest,
                                 const T *source, int nelems);

  template <typename T>
  __device__ void fcollect(roc_shmem_team_t team, T *dest, const T *source,
                           int nelems);

  template <typename T>
  __device__ void fcollect_broadcast(roc_shmem_team_t team, T *dest,
                                     const T *source, int nelems);

  template <typename T>
  __device__ void fcollect_brucks(roc_shmem_team_t team, T *dest,
                                  const T *source, int nelems);

  template <typename T>
  __device__ void fcollect_gcen(roc_shmem_team_t team, T *dest, const T *source,
                                int nelems);

  template <typename T>
  __device__ void fcollect_gcen2(roc_shmem_team_t team, T *dest,
                                 const T *source, int nelems);

  __device__ void putmem_wg(void *dest, const void *source, size_t nelems,
                            int pe);

  __device__ void getmem_wg(void *dest, const void *source, size_t nelems,
                            int pe);

  __device__ void putmem_nbi_wg(void *dest, const void *source, size_t nelems,
                                int pe);

  __device__ void getmem_nbi_wg(void *dest, const void *source, size_t size,
                                int pe);

  __device__ void putmem_wave(void *dest, const void *source, size_t nelems,
                              int pe);

  __device__ void getmem_wave(void *dest, const void *source, size_t nelems,
                              int pe);

  __device__ void putmem_nbi_wave(void *dest, const void *source, size_t nelems,
                                  int pe);

  __device__ void getmem_nbi_wave(void *dest, const void *source, size_t size,
                                  int pe);

  template <typename T>
  __device__ void put_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

 private:
  template <typename T, ROC_SHMEM_OP Op>
  __device__ void internal_direct_allreduce(
      T *dst, const T *src, int nelems, int PE_start, int logPE_stride,
      int PE_size, T *pWrk,
      long *pSync);  // NOLINT(runtime/int)

  template <typename T, ROC_SHMEM_OP Op>
  __device__ void internal_ring_allreduce(T *dst, const T *src, int nelems,
                                          int PE_start, int logPE_stride,
                                          int PE_size, T *pWrk,
                                          long *pSync,  // NOLINT(runtime/int)
                                          int n_seg, int seg_size,
                                          int chunk_size);

  template <typename T>
  __device__ void internal_put_broadcast(T *dst, const T *src, int nelems,
                                         int pe_root, int PE_start,
                                         int logPE_stride, int PE_size,
                                         long *pSync);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void internal_get_broadcast(T *dst, const T *src, int nelems,
                                         int pe_root,
                                         long *pSync);  // NOLINT(runtime/int)

  __device__ void internal_direct_barrier(int pe, int PE_start, int stride,
                                          int n_pes, int64_t *pSync);

  __device__ void internal_atomic_barrier(int pe, int PE_start, int stride,
                                          int n_pes, int64_t *pSync);

  __device__ void internal_sync(int pe, int PE_start, int stride, int PE_size,
                                int64_t *pSync);

  __device__ void quiet_single(int cq_num);

 public:
  /*
   * Collection of queue pairs that are currently checked out by this
   * context from GPUIBBackend.
   */
  // FIXME: keep it private and destroy in destructor for better
  // encapsulation.
  QueuePair *device_qp_proxy{nullptr};

  /*
   * Array of char * pointers corresponding to the heap base pointers VA for
   * each PE that we can communicate with.
   */
  char *const *base_heap{nullptr};

  /*
   * Buffer used to store the results of a *_g operation. These ops do not
   * provide a destination buffer, so the runtime must manage one.
   */
  char *g_ret{nullptr};

  NetworkImpl networkImpl{};

  /*
   * Temporary scratchpad memory used by internal barrier algorithms.
   */
  int64_t *barrier_sync{nullptr};

  int ctx_idx;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GPU_IB_CONTEXT_IB_DEVICE_HPP_
