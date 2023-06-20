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

/**
 * @file roc_shmem.cpp
 * @brief Public header for ROC_SHMEM device and host libraries.
 *
 * This is the implementation for the public roc_shmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 *
 * The device-side delegation is nasty because we can't use polymorphism with
 * our current shader compiler stack.  Maybe one day.....
 *
 * TODO: Could probably autogenerate many of these functions from macros.
 *
 * TODO: Support runtime backend detection.
 *
 */

#include <hip/hip_runtime.h>

#include <cstdlib>

#include "config.h"  // NOLINT(build/include_subdir)
#include "include/roc_shmem.hpp"
#include "src/backend_bc.hpp"
#include "src/context_incl.hpp"
#include "src/team.hpp"
#include "src/templates.hpp"
#include "src/util.hpp"

#ifdef USE_GPU_IB
#include "src/gpu_ib/context_ib_tmpl_device.hpp"
#else
#include "src/reverse_offload/context_ro_tmpl_device.hpp"
#endif

/******************************************************************************
 **************************** Device Vars And Init ****************************
 *****************************************************************************/

namespace rocshmem {

__device__ __constant__ roc_shmem_ctx_t ROC_SHMEM_CTX_DEFAULT {};

__constant__ Backend *device_backend_proxy;

__device__ void roc_shmem_wg_init() {
  int provided;

  /*
   * Non-threaded init is allowed to select any thread mode, so don't worry
   * if provided is different.
   */
  roc_shmem_wg_init_thread(ROC_SHMEM_THREAD_WG_FUNNELED, &provided);
}

__device__ void roc_shmem_wg_init_thread([[maybe_unused]] int requested,
                                         int *provided) {
  roc_shmem_query_thread(provided);
}

__device__ void roc_shmem_query_thread(int *provided) {
#ifdef USE_THREADS
  *provided = ROC_SHMEM_THREAD_MULTIPLE;
#else
  *provided = ROC_SHMEM_THREAD_WG_FUNNELED;
#endif
}

__device__ void roc_shmem_wg_finalize() {}

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

__device__ void roc_shmem_putmem(void *dest, const void *source, size_t nelems,
                                 int pe) {
  roc_shmem_ctx_putmem(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmem_put(T *dest, const T *source, size_t nelems, int pe) {
  roc_shmem_put(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmem_p(T *dest, T value, int pe) {
  roc_shmem_p(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T roc_shmem_g(const T *source, int pe) {
  return roc_shmem_g(ROC_SHMEM_CTX_DEFAULT, source, pe);
}

__device__ void roc_shmem_getmem(void *dest, const void *source, size_t nelems,
                                 int pe) {
  roc_shmem_ctx_getmem(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmem_get(T *dest, const T *source, size_t nelems, int pe) {
  roc_shmem_get(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void roc_shmem_putmem_nbi(void *dest, const void *source,
                                     size_t nelems, int pe) {
  roc_shmem_ctx_putmem_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmem_put_nbi(T *dest, const T *source, size_t nelems,
                                  int pe) {
  roc_shmem_put_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void roc_shmem_getmem_nbi(void *dest, const void *source,
                                     size_t nelems, int pe) {
  roc_shmem_ctx_getmem_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmem_get_nbi(T *dest, const T *source, size_t nelems,
                                  int pe) {
  roc_shmem_get_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void roc_shmem_fence() {
  roc_shmem_ctx_fence(ROC_SHMEM_CTX_DEFAULT);
}

__device__ void roc_shmem_fence(int pe) {
  roc_shmem_ctx_fence(ROC_SHMEM_CTX_DEFAULT, pe);
}

__device__ void roc_shmem_quiet() {
  roc_shmem_ctx_quiet(ROC_SHMEM_CTX_DEFAULT);
}

__device__ void roc_shmem_threadfence_system() {
  roc_shmem_ctx_threadfence_system(ROC_SHMEM_CTX_DEFAULT);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_add(T *dest, T val, int pe) {
  return roc_shmem_atomic_fetch_add(ROC_SHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_compare_swap(T *dest, T cond, T val, int pe) {
  return roc_shmem_atomic_compare_swap(ROC_SHMEM_CTX_DEFAULT, dest, cond, val,
                                       pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_inc(T *dest, int pe) {
  return roc_shmem_atomic_fetch_inc(ROC_SHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch(T *dest, int pe) {
  return roc_shmem_atomic_fetch(ROC_SHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_add(T *dest, T val, int pe) {
  roc_shmem_atomic_add(ROC_SHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_inc(T *dest, int pe) {
  roc_shmem_atomic_inc(ROC_SHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_set(T *dest, T value, int pe) {
  roc_shmem_atomic_set(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_swap(T *dest, T value, int pe) {
  return roc_shmem_atomic_swap(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_and(T *dest, T value, int pe) {
  return roc_shmem_atomic_fetch_and(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_and(T *dest, T value, int pe) {
  roc_shmem_atomic_and(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_or(T *dest, T value, int pe) {
  return roc_shmem_atomic_fetch_or(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_or(T *dest, T value, int pe) {
  roc_shmem_atomic_or(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_xor(T *dest, T value, int pe) {
  return roc_shmem_atomic_fetch_xor(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_xor(T *dest, T value, int pe) {
  roc_shmem_atomic_xor(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

__device__ int translate_pe(roc_shmem_ctx_t ctx, int pe) {
  if (ctx.team_opaque) {
    TeamInfo *tinfo = reinterpret_cast<TeamInfo *>(ctx.team_opaque);
    return (tinfo->pe_start + tinfo->stride * pe);
  } else {
    return pe;
  }
}

__host__ void set_internal_ctx(roc_shmem_ctx_t *ctx) {
    CHECK_HIP(hipMemcpyToSymbol(HIP_SYMBOL(ROC_SHMEM_CTX_DEFAULT), ctx,
                                sizeof(roc_shmem_ctx_t), 0,
                                hipMemcpyHostToDevice));
}

__device__ Context *get_internal_ctx(roc_shmem_ctx_t ctx) {
  return reinterpret_cast<Context *>(ctx.ctx_opaque);
}

__device__ void roc_shmem_wg_ctx_create(long option, roc_shmem_ctx_t *ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_create\n");
  if (get_flat_block_id() == 0) {
    device_backend_proxy->create_ctx(option, ctx);
    reinterpret_cast<Context *>(ctx->ctx_opaque)->setFence(option);
    ctx->team_opaque = nullptr;
  }
  __syncthreads();
}

__device__ int roc_shmem_wg_team_create_ctx(roc_shmem_team_t team, long options,
                                            roc_shmem_ctx_t *ctx) {
  GPU_DPRINTF("Function: roc_shmem_team_create_ctx\n");
  if (team == ROC_SHMEM_TEAM_INVALID) {
    return -1;
  }

  if (get_flat_block_id() == 0) {
    device_backend_proxy->create_ctx(options, ctx);
    reinterpret_cast<Context *>(ctx->ctx_opaque)->setFence(options);
    Team *team_obj{get_internal_team(team)};
    TeamInfo *info_wrt_world = team_obj->tinfo_wrt_world;
    ctx->team_opaque = info_wrt_world;
  }
  __syncthreads();

  return 0;
}

__device__ void roc_shmem_wg_ctx_destroy([[maybe_unused]] roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_destroy\n");
}

__device__ void roc_shmem_ctx_threadfence_system(roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_threadfence_system\n");

  get_internal_ctx(ctx)->threadfence_system();
}

__device__ void roc_shmem_ctx_putmem(roc_shmem_ctx_t ctx, void *dest,
                                     const void *source, size_t nelems,
                                     int pe) {
  GPU_DPRINTF("Function: roc_shmem_ctx_putmem\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->putmem(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmem_put\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->put(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe) {
  GPU_DPRINTF("Function: roc_shmem_p\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->p(dest, value, pe_in_world);
}

template <typename T>
__device__ T roc_shmem_g(roc_shmem_ctx_t ctx, const T *source, int pe) {
  GPU_DPRINTF("Function: roc_shmem_g\n");

  int pe_in_world = translate_pe(ctx, pe);

  return get_internal_ctx(ctx)->g(source, pe_in_world);
}

__device__ void roc_shmem_ctx_getmem(roc_shmem_ctx_t ctx, void *dest,
                                     const void *source, size_t nelems,
                                     int pe) {
  GPU_DPRINTF("Function: roc_shmem_ctx_getmem\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->getmem(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source,
                              size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmem_get\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->get(dest, source, nelems, pe_in_world);
}

__device__ void roc_shmem_ctx_putmem_nbi(roc_shmem_ctx_t ctx, void *dest,
                                         const void *source, size_t nelems,
                                         int pe) {
  GPU_DPRINTF("Function: roc_shmem_ctx_putmem_nbi\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmem_put_nbi\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe_in_world);
}

__device__ void roc_shmem_ctx_getmem_nbi(roc_shmem_ctx_t ctx, void *dest,
                                         const void *source, size_t nelems,
                                         int pe) {
  GPU_DPRINTF("Function: roc_shmem_ctx_getmem_nbi\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->getmem_nbi(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmem_get_nbi\n");

  int pe_in_world = translate_pe(ctx, pe);

  get_internal_ctx(ctx)->get_nbi(dest, source, nelems, pe_in_world);
}

__device__ void roc_shmem_ctx_fence(roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_fence\n");

  get_internal_ctx(ctx)->fence();
}

__device__ void roc_shmem_ctx_fence(roc_shmem_ctx_t ctx, int pe) {
  GPU_DPRINTF("Function: roc_shmem_ctx_fence\n");

  get_internal_ctx(ctx)->fence(pe);
}

__device__ void roc_shmem_ctx_quiet(roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_quiet\n");

  get_internal_ctx(ctx)->quiet();
}

__device__ void *roc_shmem_ptr(const void *dest, int pe) {
  GPU_DPRINTF("Function: roc_shmem_ptr\n");

  return get_internal_ctx(ROC_SHMEM_CTX_DEFAULT)->shmem_ptr(dest, pe);
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void roc_shmem_wg_to_all(roc_shmem_ctx_t ctx, T *dest,
                                    const T *source, int nreduce, int PE_start,
                                    int logPE_stride, int PE_size, T *pWrk,
                                    long *pSync) {
  GPU_DPRINTF("Function: roc_shmem_to_all\n");

  get_internal_ctx(ctx)->to_all<T, Op>(dest, source, nreduce, PE_start,
                                       logPE_stride, PE_size, pWrk, pSync);
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void roc_shmem_wg_to_all(roc_shmem_ctx_t ctx, roc_shmem_team_t team,
                                    T *dest, const T *source, int nreduce) {
  GPU_DPRINTF("Function: roc_shmem_to_all\n");

  get_internal_ctx(ctx)->to_all<T, Op>(team, dest, source, nreduce);
}

template <typename T>
__device__ void roc_shmem_wg_broadcast(roc_shmem_ctx_t ctx, T *dest,
                                       const T *source, int nelem, int pe_root,
                                       int pe_start, int log_pe_stride,
                                       int pe_size, long *p_sync) {
  GPU_DPRINTF("Function: roc_shmem_broadcast\n");

  get_internal_ctx(ctx)->broadcast<T>(dest, source, nelem, pe_root, pe_start,
                                      log_pe_stride, pe_size, p_sync);
}

template <typename T>
__device__ void roc_shmem_wg_broadcast(roc_shmem_ctx_t ctx,
                                       roc_shmem_team_t team, T *dest,
                                       const T *source, int nelem,
                                       int pe_root) {
  GPU_DPRINTF("Function: Team-based roc_shmem_broadcast\n");

  get_internal_ctx(ctx)->broadcast<T>(team, dest, source, nelem, pe_root);
}

template <typename T>
__device__ void roc_shmem_wg_alltoall(roc_shmem_ctx_t ctx,
                                      roc_shmem_team_t team, T *dest,
                                      const T *source, int nelem) {
  GPU_DPRINTF("Function: roc_shmem_alltoall\n");

  get_internal_ctx(ctx)->alltoall<T>(team, dest, source, nelem);
}

template <typename T>
__device__ void roc_shmem_wg_fcollect(roc_shmem_ctx_t ctx,
                                      roc_shmem_team_t team, T *dest,
                                      const T *source, int nelem) {
  GPU_DPRINTF("Function: roc_shmem_fcollect\n");

  get_internal_ctx(ctx)->fcollect<T>(team, dest, source, nelem);
}

template <typename T>
__device__ void roc_shmem_wait_until(T *ptr, roc_shmem_cmps cmp, T val) {
  GPU_DPRINTF("Function: roc_shmem_wait_until\n");

  Context *ctx_internal = get_internal_ctx(ROC_SHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL);

  ctx_internal->wait_until(ptr, cmp, val);
}

template <typename T>
__device__ int roc_shmem_test(T *ptr, roc_shmem_cmps cmp, T val) {
  GPU_DPRINTF("Function: roc_shmem_testl\n");

  Context *ctx_internal = get_internal_ctx(ROC_SHMEM_CTX_DEFAULT);
  ctx_internal->ctxStats.incStat(NUM_TEST);

  return ctx_internal->test(ptr, cmp, val);
}

__device__ void roc_shmem_ctx_wg_barrier_all(roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_barrier_all\n");

  get_internal_ctx(ctx)->barrier_all();
}

__device__ void roc_shmem_wg_barrier_all() {
  roc_shmem_ctx_wg_barrier_all(ROC_SHMEM_CTX_DEFAULT);
}

__device__ void roc_shmem_ctx_wg_sync_all(roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_sync_all\n");

  get_internal_ctx(ctx)->sync_all();
}

__device__ void roc_shmem_wg_sync_all() {
  roc_shmem_ctx_wg_sync_all(ROC_SHMEM_CTX_DEFAULT);
}

__device__ void roc_shmem_ctx_wg_team_sync(roc_shmem_ctx_t ctx,
                                           roc_shmem_team_t team) {
  GPU_DPRINTF("Function: roc_shmem_ctx_sync_all\n");

  get_internal_ctx(ctx)->sync(team);
}

__device__ void roc_shmem_wg_team_sync(roc_shmem_team_t team) {
  roc_shmem_ctx_wg_team_sync(ROC_SHMEM_CTX_DEFAULT, team);
}

__device__ int roc_shmem_ctx_n_pes(roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_n_pes\n");

  return get_internal_ctx(ctx)->num_pes;
}

__device__ int roc_shmem_n_pes() {
  return get_internal_ctx(ROC_SHMEM_CTX_DEFAULT)->num_pes;
}

__device__ int roc_shmem_ctx_my_pe(roc_shmem_ctx_t ctx) {
  GPU_DPRINTF("Function: roc_shmem_ctx_my_pe\n");

  return get_internal_ctx(ctx)->my_pe;
}

__device__ int roc_shmem_my_pe() {
  return get_internal_ctx(ROC_SHMEM_CTX_DEFAULT)->my_pe;
}

__device__ uint64_t roc_shmem_timer() {
  GPU_DPRINTF("Function: roc_shmem_timer\n");

  return __read_clock();
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_fetch_add\n");

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, val, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest, T cond,
                                           T val, int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_compare_swap\n");

  return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_fetch_inc\n");

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, 1, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_fetch\n");

  return get_internal_ctx(ctx)->amo_fetch_add<T>(dest, 0, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_add\n");

  get_internal_ctx(ctx)->amo_add<T>(dest, val, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_inc\n");

  get_internal_ctx(ctx)->amo_add<T>(dest, 1, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_set(roc_shmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_set\n");

  get_internal_ctx(ctx)->amo_set(dest, val, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_swap(roc_shmem_ctx_t ctx, T *dest, T val,
                                   int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_swap\n");

  return get_internal_ctx(ctx)->amo_swap(dest, val, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_and(roc_shmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_fetch_and\n");

  return get_internal_ctx(ctx)->amo_fetch_and(dest, val, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_and(roc_shmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_and\n");

  get_internal_ctx(ctx)->amo_and(dest, val, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_or(roc_shmem_ctx_t ctx, T *dest, T val,
                                       int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_fetch_or\n");

  return get_internal_ctx(ctx)->amo_fetch_or(dest, val, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_or(roc_shmem_ctx_t ctx, T *dest, T val,
                                    int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_or\n");

  get_internal_ctx(ctx)->amo_or(dest, val, pe);
}

template <typename T>
__device__ T roc_shmem_atomic_fetch_xor(roc_shmem_ctx_t ctx, T *dest, T val,
                                        int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_fetch_xor\n");

  return get_internal_ctx(ctx)->amo_fetch_xor(dest, val, pe);
}

template <typename T>
__device__ void roc_shmem_atomic_xor(roc_shmem_ctx_t ctx, T *dest, T val,
                                     int pe) {
  GPU_DPRINTF("Function: roc_shmem_atomic_xor\n");

  get_internal_ctx(ctx)->amo_xor(dest, val, pe);
}

/**
 *      SHMEM X RMA API for WG and Wave level
 */
__device__ void roc_shmemx_ctx_putmem_wave(roc_shmem_ctx_t ctx, void *dest,
                                           const void *source, size_t nelems,
                                           int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_wave\n");

  get_internal_ctx(ctx)->putmem_wave(dest, source, nelems, pe);
}

__device__ void roc_shmemx_ctx_putmem_wg(roc_shmem_ctx_t ctx, void *dest,
                                         const void *source, size_t nelems,
                                         int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_wg\n");

  get_internal_ctx(ctx)->putmem_wg(dest, source, nelems, pe);
}

__device__ void roc_shmemx_ctx_putmem_nbi_wave(roc_shmem_ctx_t ctx, void *dest,
                                               const void *source,
                                               size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_nbi_wave\n");

  get_internal_ctx(ctx)->putmem_nbi_wave(dest, source, nelems, pe);
}

__device__ void roc_shmemx_ctx_putmem_nbi_wg(roc_shmem_ctx_t ctx, void *dest,
                                             const void *source, size_t nelems,
                                             int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_nbi_wg\n");

  get_internal_ctx(ctx)->putmem_nbi_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_put_wave(roc_shmem_ctx_t ctx, T *dest,
                                    const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_put_wave\n");

  get_internal_ctx(ctx)->put_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_put_wg(roc_shmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_put_wg\n");

  get_internal_ctx(ctx)->put_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_put_nbi_wave(roc_shmem_ctx_t ctx, T *dest,
                                        const T *source, size_t nelems,
                                        int pe) {
  GPU_DPRINTF("Function: roc_shmemx_put_nbi_wave\n");

  get_internal_ctx(ctx)->put_nbi_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_put_nbi_wg(roc_shmem_ctx_t ctx, T *dest,
                                      const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_put_nbi_wg\n");

  get_internal_ctx(ctx)->put_nbi_wg(dest, source, nelems, pe);
}

__device__ void roc_shmemx_ctx_getmem_wg(roc_shmem_ctx_t ctx, void *dest,
                                         const void *source, size_t nelems,
                                         int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_wg\n");

  get_internal_ctx(ctx)->getmem_wg(dest, source, nelems, pe);
}

__device__ void roc_shmemx_ctx_getmem_wave(roc_shmem_ctx_t ctx, void *dest,
                                           const void *source, size_t nelems,
                                           int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_wave\n");

  get_internal_ctx(ctx)->getmem_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_get_wg(roc_shmem_ctx_t ctx, T *dest, const T *source,
                                  size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_get_wg\n");

  get_internal_ctx(ctx)->get_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_get_wave(roc_shmem_ctx_t ctx, T *dest,
                                    const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_get_wave\n");

  get_internal_ctx(ctx)->get_wave(dest, source, nelems, pe);
}

__device__ void roc_shmemx_ctx_getmem_nbi_wg(roc_shmem_ctx_t ctx, void *dest,
                                             const void *source, size_t nelems,
                                             int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_nbi_wg\n");

  get_internal_ctx(ctx)->getmem_nbi_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_get_nbi_wg(roc_shmem_ctx_t ctx, T *dest,
                                      const T *source, size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_get_nbi_wg\n");

  get_internal_ctx(ctx)->get_nbi_wg(dest, source, nelems, pe);
}

__device__ void roc_shmemx_ctx_getmem_nbi_wave(roc_shmem_ctx_t ctx, void *dest,
                                               const void *source,
                                               size_t nelems, int pe) {
  GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_nbi_wave\n");

  get_internal_ctx(ctx)->getmem_nbi_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void roc_shmemx_get_nbi_wave(roc_shmem_ctx_t ctx, T *dest,
                                        const T *source, size_t nelems,
                                        int pe) {
  GPU_DPRINTF("Function: roc_shmemx_get_nbi_wave\n");

  get_internal_ctx(ctx)->get_nbi_wave(dest, source, nelems, pe);
}

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__device__ int roc_shmem_team_translate_pe(roc_shmem_team_t src_team,
                                           int src_pe,
                                           roc_shmem_team_t dst_team) {
  return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************* Template Generation Macros *************************
 *****************************************************************************/

/**
 * Template generator for reductions
 */
#define REDUCTION_GEN(T, Op)                                                 \
  template __device__ void roc_shmem_wg_to_all<T, Op>(                       \
      roc_shmem_ctx_t ctx, T * dest, const T *source, int nreduce,           \
      int PE_start, int logPE_stride, int PE_size, T *pWrk, long *pSync);    \
  template __device__ void roc_shmem_wg_to_all<T, Op>(                       \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T * dest, const T *source, \
      int nreduce);

/**
 * Declare templates for the required datatypes (for the compiler)
 */
#define RMA_GEN(T)                                                             \
  template __device__ void roc_shmem_put<T>(                                   \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmem_put_nbi<T>(                               \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmem_p<T>(roc_shmem_ctx_t ctx, T * dest,       \
                                          T value, int pe);                    \
  template __device__ void roc_shmem_get<T>(                                   \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmem_get_nbi<T>(                               \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ T roc_shmem_g<T>(roc_shmem_ctx_t ctx, const T *source,   \
                                       int pe);                                \
  template __device__ void roc_shmem_put<T>(T * dest, const T *source,         \
                                            size_t nelems, int pe);            \
  template __device__ void roc_shmem_put_nbi<T>(T * dest, const T *source,     \
                                                size_t nelems, int pe);        \
  template __device__ void roc_shmem_p<T>(T * dest, T value, int pe);          \
  template __device__ void roc_shmem_get<T>(T * dest, const T *source,         \
                                            size_t nelems, int pe);            \
  template __device__ void roc_shmem_get_nbi<T>(T * dest, const T *source,     \
                                                size_t nelems, int pe);        \
  template __device__ T roc_shmem_g<T>(const T *source, int pe);               \
  template __device__ void roc_shmem_wg_broadcast<T>(                          \
      roc_shmem_ctx_t ctx, T * dest, const T *source, int nelem, int pe_root,  \
      int pe_start, int log_pe_stride, int pe_size, long *p_sync);             \
  template __device__ void roc_shmem_wg_broadcast<T>(                          \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T * dest, const T *source,   \
      int nelem, int pe_root);                                                 \
  template __device__ void roc_shmem_wg_alltoall<T>(                           \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T * dest, const T *source,   \
      int nelem);                                                              \
  template __device__ void roc_shmem_wg_fcollect<T>(                           \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T * dest, const T *source,   \
      int nelem);                                                              \
  template __device__ void roc_shmemx_put_wave<T>(                             \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_put_wg<T>(                               \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_put_wave<T>(T * dest, const T *source,   \
                                                  size_t nelems, int pe);      \
  template __device__ void roc_shmemx_put_wg<T>(T * dest, const T *source,     \
                                                size_t nelems, int pe);        \
  template __device__ void roc_shmemx_put_nbi_wave<T>(                         \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_put_nbi_wg<T>(                           \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_put_nbi_wave<T>(                         \
      T * dest, const T *source, size_t nelems, int pe);                       \
  template __device__ void roc_shmemx_put_nbi_wg<T>(T * dest, const T *source, \
                                                    size_t nelems, int pe);    \
  template __device__ void roc_shmemx_get_wave<T>(                             \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_get_wg<T>(                               \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_get_wave<T>(T * dest, const T *source,   \
                                                  size_t nelems, int pe);      \
  template __device__ void roc_shmemx_get_wg<T>(T * dest, const T *source,     \
                                                size_t nelems, int pe);        \
  template __device__ void roc_shmemx_get_nbi_wave<T>(                         \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_get_nbi_wg<T>(                           \
      roc_shmem_ctx_t ctx, T * dest, const T *source, size_t nelems, int pe);  \
  template __device__ void roc_shmemx_get_nbi_wave<T>(                         \
      T * dest, const T *source, size_t nelems, int pe);                       \
  template __device__ void roc_shmemx_get_nbi_wg<T>(T * dest, const T *source, \
                                                    size_t nelems, int pe);

/**
 * Declare templates for the standard amo types
 */
#define AMO_STANDARD_GEN(T)                                                    \
  template __device__ T roc_shmem_atomic_compare_swap<T>(                      \
      roc_shmem_ctx_t ctx, T * dest, T cond, T value, int pe);                 \
  template __device__ T roc_shmem_atomic_compare_swap<T>(T * dest, T cond,     \
                                                         T value, int pe);     \
  template __device__ T roc_shmem_atomic_fetch_inc<T>(roc_shmem_ctx_t ctx,     \
                                                      T * dest, int pe);       \
  template __device__ T roc_shmem_atomic_fetch_inc<T>(T * dest, int pe);       \
  template __device__ void roc_shmem_atomic_inc<T>(roc_shmem_ctx_t ctx,        \
                                                   T * dest, int pe);          \
  template __device__ void roc_shmem_atomic_inc<T>(T * dest, int pe);          \
  template __device__ T roc_shmem_atomic_fetch_add<T>(                         \
      roc_shmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __device__ T roc_shmem_atomic_fetch_add<T>(T * dest, T value,       \
                                                      int pe);                 \
  template __device__ void roc_shmem_atomic_add<T>(roc_shmem_ctx_t ctx,        \
                                                   T * dest, T value, int pe); \
  template __device__ void roc_shmem_atomic_add<T>(T * dest, T value, int pe);

/**
 * Declare templates for the extended amo types
 */
#define AMO_EXTENDED_GEN(T)                                                    \
  template __device__ T roc_shmem_atomic_fetch<T>(roc_shmem_ctx_t ctx,         \
                                                  T * dest, int pe);           \
  template __device__ T roc_shmem_atomic_fetch<T>(T * dest, int pe);           \
  template __device__ void roc_shmem_atomic_set<T>(roc_shmem_ctx_t ctx,        \
                                                   T * dest, T value, int pe); \
  template __device__ void roc_shmem_atomic_set<T>(T * dest, T value, int pe); \
  template __device__ T roc_shmem_atomic_swap<T>(roc_shmem_ctx_t ctx,          \
                                                 T * dest, T value, int pe);   \
  template __device__ T roc_shmem_atomic_swap<T>(T * dest, T value, int pe);

/**
 * Declare templates for the bitwise amo types
 */
#define AMO_BITWISE_GEN(T)                                                     \
  template __device__ T roc_shmem_atomic_fetch_and<T>(                         \
      roc_shmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __device__ T roc_shmem_atomic_fetch_and<T>(T * dest, T value,       \
                                                      int pe);                 \
  template __device__ void roc_shmem_atomic_and<T>(roc_shmem_ctx_t ctx,        \
                                                   T * dest, T value, int pe); \
  template __device__ void roc_shmem_atomic_and<T>(T * dest, T value, int pe); \
  template __device__ T roc_shmem_atomic_fetch_or<T>(                          \
      roc_shmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __device__ T roc_shmem_atomic_fetch_or<T>(T * dest, T value,        \
                                                     int pe);                  \
  template __device__ void roc_shmem_atomic_or<T>(roc_shmem_ctx_t ctx,         \
                                                  T * dest, T value, int pe);  \
  template __device__ void roc_shmem_atomic_or<T>(T * dest, T value, int pe);  \
  template __device__ T roc_shmem_atomic_fetch_xor<T>(                         \
      roc_shmem_ctx_t ctx, T * dest, T value, int pe);                         \
  template __device__ T roc_shmem_atomic_fetch_xor<T>(T * dest, T value,       \
                                                      int pe);                 \
  template __device__ void roc_shmem_atomic_xor<T>(roc_shmem_ctx_t ctx,        \
                                                   T * dest, T value, int pe); \
  template __device__ void roc_shmem_atomic_xor<T>(T * dest, T value, int pe);

/**
 * Declare templates for the wait types
 */
#define WAIT_GEN(T)                                                            \
  template __device__ void roc_shmem_wait_until<T>(T * ptr,                    \
                                                   roc_shmem_cmps cmp, T val); \
  template __device__ int roc_shmem_test<T>(T * ptr, roc_shmem_cmps cmp,       \
                                            T val);                            \
  template __device__ void Context::wait_until<T>(T * ptr, roc_shmem_cmps cmp, \
                                                  T val);                      \
  template __device__ int Context::test<T>(T * ptr, roc_shmem_cmps cmp, T val);

#define ARITH_REDUCTION_GEN(T)    \
  REDUCTION_GEN(T, ROC_SHMEM_SUM) \
  REDUCTION_GEN(T, ROC_SHMEM_MIN) \
  REDUCTION_GEN(T, ROC_SHMEM_MAX) \
  REDUCTION_GEN(T, ROC_SHMEM_PROD)

#define BITWISE_REDUCTION_GEN(T)  \
  REDUCTION_GEN(T, ROC_SHMEM_OR)  \
  REDUCTION_GEN(T, ROC_SHMEM_AND) \
  REDUCTION_GEN(T, ROC_SHMEM_XOR)

#define INT_REDUCTION_GEN(T) \
  ARITH_REDUCTION_GEN(T)     \
  BITWISE_REDUCTION_GEN(T)

#define FLOAT_REDUCTION_GEN(T) ARITH_REDUCTION_GEN(T)

/**
 * Define APIs to call the template functions
 **/

#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op)                             \
  __device__ void roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(             \
      roc_shmem_ctx_t ctx, T *dest, const T *source, int nreduce,           \
      int PE_start, int logPE_stride, int PE_size, T *pWrk, long *pSync) {  \
    roc_shmem_wg_to_all<T, Op>(ctx, dest, source, nreduce, PE_start,        \
                               logPE_stride, PE_size, pWrk, pSync);         \
  }                                                                         \
  __device__ void roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(             \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T *dest, const T *source, \
      int nreduce) {                                                        \
    roc_shmem_wg_to_all<T, Op>(ctx, team, dest, source, nreduce);           \
  }

#define ARITH_REDUCTION_DEF_GEN(T, TNAME)         \
  REDUCTION_DEF_GEN(T, TNAME, sum, ROC_SHMEM_SUM) \
  REDUCTION_DEF_GEN(T, TNAME, min, ROC_SHMEM_MIN) \
  REDUCTION_DEF_GEN(T, TNAME, max, ROC_SHMEM_MAX) \
  REDUCTION_DEF_GEN(T, TNAME, prod, ROC_SHMEM_PROD)

#define BITWISE_REDUCTION_DEF_GEN(T, TNAME)       \
  REDUCTION_DEF_GEN(T, TNAME, or, ROC_SHMEM_OR)   \
  REDUCTION_DEF_GEN(T, TNAME, and, ROC_SHMEM_AND) \
  REDUCTION_DEF_GEN(T, TNAME, xor, ROC_SHMEM_XOR)

#define INT_REDUCTION_DEF_GEN(T, TNAME) \
  ARITH_REDUCTION_DEF_GEN(T, TNAME)     \
  BITWISE_REDUCTION_DEF_GEN(T, TNAME)

#define FLOAT_REDUCTION_DEF_GEN(T, TNAME) ARITH_REDUCTION_DEF_GEN(T, TNAME)

#define RMA_DEF_GEN(T, TNAME)                                                  \
  __device__ void roc_shmem_ctx_##TNAME##_put(                                 \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmem_put<T>(ctx, dest, source, nelems, pe);                           \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_put_nbi(                             \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmem_put_nbi<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_p(roc_shmem_ctx_t ctx, T *dest,      \
                                            T value, int pe) {                 \
    roc_shmem_p<T>(ctx, dest, value, pe);                                      \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_get(                                 \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmem_get<T>(ctx, dest, source, nelems, pe);                           \
  }                                                                            \
  __device__ T roc_shmem_ctx_##TNAME##_g(roc_shmem_ctx_t ctx, const T *source, \
                                         int pe) {                             \
    return roc_shmem_g<T>(ctx, source, pe);                                    \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_get_nbi(                             \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmem_get_nbi<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_put(T *dest, const T *source,            \
                                          size_t nelems, int pe) {             \
    roc_shmem_put<T>(dest, source, nelems, pe);                                \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_put_nbi(T *dest, const T *source,        \
                                              size_t nelems, int pe) {         \
    roc_shmem_put_nbi<T>(dest, source, nelems, pe);                            \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_p(T *dest, T value, int pe) {            \
    roc_shmem_p<T>(dest, value, pe);                                           \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_get(T *dest, const T *source,            \
                                          size_t nelems, int pe) {             \
    roc_shmem_get<T>(dest, source, nelems, pe);                                \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_get_nbi(T *dest, const T *source,        \
                                              size_t nelems, int pe) {         \
    roc_shmem_get_nbi<T>(dest, source, nelems, pe);                            \
  }                                                                            \
  __device__ T roc_shmem_##TNAME##_g(const T *source, int pe) {                \
    return roc_shmem_g<T>(source, pe);                                         \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_put_wave(                           \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_put_wave<T>(ctx, dest, source, nelems, pe);                     \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_put_wg(                             \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_put_wg<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_put_wave(T *dest, const T *source,      \
                                                size_t nelems, int pe) {       \
    roc_shmemx_put_wave<T>(dest, source, nelems, pe);                          \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_put_wg(T *dest, const T *source,        \
                                              size_t nelems, int pe) {         \
    roc_shmemx_put_wg<T>(dest, source, nelems, pe);                            \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_put_nbi_wave(                       \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_put_nbi_wave<T>(ctx, dest, source, nelems, pe);                 \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_put_nbi_wg(                         \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_put_nbi_wg<T>(ctx, dest, source, nelems, pe);                   \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_put_nbi_wave(T *dest, const T *source,  \
                                                    size_t nelems, int pe) {   \
    roc_shmemx_put_nbi_wave<T>(dest, source, nelems, pe);                      \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_put_nbi_wg(T *dest, const T *source,    \
                                                  size_t nelems, int pe) {     \
    roc_shmemx_put_nbi_wg<T>(dest, source, nelems, pe);                        \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_get_wave(                           \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_get_wave<T>(ctx, dest, source, nelems, pe);                     \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_get_wg(                             \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_get_wg<T>(ctx, dest, source, nelems, pe);                       \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_get_wave(T *dest, const T *source,      \
                                                size_t nelems, int pe) {       \
    roc_shmemx_get_wave<T>(dest, source, nelems, pe);                          \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_get_wg(T *dest, const T *source,        \
                                              size_t nelems, int pe) {         \
    roc_shmemx_get_wg<T>(dest, source, nelems, pe);                            \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_get_nbi_wave(                       \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_get_nbi_wave<T>(ctx, dest, source, nelems, pe);                 \
  }                                                                            \
  __device__ void roc_shmemx_ctx_##TNAME##_get_nbi_wg(                         \
      roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems, int pe) {  \
    roc_shmemx_get_nbi_wg<T>(ctx, dest, source, nelems, pe);                   \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_get_nbi_wave(T *dest, const T *source,  \
                                                    size_t nelems, int pe) {   \
    roc_shmemx_get_nbi_wave<T>(dest, source, nelems, pe);                      \
  }                                                                            \
  __device__ void roc_shmemx_##TNAME##_get_nbi_wg(T *dest, const T *source,    \
                                                  size_t nelems, int pe) {     \
    roc_shmemx_get_nbi_wg<T>(dest, source, nelems, pe);                        \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_wg_broadcast(                        \
      roc_shmem_ctx_t ctx, T *dest, const T *source, int nelem, int pe_root,   \
      int pe_start, int log_pe_stride, int pe_size, long *p_sync) {            \
    roc_shmem_wg_broadcast<T>(ctx, dest, source, nelem, pe_root, pe_start,     \
                              log_pe_stride, pe_size, p_sync);                 \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_wg_broadcast(                        \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T *dest, const T *source,    \
      int nelem, int pe_root) {                                                \
    roc_shmem_wg_broadcast<T>(ctx, team, dest, source, nelem, pe_root);        \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_wg_alltoall(                         \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T *dest, const T *source,    \
      int nelem) {                                                             \
    roc_shmem_wg_alltoall<T>(ctx, team, dest, source, nelem);                  \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_wg_fcollect(                         \
      roc_shmem_ctx_t ctx, roc_shmem_team_t team, T *dest, const T *source,    \
      int nelem) {                                                             \
    roc_shmem_wg_fcollect<T>(ctx, team, dest, source, nelem);                  \
  }

#define AMO_STANDARD_DEF_GEN(T, TNAME)                                       \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_compare_swap(                  \
      roc_shmem_ctx_t ctx, T *dest, T cond, T value, int pe) {               \
    return roc_shmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe);     \
  }                                                                          \
  __device__ T roc_shmem_##TNAME##_atomic_compare_swap(T *dest, T cond,      \
                                                       T value, int pe) {    \
    return roc_shmem_atomic_compare_swap<T>(dest, cond, value, pe);          \
  }                                                                          \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_fetch_inc(roc_shmem_ctx_t ctx, \
                                                        T *dest, int pe) {   \
    return roc_shmem_atomic_fetch_inc<T>(ctx, dest, pe);                     \
  }                                                                          \
  __device__ T roc_shmem_##TNAME##_atomic_fetch_inc(T *dest, int pe) {       \
    return roc_shmem_atomic_fetch_inc<T>(dest, pe);                          \
  }                                                                          \
  __device__ void roc_shmem_ctx_##TNAME##_atomic_inc(roc_shmem_ctx_t ctx,    \
                                                     T *dest, int pe) {      \
    roc_shmem_atomic_inc<T>(ctx, dest, pe);                                  \
  }                                                                          \
  __device__ void roc_shmem_##TNAME##_atomic_inc(T *dest, int pe) {          \
    roc_shmem_atomic_inc<T>(dest, pe);                                       \
  }                                                                          \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_fetch_add(                     \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                       \
    return roc_shmem_atomic_fetch_add<T>(ctx, dest, value, pe);              \
  }                                                                          \
  __device__ T roc_shmem_##TNAME##_atomic_fetch_add(T *dest, T value,        \
                                                    int pe) {                \
    return roc_shmem_atomic_fetch_add<T>(dest, value, pe);                   \
  }                                                                          \
  __device__ void roc_shmem_ctx_##TNAME##_atomic_add(                        \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                       \
    roc_shmem_atomic_add<T>(ctx, dest, value, pe);                           \
  }                                                                          \
  __device__ void roc_shmem_##TNAME##_atomic_add(T *dest, T value, int pe) { \
    roc_shmem_atomic_add<T>(dest, value, pe);                                \
  }

#define AMO_EXTENDED_DEF_GEN(T, TNAME)                                         \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_fetch(roc_shmem_ctx_t ctx,       \
                                                    T *dest, int pe) {         \
    return roc_shmem_atomic_fetch<T>(ctx, dest, pe);                           \
  }                                                                            \
  __device__ T roc_shmem_##TNAME##_atomic_fetch(T *dest, int pe) {             \
    return roc_shmem_atomic_fetch<T>(dest, pe);                                \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_atomic_set(                          \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    roc_shmem_atomic_set<T>(ctx, dest, value, pe);                             \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_atomic_set(T *dest, T value, int pe) {   \
    roc_shmem_atomic_set<T>(dest, value, pe);                                  \
  }                                                                            \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_swap(roc_shmem_ctx_t ctx,        \
                                                   T *dest, T value, int pe) { \
    return roc_shmem_atomic_swap<T>(ctx, dest, value, pe);                     \
  }                                                                            \
  __device__ T roc_shmem_##TNAME##_atomic_swap(T *dest, T value, int pe) {     \
    return roc_shmem_atomic_swap<T>(dest, value, pe);                          \
  }

#define AMO_BITWISE_DEF_GEN(T, TNAME)                                          \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_fetch_and(                       \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return roc_shmem_atomic_fetch_and<T>(ctx, dest, value, pe);                \
  }                                                                            \
  __device__ T roc_shmem_##TNAME##_atomic_fetch_and(T *dest, T value,          \
                                                    int pe) {                  \
    return roc_shmem_atomic_fetch_and<T>(dest, value, pe);                     \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_atomic_and(                          \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    roc_shmem_atomic_and<T>(ctx, dest, value, pe);                             \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_atomic_and(T *dest, T value, int pe) {   \
    roc_shmem_atomic_and<T>(dest, value, pe);                                  \
  }                                                                            \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_fetch_or(                        \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return roc_shmem_atomic_fetch_or<T>(ctx, dest, value, pe);                 \
  }                                                                            \
  __device__ T roc_shmem_##TNAME##_atomic_fetch_or(T *dest, T value, int pe) { \
    return roc_shmem_atomic_fetch_or<T>(dest, value, pe);                      \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_atomic_or(                           \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    roc_shmem_atomic_or<T>(ctx, dest, value, pe);                              \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_atomic_or(T *dest, T value, int pe) {    \
    roc_shmem_atomic_or<T>(dest, value, pe);                                   \
  }                                                                            \
  __device__ T roc_shmem_ctx_##TNAME##_atomic_fetch_xor(                       \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    return roc_shmem_atomic_fetch_xor<T>(ctx, dest, value, pe);                \
  }                                                                            \
  __device__ T roc_shmem_##TNAME##_atomic_fetch_xor(T *dest, T value,          \
                                                    int pe) {                  \
    return roc_shmem_atomic_fetch_xor<T>(dest, value, pe);                     \
  }                                                                            \
  __device__ void roc_shmem_ctx_##TNAME##_atomic_xor(                          \
      roc_shmem_ctx_t ctx, T *dest, T value, int pe) {                         \
    roc_shmem_atomic_xor<T>(ctx, dest, value, pe);                             \
  }                                                                            \
  __device__ void roc_shmem_##TNAME##_atomic_xor(T *dest, T value, int pe) {   \
    roc_shmem_atomic_xor<T>(dest, value, pe);                                  \
  }

#define WAIT_DEF_GEN(T, TNAME)                                                 \
  __device__ void roc_shmem_##TNAME##_wait_until(T *ptr, roc_shmem_cmps cmp,   \
                                                 T val) {                      \
    roc_shmem_wait_until<T>(ptr, cmp, val);                                    \
  }                                                                            \
  __device__ int roc_shmem_##TNAME##_test(T *ptr, roc_shmem_cmps cmp, T val) { \
    return roc_shmem_test<T>(ptr, cmp, val);                                   \
  }

/******************************************************************************
 ************************* Macro Invocation Per Type **************************
 *****************************************************************************/

// clang-format off
INT_REDUCTION_GEN(int)
INT_REDUCTION_GEN(short)
INT_REDUCTION_GEN(long)
INT_REDUCTION_GEN(long long)
FLOAT_REDUCTION_GEN(float)
FLOAT_REDUCTION_GEN(double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_GEN(long double)

RMA_GEN(float)
RMA_GEN(double)
// RMA_GEN(long double)
RMA_GEN(char)
RMA_GEN(signed char)
RMA_GEN(short)
RMA_GEN(int)
RMA_GEN(long)
RMA_GEN(long long)
RMA_GEN(unsigned char)
RMA_GEN(unsigned short)
RMA_GEN(unsigned int)
RMA_GEN(unsigned long)
RMA_GEN(unsigned long long)

AMO_STANDARD_GEN(int)
AMO_STANDARD_GEN(long)
AMO_STANDARD_GEN(long long)
AMO_STANDARD_GEN(unsigned int)
AMO_STANDARD_GEN(unsigned long)
AMO_STANDARD_GEN(unsigned long long)

AMO_EXTENDED_GEN(float)
AMO_EXTENDED_GEN(double)
AMO_EXTENDED_GEN(int)
AMO_EXTENDED_GEN(long)
AMO_EXTENDED_GEN(long long)
AMO_EXTENDED_GEN(unsigned int)
AMO_EXTENDED_GEN(unsigned long)
AMO_EXTENDED_GEN(unsigned long long)

AMO_BITWISE_GEN(unsigned int)
AMO_BITWISE_GEN(unsigned long)
AMO_BITWISE_GEN(unsigned long long)

/* Supported synchronization types */
WAIT_GEN(float)
WAIT_GEN(double)
// WAIT_GEN(long double)
WAIT_GEN(char)
WAIT_GEN(unsigned char)
WAIT_GEN(unsigned short)
WAIT_GEN(signed char)
WAIT_GEN(short)
WAIT_GEN(int)
WAIT_GEN(long)
WAIT_GEN(long long)
WAIT_GEN(unsigned int)
WAIT_GEN(unsigned long)
WAIT_GEN(unsigned long long)

INT_REDUCTION_DEF_GEN(int, int)
INT_REDUCTION_DEF_GEN(short, short)
INT_REDUCTION_DEF_GEN(long, long)
INT_REDUCTION_DEF_GEN(long long, longlong)
FLOAT_REDUCTION_DEF_GEN(float, float)
FLOAT_REDUCTION_DEF_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_DEF_GEN(long double, longdouble)

RMA_DEF_GEN(float, float)
RMA_DEF_GEN(double, double)
RMA_DEF_GEN(char, char)
// RMA_DEF_GEN(long double, longdouble)
RMA_DEF_GEN(signed char, schar)
RMA_DEF_GEN(short, short)
RMA_DEF_GEN(int, int)
RMA_DEF_GEN(long, long)
RMA_DEF_GEN(long long, longlong)
RMA_DEF_GEN(unsigned char, uchar)
RMA_DEF_GEN(unsigned short, ushort)
RMA_DEF_GEN(unsigned int, uint)
RMA_DEF_GEN(unsigned long, ulong)
RMA_DEF_GEN(unsigned long long, ulonglong)
RMA_DEF_GEN(int8_t, int8)
RMA_DEF_GEN(int16_t, int16)
RMA_DEF_GEN(int32_t, int32)
RMA_DEF_GEN(int64_t, int64)
RMA_DEF_GEN(uint8_t, uint8)
RMA_DEF_GEN(uint16_t, uint16)
RMA_DEF_GEN(uint32_t, uint32)
RMA_DEF_GEN(uint64_t, uint64)
RMA_DEF_GEN(size_t, size)
RMA_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_STANDARD_DEF_GEN(int, int)
AMO_STANDARD_DEF_GEN(long, long)
AMO_STANDARD_DEF_GEN(long long, longlong)
AMO_STANDARD_DEF_GEN(unsigned int, uint)
AMO_STANDARD_DEF_GEN(unsigned long, ulong)
AMO_STANDARD_DEF_GEN(unsigned long long, ulonglong)
AMO_STANDARD_DEF_GEN(int32_t, int32)
AMO_STANDARD_DEF_GEN(int64_t, int64)
AMO_STANDARD_DEF_GEN(uint32_t, uint32)
AMO_STANDARD_DEF_GEN(uint64_t, uint64)
AMO_STANDARD_DEF_GEN(size_t, size)
AMO_STANDARD_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_EXTENDED_DEF_GEN(float, float)
AMO_EXTENDED_DEF_GEN(double, double)
AMO_EXTENDED_DEF_GEN(int, int)
AMO_EXTENDED_DEF_GEN(long, long)
AMO_EXTENDED_DEF_GEN(long long, longlong)
AMO_EXTENDED_DEF_GEN(unsigned int, uint)
AMO_EXTENDED_DEF_GEN(unsigned long, ulong)
AMO_EXTENDED_DEF_GEN(unsigned long long, ulonglong)
AMO_EXTENDED_DEF_GEN(int32_t, int32)
AMO_EXTENDED_DEF_GEN(int64_t, int64)
AMO_EXTENDED_DEF_GEN(uint32_t, uint32)
AMO_EXTENDED_DEF_GEN(uint64_t, uint64)
AMO_EXTENDED_DEF_GEN(size_t, size)
AMO_EXTENDED_DEF_GEN(ptrdiff_t, ptrdiff)

AMO_BITWISE_DEF_GEN(unsigned int, uint)
AMO_BITWISE_DEF_GEN(unsigned long, ulong)
AMO_BITWISE_DEF_GEN(unsigned long long, ulonglong)
AMO_BITWISE_DEF_GEN(int32_t, int32)
AMO_BITWISE_DEF_GEN(int64_t, int64)
AMO_BITWISE_DEF_GEN(uint32_t, uint32)
AMO_BITWISE_DEF_GEN(uint64_t, uint64)

WAIT_DEF_GEN(float, float)
WAIT_DEF_GEN(double, double)
// WAIT_DEF_GEN(long double, longdouble)
WAIT_DEF_GEN(char, char)
WAIT_DEF_GEN(signed char, schar)
WAIT_DEF_GEN(short, short)
WAIT_DEF_GEN(int, int)
WAIT_DEF_GEN(long, long)
WAIT_DEF_GEN(long long, longlong)
WAIT_DEF_GEN(unsigned char, uchar)
WAIT_DEF_GEN(unsigned short, ushort)
WAIT_DEF_GEN(unsigned int, uint)
WAIT_DEF_GEN(unsigned long, ulong)
WAIT_DEF_GEN(unsigned long long, ulonglong)
// clang-format on

}  // namespace rocshmem
