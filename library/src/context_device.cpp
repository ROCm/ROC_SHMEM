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

#include "config.h"  // NOLINT(build/include_subdir)
#include "src/backend_bc.hpp"
#include "src/context_incl.hpp"
#include "src/util.hpp"

namespace rocshmem {

__device__ Context::Context(Backend* handle, bool shareable)
    : num_pes(handle->getNumPEs()),
      my_pe(handle->getMyPE()),
      fence_(shareable),
      dev_mtx_(shareable) {
  /*
   * Device-side context constructor is a work-group collective, so make
   * sure all the members have their default values before returning.
   *
   * Each thread is essentially initializing the same thing right over the
   * top of each other for all the default values in context.hh (and the
   * initializer list). It's not incorrect, but it is weird and probably
   * wasteful.
   *
   * TODO: Might consider refactoring so that constructor is always called
   * from a single thread, and the parallel portion of initialization can be
   * a separate function. This requires reworking all the derived classes
   * since their constructors actually make use of all the threads to boost
   * performance.
   */
  __syncthreads();
}

/******************************************************************************
 ********************** CONTEXT DISPATCH IMPLEMENTATIONS **********************
 *****************************************************************************/

__device__ void Context::threadfence_system() {
  DISPATCH(threadfence_system());
}

__device__ void Context::ctx_create() {
  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_CREATE);
  }

  DISPATCH(ctx_create());
}

__device__ void Context::ctx_destroy() {
  if (is_thread_zero_in_block()) {
    ctxStats.incStat(NUM_FINALIZE);
    device_backend_proxy->globalStats.accumulateStats(ctxStats);
  }

  DISPATCH(ctx_destroy());
}

__device__ void Context::putmem(void* dest, const void* source, size_t nelems,
                                int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT);

  DISPATCH(putmem(dest, source, nelems, pe));
}

__device__ void Context::getmem(void* dest, const void* source, size_t nelems,
                                int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET);

  DISPATCH(getmem(dest, source, nelems, pe));
}

__device__ void Context::putmem_nbi(void* dest, const void* source,
                                    size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI);

  DISPATCH(putmem_nbi(dest, source, nelems, pe));
}

__device__ void Context::getmem_nbi(void* dest, const void* source, size_t size,
                                    int pe) {
  if (size == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI);

  DISPATCH(getmem_nbi(dest, source, size, pe));
}

__device__ void Context::fence() {
  ctxStats.incStat(NUM_FENCE);

  DISPATCH(fence());
}

__device__ void Context::fence(int pe) {
  ctxStats.incStat(NUM_FENCE);

  DISPATCH(fence(pe));
}

__device__ void Context::quiet() {
  ctxStats.incStat(NUM_QUIET);

  DISPATCH(quiet());
}

__device__ void* Context::shmem_ptr(const void* dest, int pe) {
  ctxStats.incStat(NUM_SHMEM_PTR);

  DISPATCH_RET_PTR(shmem_ptr(dest, pe));
}

__device__ void Context::barrier_all() {
  ctxStats.incStat(NUM_BARRIER_ALL);

  DISPATCH(barrier_all());
}

__device__ void Context::sync_all() {
  ctxStats.incStat(NUM_SYNC_ALL);

  DISPATCH(sync_all());
}

__device__ void Context::sync(roc_shmem_team_t team) {
  ctxStats.incStat(NUM_SYNC_ALL);

  DISPATCH(sync(team));
}

__device__ void Context::putmem_wg(void* dest, const void* source,
                                   size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_WG);

  DISPATCH(putmem_wg(dest, source, nelems, pe));
}

__device__ void Context::getmem_wg(void* dest, const void* source,
                                   size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_WG);

  DISPATCH(getmem_wg(dest, source, nelems, pe));
}

__device__ void Context::putmem_nbi_wg(void* dest, const void* source,
                                       size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI_WG);

  DISPATCH(putmem_nbi_wg(dest, source, nelems, pe));
}

__device__ void Context::getmem_nbi_wg(void* dest, const void* source,
                                       size_t size, int pe) {
  if (size == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI_WG);

  DISPATCH(getmem_nbi_wg(dest, source, size, pe));
}

__device__ void Context::putmem_wave(void* dest, const void* source,
                                     size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_WAVE);

  DISPATCH(putmem_wave(dest, source, nelems, pe));
}

__device__ void Context::getmem_wave(void* dest, const void* source,
                                     size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_WAVE);

  DISPATCH(getmem_wave(dest, source, nelems, pe));
}

__device__ void Context::putmem_nbi_wave(void* dest, const void* source,
                                         size_t nelems, int pe) {
  if (nelems == 0) {
    return;
  }

  ctxStats.incStat(NUM_PUT_NBI_WAVE);

  DISPATCH(putmem_nbi_wave(dest, source, nelems, pe));
}

__device__ void Context::getmem_nbi_wave(void* dest, const void* source,
                                         size_t size, int pe) {
  if (size == 0) {
    return;
  }

  ctxStats.incStat(NUM_GET_NBI_WAVE);

  DISPATCH(getmem_nbi_wave(dest, source, size, pe));
}

}  // namespace rocshmem
