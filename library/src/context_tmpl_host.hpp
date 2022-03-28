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

#ifndef ROCSHMEM_LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP
#define ROCSHMEM_LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP

#include "backend_type.hpp"
#include "context_ib_host.hpp"
#include "context_ro_host.hpp"

namespace rocshmem {

template <typename T>
__host__ void
Context::p(T *dest,
           T value,
           int pe) {
    ctxHostStats.incStat(NUM_HOST_P);

    HOST_DISPATCH(p(dest, value, pe));
}

template <typename T>
__host__ T
Context::g(const T *source,
           int pe) {
    ctxHostStats.incStat(NUM_HOST_G);

    HOST_DISPATCH_RET(g(source, pe));
}

template <typename T>
__host__ void
Context::put(T *dest,
             const T *source,
             size_t nelems,
             int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_PUT);

    HOST_DISPATCH(put(dest, source, nelems, pe));
}

template <typename T>
__host__ void
Context::get(T *dest,
             const T *source,
             size_t nelems,
             int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_GET);

    HOST_DISPATCH(get(dest, source, nelems, pe));
}

template <typename T>
__host__ void
Context::put_nbi(T *dest,
                 const T *source,
                 size_t nelems,
                 int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_PUT_NBI);

    HOST_DISPATCH(put_nbi(dest, source, nelems, pe));
}

template <typename T>
__host__ void
Context::get_nbi(T *dest,
                 const T *source,
                 size_t nelems,
                 int pe) {
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_GET_NBI);

    HOST_DISPATCH(get_nbi(dest, source, nelems, pe));
}

template <typename T>
__host__ void
Context::broadcast(T *dest,
                   const T *source,
                   int nelems,
                   int pe_root,
                   int pe_start,
                   int log_pe_stride,
                   int pe_size,
                   long *p_sync) {  // NOLINT(runtime/int)
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_BROADCAST);

    HOST_DISPATCH(broadcast<T>(dest,
                          source,
                          nelems,
                          pe_root,
                          pe_start,
                          log_pe_stride,
                          pe_size,
                          p_sync));
}

template <typename T>
__host__ void
Context::broadcast(roc_shmem_team_t team,
                   T *dest,
                   const T *source,
                   int nelems,
                   int pe_root) {  // NOLINT(runtime/int)
    if (nelems == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_BROADCAST);

    HOST_DISPATCH(broadcast<T>(team,
                               dest,
                               source,
                               nelems,
                               pe_root));
}

template <typename T, ROC_SHMEM_OP Op>
__host__ void
Context::to_all(T *dest,
                const T *source,
                int nreduce,
                int PE_start,
                int logPE_stride,
                int PE_size,
                T *pWrk,
                long *pSync) {  // NOLINT(runtime/int)
    if (nreduce == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_TO_ALL);

    HOST_DISPATCH(to_all<PAIR(T, Op)>(dest,
                                      source,
                                      nreduce,
                                      PE_start,
                                      logPE_stride,
                                      PE_size,
                                      pWrk,
                                      pSync));
}

template <typename T, ROC_SHMEM_OP Op>
__host__ void
Context::to_all(roc_shmem_team_t team,
                T *dest,
                const T *source,
                int nreduce) {  // NOLINT(runtime/int)
    if (nreduce == 0) {
        return;
    }

    ctxHostStats.incStat(NUM_HOST_TO_ALL);

    HOST_DISPATCH(to_all<PAIR(T, Op)>(team,
                                      dest,
                                      source,
                                      nreduce));
}

template <typename T>
__host__ void
Context::wait_until(T *ptr,
                    roc_shmem_cmps cmp,
                    T val) {
    ctxHostStats.incStat(NUM_HOST_WAIT_UNTIL);

    HOST_DISPATCH(wait_until<T>(ptr, cmp, val));
}

template <typename T>
__host__ int
Context::test(T *ptr,
              roc_shmem_cmps cmp,
              T val) {
    ctxHostStats.incStat(NUM_HOST_TEST);

    HOST_DISPATCH_RET(test<T>(ptr, cmp, val));
}

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_CONTEXT_TMPL_HOST_HPP
