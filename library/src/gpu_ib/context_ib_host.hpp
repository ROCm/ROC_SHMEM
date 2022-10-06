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

#ifndef ROCSHMEM_LIBRARY_SRC_GPU_IB_CONTEXT_IB_HOST_HPP
#define ROCSHMEM_LIBRARY_SRC_GPU_IB_CONTEXT_IB_HOST_HPP

#include "context.hpp"

namespace rocshmem {

class GPUIBHostContext : public Context {
 public:
    /* Pointer to the backend's host interface */
    HostInterface *host_interface {nullptr};

    /* An MPI Window implements a context */
    WindowInfo *context_window_info {nullptr};

    __host__
    GPUIBHostContext(Backend *b,
                     int64_t options);

    __host__
    ~GPUIBHostContext();

    /* Host functions */
    template <typename T>
    __host__ void
    p(T *dest,
      T value,
      int pe);

    template <typename T>
    __host__ T
    g(const T *source,
      int pe);

    template <typename T>
    __host__ void
    put(T *dest,
        const T *source,
        size_t nelems,
        int pe);

    template <typename T>
    __host__ void
    get(T *dest,
        const T *source,
        size_t nelems,
        int pe);

    template <typename T>
    __host__ void
    put_nbi(T *dest,
            const T *source,
            size_t nelems,
            int pe);

    template <typename T>
    __host__ void
    get_nbi(T *dest,
            const T *source,
            size_t nelems,
            int pe);

    __host__ void
    putmem(void *dest,
           const void *source,
           size_t nelems,
           int pe);

    __host__ void
    getmem(void *dest,
           const void *source,
           size_t nelems,
           int pe);

    __host__ void
    putmem_nbi(void *dest,
               const void *source,
               size_t nelems,
               int pe);

    __host__ void
    getmem_nbi(void *dest,
               const void *source,
               size_t size,
               int pe);

    __host__ void
    amo_add(void *dst,
            int64_t value,
            int64_t cond,
            int pe);

    __host__ void
    amo_cas(void *dst,
            int64_t value,
            int64_t cond,
            int pe);

    __host__ int64_t
    amo_fetch_add(void *dst,
                  int64_t value,
                  int64_t cond,
                  int pe);

    __host__ int64_t
    amo_fetch_cas(void *dst,
                  int64_t value,
                  int64_t cond,
                  int pe);

    __host__ void
    fence();

    __host__ void
    quiet();

    __host__ void
    barrier_all();

    __host__ void
    sync_all();

    template <typename T>
    __host__ void
    broadcast(T *dest,
              const T *source,
              int nelems,
              int pe_root,
              int pe_start,
              int log_pe_stride,
              int pe_size,
              long *p_sync);  // NOLINT(runtime/int)

    template <typename T>
    __host__ void
    broadcast(roc_shmem_team_t team,
              T *dest,
              const T *source,
              int nelems,
              int pe_root);

    template <typename T, ROC_SHMEM_OP Op>
    __host__ void
    to_all(T *dest,
           const T *source,
           int nreduce,
           int pe_start,
           int log_pe_stride,
           int pe_size,
           T *p_wrk,
           long *p_sync);  // NOLINT(runtime/int)

    template <typename T, ROC_SHMEM_OP Op>
    __host__ void
    to_all(roc_shmem_team_t team,
           T *dest,
           const T *source,
           int nreduce);

    template <typename T>
    __host__ void
    wait_until(T *ptr,
               roc_shmem_cmps cmp,
               T val);

    template <typename T>
    __host__ int
    test(T *ptr,
         roc_shmem_cmps cmp,
         T val);
};

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_GPU_IB_CONTEXT_IB_HOST_HPP
