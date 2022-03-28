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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_DEVICE_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_DEVICE_HPP

#include "context.hpp"

namespace rocshmem {

struct ro_net_wg_handle;

class ROContext : public Context {
    ro_net_wg_handle *backend_ctx = nullptr;

 public:
    __host__
    ROContext(const Backend &b,
              int64_t options);

    __device__
    ROContext(const Backend &b,
              int64_t options);

    /**************************************************************************
     ************************ CONTEXT DISPATCH METHODS ************************
     *************************************************************************/

    /**************************************************************************
     ***************************** DEVICE METHODS *****************************
     *************************************************************************/
    __device__ void
    threadfence_system();

    __device__ void
    ctx_destroy();

    __device__ void
    putmem(void *dest,
           const void *source,
           size_t nelems,
           int pe);

    __device__ void
    getmem(void *dest,
           const void *source,
           size_t nelems,
           int pe);

    __device__ void
    putmem_nbi(void *dest,
               const void *source,
               size_t nelems,
               int pe);

    __device__ void
    getmem_nbi(void *dest,
               const void *source,
               size_t size,
               int pe);

    __device__ void
    fence();

    __device__ void
    quiet();

    __device__ void*
    shmem_ptr(const void* dest,
              int pe);

    __device__ void
    barrier_all();

    __device__ void
    sync_all();

    __device__ void
    amo_add(void *dst,
            int64_t value,
            int64_t cond,
            int pe);

    __device__ void
    amo_cas(void *dst,
            int64_t value,
            int64_t cond,
            int pe);

    __device__ int64_t
    amo_fetch_add(void *dst,
                  int64_t value,
                  int64_t cond,
                  int pe);

    __device__ int64_t
    amo_fetch_cas(void *dst,
                  int64_t value,
                  int64_t cond,
                  int pe);

    template <typename T>
    __device__ void
    p(T *dest,
      T value,
      int pe);

    template <typename T>
    __device__ T
    g(const T *source,
      int pe);

    template <typename T, ROC_SHMEM_OP Op>
    __device__ void
    to_all(T *dest,
           const T *source,
           int nreduce,
           int PE_start,
           int logPE_stride,
           int PE_size,
           T *pWrk,
           long *pSync);  // NOLINT(runtime/int)

    template <typename T, ROC_SHMEM_OP Op>
    __device__ void
    to_all(roc_shmem_team_t team,
           T *dest,
           const T *source,
           int nreduce);

    template <typename T>
    __device__ void
    put(T *dest,
        const T *source,
        size_t nelems,
        int pe);

    template <typename T>
    __device__ void
    put_nbi(T *dest,
            const T *source,
            size_t nelems,
            int pe);

    template <typename T>
    __device__ void
    get(T *dest,
        const T *source,
        size_t nelems,
        int pe);

    template <typename T>
    __device__ void
    get_nbi(T *dest,
            const T *source,
            size_t nelems,
            int pe);

    template <typename T>
    __device__ void
    broadcast(roc_shmem_team_t team,
              T *dest,
              const T *source,
              int nelems,
              int pe_root);

    template <typename T>
    __device__ void
    broadcast(T *dest,
              const T *source,
              int nelems,
              int pe_root,
              int pe_start,
              int log_pe_stride,
              int pe_size,
              long *p_sync);  // NOLINT(runtime/int)

    __device__ void
    putmem_wg(void *dest,
              const void *source,
              size_t nelems,
              int pe);

    __device__ void
    getmem_wg(void *dest,
              const void *source,
              size_t nelems,
              int pe);

    __device__ void
    putmem_nbi_wg(void *dest,
                  const void *source,
                  size_t nelems,
                  int pe);

    __device__ void
    getmem_nbi_wg(void *dest,
                  const void *source,
                  size_t size,
                  int pe);

    __device__ void
    putmem_wave(void *dest,
                const void *source,
                size_t nelems,
                int pe);

    __device__ void
    getmem_wave(void *dest,
                const void *source,
                size_t nelems,
                int pe);

    __device__ void
    putmem_nbi_wave(void *dest,
                    const void *source,
                    size_t nelems,
                    int pe);

    __device__ void
    getmem_nbi_wave(void *dest,
                    const void *source,
                    size_t size,
                    int pe);

    template <typename T>
    __device__ void
    put_wg(T *dest,
           const T *source,
           size_t nelems,
           int pe);

    template <typename T>
    __device__ void
    put_nbi_wg(T *dest,
               const T *source,
               size_t nelems,
               int pe);

    template <typename T>
    __device__ void
    get_wg(T *dest,
           const T *source,
           size_t nelems,
           int pe);

    template <typename T>
    __device__ void
    get_nbi_wg(T *dest,
               const T *source,
               size_t nelems,
               int pe);

    template <typename T>
    __device__ void
    put_wave(T *dest,
             const T *source,
             size_t nelems,
             int pe);

    template <typename T>
    __device__ void
    put_nbi_wave(T *dest,
                 const T *source,
                 size_t nelems,
                 int pe);

    template <typename T>
    __device__ void
    get_wave(T *dest,
             const T *source,
             size_t nelems,
             int pe);

    template <typename T>
    __device__ void
    get_nbi_wave(T *dest,
                 const T *source,
                 size_t nelems,
                 int pe);
};

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_DEVICE_HPP
