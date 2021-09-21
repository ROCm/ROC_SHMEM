/******************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LIBRARY_SRC_HOST_HOST_HPP_
#define LIBRARY_SRC_HOST_HOST_HPP_

/**
 * @file host.hpp
 * Defines the HostInterface class.
 *
 * The file contains the HostInterface class that defines all the
 * host-facing functions that will be used by all host contexts of
 * any backend type.
 */

#include <mpi.h>

#include <map>

#include <roc_shmem.hpp>
#include "hdp_policy.hpp"

class WindowInfo {
    MPI_Win win;
    void *win_start;
    void *win_end;

 public:
    WindowInfo(MPI_Win _win,
               void *_start,
               size_t size)
        : win(_win),
          win_start(_start),
          win_end(reinterpret_cast<char*>(_start) + size) {
    }

    MPI_Win
    get_win() const {
        return win;
    }

    void*
    get_start() const {
        return win_start;
    }

    void*
    get_end() const {
        return win_end;
    }

    void
    set_win(MPI_Win _win) {
        win = _win;
    }

    void
    set_start(void *_start) {
        win_start = _start;
    }

    void
    set_end(void *_end) {
        win_end = _end;
    }
};

class HostInterface {
 private:
    /*
     * A pointer to the Backend's hdp policy
     */
    HdpPolicy *hdp_policy = nullptr;

    MPI_Comm host_comm_world;

    int my_pe;
    int num_pes;

    /* MPI window for hdp flushing */
    // TODO(rozambre): enable for rocm 4.5
    // MPI_Win hdp_win;

    /*
     * Data structure that stores the parameters defining the active
     * set of PEs in a collective. This struct also serves as a key
     * into the comm_map map.
     */
    struct active_set_key {
        int pe_start;
        int log_pe_stride;
        int pe_size;

        active_set_key(int _pe_start,
                       int _log_pe_stride,
                       int _pe_size)
            : pe_start(_pe_start),
              log_pe_stride(_log_pe_stride),
              pe_size(_pe_size) {
        }

        bool
        operator< (const active_set_key& key) const {
            return pe_start < key.pe_start ||
                   (pe_start == key.pe_start &&
                       log_pe_stride < key.log_pe_stride) ||
                   (pe_start == key.pe_start &&
                       log_pe_stride == key.log_pe_stride &&
                       pe_size < key.pe_size);
        }
    };

    /*
     * Map of active set descriptors to MPI communicators
     */
    std::map<active_set_key, MPI_Comm> comm_map;

    /**************************************************************************
     **************************** INTERNAL METHODS ****************************
     *************************************************************************/
    __host__ void
    flush_remote_hdps();

    __host__ void
    flush_remote_hdp(int pe);

    __host__ void
    initiate_put(void *dest,
                 const void *source,
                 size_t nelems,
                 int pe,
                 WindowInfo *window_info);

    __host__ void
    initiate_get(void *dest,
                 const void *source,
                 size_t nelems,
                 int pe,
                 WindowInfo *window_info);

    __host__ void
    complete_all(MPI_Win win);

    __host__ MPI_Aint
    compute_offset(const void *dest,
                   void *win_start,
                   void *win_end);

    __host__ MPI_Comm
    get_mpi_comm(int pe_start,
                 int log_pe_stride,
                 int pe_size);

    __host__ MPI_Op
    get_mpi_op(ROC_SHMEM_OP Op);

    template <typename T>
    __host__ MPI_Datatype
    get_mpi_type();

    template <typename T>
    __host__ int
    compare(roc_shmem_cmps cmp,
            T input_val,
            T target_val);

    template <typename T>
    __host__ int
    test_and_compare(MPI_Aint offset,
                     MPI_Datatype mpi_type,
                     roc_shmem_cmps cmp,
                     T val,
                     MPI_Win win);

 public:
    __host__
    HostInterface(HdpPolicy *hdp_policy,
                  MPI_Comm roc_shmem_comm);

    __host__
    ~HostInterface();

    MPI_Comm
    get_comm_world() {
        return host_comm_world;
    }

    /**************************************************************************
     ***************************** HOST FUNCTIONS *****************************
     *************************************************************************/
    template <typename T>
    __host__ void
    p(T *dest,
      T value,
      int pe,
      WindowInfo *window_info);

    template <typename T>
    __host__ T
    g(const T *source,
      int pe,
      WindowInfo *window_info);

    template <typename T>
    __host__ void
    put(T *dest,
        const T *source,
        size_t nelems,
        int pe,
        WindowInfo *window_info);

    template <typename T>
    __host__ void
    get(T *dest,
        const T *source,
        size_t nelems,
        int pe,
        WindowInfo *window_info);

    template <typename T>
    __host__ void
    put_nbi(T *dest,
            const T *source,
            size_t nelems,
            int pe,
            WindowInfo *window_info);

    template <typename T>
    __host__ void
    get_nbi(T *dest,
            const T *source,
            size_t nelems,
            int pe,
            WindowInfo *window_info);

    __host__ void
    putmem(void *dest,
           const void *source,
           size_t nelems,
           int pe,
           WindowInfo *window_info);

    __host__ void
    getmem(void *dest,
           const void *source,
           size_t nelems,
           int pe,
           WindowInfo *window_info);

    __host__ void
    putmem_nbi(void *dest,
               const void *source,
               size_t nelems,
               int pe,
               WindowInfo *window_info);

    __host__ void
    getmem_nbi(void *dest,
               const void *source,
               size_t size,
               int pe,
               WindowInfo *window_info);

    __host__ void
    amo_add(void *dst,
            int64_t value,
            int64_t cond,
            int pe,
            WindowInfo *window_info);

    __host__ void
    amo_cas(void *dst,
            int64_t value,
            int64_t cond,
            int pe,
            WindowInfo *window_info);

    __host__ int64_t
    amo_fetch_add(void *dst,
                  int64_t value,
                  int64_t cond,
                  int pe,
                  WindowInfo *window_info);

    __host__ int64_t
    amo_fetch_cas(void *dst,
                  int64_t value,
                  int64_t cond,
                  int pe,
                  WindowInfo *window_info);

    __host__ void
    fence(WindowInfo *window_info);

    __host__ void
    quiet(WindowInfo *window_info);

    __host__ void
    barrier_all(WindowInfo *window_info);

    __host__ void
    barrier_for_sync();

    __host__ void
    sync_all(WindowInfo *window_info);

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

    template <typename T>
    __host__ void
    wait_until(T *ptr,
               roc_shmem_cmps cmp,
               T val,
               WindowInfo *window_info);

    template <typename T>
    __host__ int
    test(T *ptr,
         roc_shmem_cmps cmp,
         T val,
         WindowInfo *window_info);
};

#endif  // LIBRARY_SRC_HOST_HOST_HPP_
