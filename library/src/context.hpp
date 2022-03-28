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

#ifndef ROCSHMEM_LIBRARY_SRC_CONTEXT_HPP
#define ROCSHMEM_LIBRARY_SRC_CONTEXT_HPP

#include <hip/hip_runtime.h>

#include "backend_type.hpp"
#include "device_mutex.hpp"
#include "ipc_policy.hpp"
#include "fence_policy.hpp"
#include "stats.hpp"
#include "wf_coal_policy.hpp"
#include "host.hpp"

namespace rocshmem {

class Backend;

/**
 * @file context.hpp
 * @brief Context class corresponds directly to an OpenSHMEM context.
 *
 * GPUs perform networking operations on a context that is created by the
 * application programmer or a "default context" managed by the runtime.
 *
 * Contexts can be allocated in shared memory, in which case they are private
 * to the creating workgroup, or they can be allocated in global memory, in
 * which case they are shareable across workgroups.
 *
 * This is an 'abstract' class, as much as there is such a thing on a GPU.
 * It uses 'type' to dispatch to a derived class for most of the interesting
 * behavior.
 */
class Context {
 public:
    __host__
    Context(const Backend &handle,
            bool shareable);

    __device__
    Context(const Backend &handle,
            bool shareable);

    /*
     * Dispatch functions to get runtime polymorphism without 'virtual' or
     * function pointers. Each one of these guys will use 'type' to
     * static_cast themselves and dispatch to the appropriate derived class.
     * It's basically doing part of what the 'virtual' keyword does, so when
     * we get that working in ROCm it will be super easy to adapt to it by
     * just removing the dispatch implementations.
     *
     * No comments for these guys since its basically the same as in the
     * roc_shmem.hpp public header.
     */

    /**************************************************************************
     ***************************** DEVICE METHODS *****************************
     *************************************************************************/
    template <typename T>
    __device__ void
    wait_until(T* ptr,
              roc_shmem_cmps cmp,
              T val);

    template <typename T>
    __device__ int
    test(T* ptr,
         roc_shmem_cmps cmp,
         T val);

    __device__ void
    threadfence_system();

    __device__ void
    ctx_destroy();

    __device__ void
    putmem(void* dest,
           const void* source,
           size_t nelems,
           int pe);

    __device__ void
    getmem(void* dest,
           const void* source,
           size_t nelems,
           int pe);

    __device__ void
    putmem_nbi(void* dest,
               const void* source,
               size_t nelems,
               int pe);

    __device__ void
    getmem_nbi(void* dest,
               const void* source,
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

    __device__ int64_t
    amo_fetch(void* dst,
              int64_t value,
              int64_t cond,
              int pe,
              uint8_t atomic_op);

    __device__ void
    sync_all();

    __device__ void
    amo_add(void* dst,
            int64_t value,
            int64_t cond,
            int pe);

    __device__ void
    amo_cas(void* dst,
            int64_t value,
            int64_t cond,
            int pe);

    __device__ int64_t
    amo_fetch_add(void* dst,
                  int64_t value,
                  int64_t cond,
                  int pe);

    __device__ int64_t
    amo_fetch_cas(void* dst,
                  int64_t value,
                  int64_t cond,
                  int pe);

    template <typename T>
    __device__ void
    p(T* dest,
      T value,
      int pe);

    template <typename T>
    __device__ T
    g(T* source,
      int pe);

    template <typename T, ROC_SHMEM_OP Op>
    __device__ void
    to_all(T* dest,
           const T* source,
           int nreduce,
           int PE_start,
           int logPE_stride,
           int PE_size,
           T* pWrk,
           long* pSync);  // NOLINT(runtime/int)

    template <typename T, ROC_SHMEM_OP Op>
    __device__ void
    to_all(roc_shmem_team_t team,
           T* dest,
           const T* source,
           int nreduce);

    template <typename T>
    __device__ void
    put(T* dest,
        const T* source,
        size_t nelems,
        int pe);

    template <typename T>
    __device__ void
    put_nbi(T* dest,
            const T* source,
            size_t nelems,
            int pe);

    template <typename T>
    __device__ void
    get(T* dest,
        const T* source,
        size_t nelems,
        int pe);

    template <typename T>
    __device__ void
    get_nbi(T* dest,
            const T* source,
            size_t nelems,
            int pe);

    template <typename T>
    __device__ void
    broadcast(roc_shmem_team_t team,
              T* dest,
              const T* source,
              int nelems,
              int pe_root);

    template <typename T>
    __device__ void
    broadcast(T* dest,
              const T* source,
              int nelems,
              int pe_root,
              int pe_start,
              int log_pe_stride,
              int pe_size,
              long* p_sync);  // NOLINT(runtime/int)

    __device__ void
    putmem_wg(void* dest,
              const void* source,
              size_t nelems,
              int pe);

    __device__ void
    getmem_wg(void* dest,
              const void* source,
              size_t nelems,
              int pe);

    __device__ void
    putmem_nbi_wg(void* dest,
                  const void* source,
                  size_t nelems,
                  int pe);

    __device__ void
    getmem_nbi_wg(void* dest,
                  const void* source,
                  size_t size,
                  int pe);

    __device__ void
    putmem_wave(void* dest,
                const void* source,
                size_t nelems,
                int pe);

    __device__ void
    getmem_wave(void* dest,
                const void* source,
                size_t nelems,
                int pe);

    __device__ void
    putmem_nbi_wave(void* dest,
                    const void* source,
                    size_t nelems,
                    int pe);

    __device__ void
    getmem_nbi_wave(void* dest,
                    const void* source,
                    size_t size,
                    int pe);

    template <typename T>
    __device__ void
    put_wg(T* dest,
           const T* source,
           size_t nelems,
           int pe);

    template <typename T>
    __device__ void
    put_nbi_wg(T* dest,
               const T* source,
               size_t nelems,
               int pe);

    template <typename T>
    __device__ void
    get_wg(T* dest,
           const T* source,
           size_t nelems,
           int pe);

    template <typename T>
    __device__ void
    get_nbi_wg(T* dest,
               const T* source,
               size_t nelems,
               int pe);

    template <typename T>
    __device__ void
    put_wave(T* dest,
             const T* source,
             size_t nelems,
             int pe);

    template <typename T>
    __device__ void
    put_nbi_wave(T* dest,
                 const T* source,
                 size_t nelems,
                 int pe);

    template <typename T>
    __device__ void
    get_wave(T* dest,
             const T* source,
             size_t nelems,
             int pe);

    template <typename T>
    __device__ void
    get_nbi_wave(T* dest,
                 const T* source,
                 size_t nelems,
                 int pe);

    /**************************************************************************
     ****************************** HOST METHODS ******************************
     *************************************************************************/
    template <typename T>
    __host__ void
    p(T* dest,
      T value,
      int pe);

    template <typename T>
    __host__ T
    g(const T* source,
      int pe);

    template <typename T>
    __host__ void
    put(T* dest,
        const T* source,
        size_t nelems,
        int pe);

    template <typename T>
    __host__ void
    get(T* dest,
        const T* source,
        size_t nelems,
        int pe);

    template <typename T>
    __host__ void
    put_nbi(T* dest,
            const T* source,
            size_t nelems,
            int pe);

    template <typename T>
    __host__ void
    get_nbi(T* dest,
            const T* source,
            size_t nelems,
            int pe);

    __host__ void
    putmem(void* dest,
           const void* source,
           size_t nelems,
           int pe);

    __host__ void
    getmem(void* dest,
           const void* source,
           size_t nelems,
           int pe);

    __host__ void
    putmem_nbi(void* dest,
               const void* source,
               size_t nelems,
               int pe);

    __host__ void
    getmem_nbi(void* dest,
               const void* source,
               size_t size,
               int pe);

    __host__ void
    amo_add(void* dst,
            int64_t value,
            int64_t cond,
            int pe);

    __host__ void
    amo_cas(void* dst,
            int64_t value,
            int64_t cond,
            int pe);

    __host__ int64_t
    amo_fetch_add(void* dst,
                  int64_t value,
                  int64_t cond,
                  int pe);

    __host__ int64_t
    amo_fetch_cas(void* dst,
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
    broadcast(T* dest,
              const T* source,
              int nelems,
              int pe_root,
              int pe_start,
              int log_pe_stride,
              int pe_size,
              long* p_sync);  // NOLINT(runtime/int)

    template <typename T>
    __host__ void
    broadcast(roc_shmem_team_t team,
              T* dest,
              const T* source,
              int nelems,
              int pe_root);

    template <typename T, ROC_SHMEM_OP Op>
    __host__ void
    to_all(T* dest,
           const T* source,
           int nreduce,
           int PE_start,
           int logPE_stride,
           int PE_size,
           T* pWrk,
           long* pSync);  // NOLINT(runtime/int)

    template <typename T, ROC_SHMEM_OP Op>
    __host__ void
    to_all(roc_shmem_team_t team,
           T* dest,
           const T* source,
           int nreduce);

    template <typename T>
    __host__ void
    wait_until(T* ptr,
               roc_shmem_cmps cmp,
               T val);

    template <typename T>
    __host__ int
    test(T* ptr,
         roc_shmem_cmps cmp,
         T val);

 public:
    /**
     * @brief Set the fence policy using a runtime option
     *
     * @param[in] options interpreted as a bitfield using bitwise operations
     */
    __device__ void
    setFence(long options) {
        fence_ = Fence(options);
    };

    /**************************************************************************
     ***************************** PUBLIC MEMBERS *****************************
     *************************************************************************/
    /**
     * @brief Duplicated local copy of backend's num_pes
     */
    int num_pes {0};

    /**
     * @brief Duplicated local copy of backend's my_pe
     */
    int my_pe {-1};

    /**
     * @brief Used to static dispatch to correct context type
     *
     * Used only to dispatch to the correct derived type. This is used to
     * get around the fact that there are no virtual functions for device code.
     * See the 'DISPATCH' macro and usage for more details.
     */
    BackendType type {BackendType::GPU_IB_BACKEND};

    /**
     * @brief Stats common to all types of device contexts.
     */
    ROCStats ctxStats {};

    /**
     * @brief Stats common to all types of host contexts.
     */
    ROCHostStats ctxHostStats {};

 protected:
    /**************************************************************************
     ***************************** POLICY MEMBERS *****************************
     *************************************************************************/
    /**
     * @brief Lock to prevent data races on shared data
     */
    DeviceMutex dev_mtx_ {};

    /**
     * @brief Coalesce policy for 'multi' configuration builds
     */
    WavefrontCoalescer wf_coal_ {};

    /**
     * @brief Inter-Process Communication (IPC) interface for context class
     *
     * This member is an interface to allow intra-node interprocess
     * communication through shared memory.
     */
    IpcImpl ipcImpl_ {};

    /**
     * @brief Controls fence behavior in device code
     */
    Fence fence_ {};
};

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_CONTEXT_HPP
