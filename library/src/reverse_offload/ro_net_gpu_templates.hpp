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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_GPU_TEMPLATES_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_GPU_TEMPLATES_HPP

#include "config.h"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include "context_ro_device.hpp"
#include "ro_net_internal.hpp"
#include "ro_net_team.hpp"
#include "wg_state.hpp"

namespace rocshmem {

template <typename T>
struct GetROType {
};

template <>
struct GetROType<char> {
    static constexpr ro_net_types Type = RO_NET_CHAR;
};

template <>
struct GetROType<unsigned char> {
    static constexpr ro_net_types Type = RO_NET_CHAR;
};

template <>
struct GetROType<signed char> {
    static constexpr ro_net_types Type = RO_NET_CHAR;
};

template <>
struct GetROType<unsigned short> {
    static constexpr ro_net_types Type = RO_NET_SHORT;
};

template <>
struct GetROType<unsigned int> {
    static constexpr ro_net_types Type = RO_NET_INT;
};

template <>
struct GetROType<unsigned long> {
    static constexpr ro_net_types Type = RO_NET_LONG;
};

template <>
struct GetROType<unsigned long long> {
    static constexpr ro_net_types Type = RO_NET_LONG_LONG;
};

template <>
struct GetROType<float> {
    static constexpr ro_net_types Type = RO_NET_FLOAT;
};

template <>
struct GetROType<double> {
    static constexpr ro_net_types Type = RO_NET_DOUBLE;
};

template <>
struct GetROType<int> {
    static constexpr ro_net_types Type = RO_NET_INT;
};

template <>
struct GetROType<short> {
    static constexpr ro_net_types Type = RO_NET_SHORT;
};

template <>
struct GetROType<long> {
    static constexpr ro_net_types Type = RO_NET_LONG;
};

template <>
struct GetROType<long long> {
    static constexpr ro_net_types Type = RO_NET_LONG_LONG;
};

template <>
struct GetROType<long double> {
    static constexpr ro_net_types Type = RO_NET_LONG_DOUBLE;
};

/******************************************************************************
 ********************************* DEVICE API *********************************
 *****************************************************************************/

template <typename T, ROC_SHMEM_OP Op>
__device__ void
ROContext::to_all(roc_shmem_team_t team,
                  T *dest,
                  const T *source,
                  int nreduce) {
    if (!is_thread_zero_in_block()) {
        __syncthreads();
        return;
    }

    ROTeam *team_obj = reinterpret_cast<ROTeam*>(team);

    build_queue_element(RO_NET_TEAM_TO_ALL,
                        dest,
                        (void*)source,
                        nreduce,
                        0,
                        0,
                        0,
                        0,
                        nullptr,
                        nullptr,
                        team_obj->mpi_comm,
                        ro_net_win_id,
                        (struct ro_net_wg_handle*)backend_ctx,
                        true,
                        Op,
                        GetROType<T>::Type);

    __syncthreads();
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void
ROContext::to_all(T *dest,
                  const T *source,
                  int nreduce,
                  int PE_start,
                  int logPE_stride,
                  int PE_size,
                  T *pWrk,
                  long *pSync) {
    if (!is_thread_zero_in_block()) {
        __syncthreads();
        return;
    }

    build_queue_element(RO_NET_TO_ALL,
                        dest,
                        (void*)source,
                        nreduce,
                        PE_start,
                        logPE_stride,
                        PE_size,
                        0,
                        pWrk,
                        pSync,
                        (MPI_Comm)NULL,
                        ro_net_win_id,
                        (struct ro_net_wg_handle*)backend_ctx,
                        true,
                        Op,
                        GetROType<T>::Type);

    __syncthreads();
}

template <typename T>
__device__ void
ROContext::put(T *dest, const T *source, size_t nelems, int pe) {
    size_t size {sizeof(T) * nelems};
    putmem((void*)dest, (void*)source, size, pe);
}

template <typename T>
__device__ void
ROContext::put_nbi(T *dest, const T *source, size_t nelems, int pe) {
    size_t size {sizeof(T) * nelems};
    putmem_nbi((void*)dest, (void*)source, size, pe);
}

template <typename T>
__device__ void
ROContext::p(T *dest, T value, int pe) {
     if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        long L_offset {reinterpret_cast<char*>(dest) - ipcImpl_.ipc_bases[my_pe]};
        ipcImpl_.ipcCopy(ipcImpl_.ipc_bases[pe] + L_offset,
                         reinterpret_cast<void*>(&value),
                         sizeof(T));
    } else {
        build_queue_element(RO_NET_P,
                            dest,
                            &value,
                            sizeof(T),
                            pe,
                            0,
                            0,
                            0,
                            nullptr,
                            nullptr,
                            (MPI_Comm)NULL,
                            ro_net_win_id,
                            (struct ro_net_wg_handle*)backend_ctx,
                            true);
    }
}

template <typename T>
__device__ T
ROContext::g(const T *source, int pe) {
     if (ipcImpl_.isIpcAvailable(my_pe, pe)) {
        const char *src_typed {reinterpret_cast<const char*>(source)};
        long L_offset {const_cast<char*>(src_typed) - ipcImpl_.ipc_bases[my_pe]};
        T dest;
        ipcImpl_.ipcCopy(&dest,
                         ipcImpl_.ipc_bases[pe] + L_offset,
                         sizeof(T));
        return dest;
    } else {
        int thread_id {get_flat_block_id()};
        int block_size {get_flat_block_size()};
        auto *wg_state {WGState::instance()};
        int offset {wg_state->get_global_buffer_id() * block_size + thread_id};

        char *base_dest {backend_ctx->g_ret};
        char *dest {&base_dest[offset * sizeof(int64_t)]};
        size_t nelems {sizeof(T)};
        get<T>((T*)dest, source, 1, pe);
        return *(reinterpret_cast<T*>(dest));
    }
}

template <typename T>
__device__ void
ROContext::get(T *dest, const T *source, size_t nelems, int pe) {
    size_t size {sizeof(T) * nelems};
    getmem((void*)dest, (void*)source, size, pe);
}

template <typename T>
__device__ void
ROContext::get_nbi(T *dest, const T *source, size_t nelems, int pe) {
    size_t size {sizeof(T) * nelems};
    getmem_nbi((void*)dest, (void*)source, size, pe);
}

template <typename T>
__device__ void
ROContext::broadcast(roc_shmem_team_t team,
                     T *dest,
                     const T *source,
                     int nelems,
                     int pe_root) {
     if (!is_thread_zero_in_block()) {
        __syncthreads();
        return;
    }

    ROTeam *team_obj {reinterpret_cast<ROTeam*>(team)};

    build_queue_element(RO_NET_TEAM_BROADCAST,
                        dest,
                        (void*)source,
                        nelems,
                        0,
                        0,
                        0,
                        pe_root,
                        nullptr,
                        nullptr,
                        team_obj->mpi_comm,
                        ro_net_win_id,
                        (struct ro_net_wg_handle*)backend_ctx,
                        true,
                        ROC_SHMEM_SUM,
                        GetROType<T>::Type);

    __syncthreads();
}

template <typename T>
__device__ void
ROContext::broadcast(T *dest,
                     const T *source,
                     int nelems,
                     int pe_root,
                     int pe_start,
                     int log_pe_stride,
                     int pe_size,
                     long *p_sync) {
     if (!is_thread_zero_in_block()) {
        __syncthreads();
        return;
    }

    build_queue_element(RO_NET_BROADCAST,
                        dest,
                        (void*)source,
                        nelems,
                        pe_start,
                        log_pe_stride,
                        pe_size,
                        pe_root,
                        nullptr,
                        p_sync,
                        (MPI_Comm)NULL,
                        ro_net_win_id,
                        (struct ro_net_wg_handle*)backend_ctx,
                        true,
                        ROC_SHMEM_SUM,
                        GetROType<T>::Type);

    __syncthreads();
}

template <typename T>
__device__ void
ROContext::alltoall(roc_shmem_team_t team,
                    T *dest,
                    const T *source,
                    int nelems)
{
     if (!is_thread_zero_in_block()) {
        __syncthreads();
        return;
    }

    ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);

    build_queue_element(RO_NET_ALLTOALL,
                        dest,
                        (void *)source,
                        nelems,
                        0,
                        0,
                        0,
                        0,
                        team_obj->ata_buffer,
                        nullptr,
                        team_obj->mpi_comm,
                        ro_net_win_id,
                        (struct ro_net_wg_handle *)backend_ctx,
                        true,
                        ROC_SHMEM_SUM,
                        GetROType<T>::Type);

    __syncthreads();
}

template <typename T>
__device__ void
ROContext::fcollect(roc_shmem_team_t team,
                    T *dest,
                    const T *source,
                    int nelems)
{
     if (!is_thread_zero_in_block()) {
        __syncthreads();
        return;
    }

    ROTeam *team_obj = reinterpret_cast<ROTeam *>(team);

    build_queue_element(RO_NET_FCOLLECT,
                        dest,
                        (void *)source,
                        nelems,
                        0,
                        0,
                        0,
                        0,
                        team_obj->ata_buffer,
                        nullptr,
                        team_obj->mpi_comm,
                        ro_net_win_id,
                        (struct ro_net_wg_handle *)backend_ctx,
                        true,
                        ROC_SHMEM_SUM,
                        GetROType<T>::Type);

    __syncthreads();
}

/**
 * WG and WAVE level API
 */

template <typename T>
__device__ void
ROContext::put_wg(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    putmem_wg((void*) dest, (void*) source, size, pe);
}

template <typename T>
__device__ void
ROContext::put_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    putmem_nbi_wg((void*) dest, (void*) source, size, pe);
}

template <typename T>
__device__ void
ROContext::put_wave(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    putmem_wave((void*) dest, (void*) source, size, pe);
}

template <typename T>
__device__ void
ROContext::put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    putmem_nbi_wave((void*) dest, (void*) source, size, pe);
}

template <typename T>
__device__ void
ROContext::get_wg(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    getmem_wg((void*)dest, (void*) source, size, pe);
}

template <typename T>
__device__ void
ROContext::get_nbi_wg(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    getmem_nbi_wg((void*)dest, (void*) source, size, pe);
}

template <typename T>
__device__ void
ROContext::get_wave(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    getmem_wave((void*)dest, (void*) source, size, pe);
}

template <typename T>
__device__ void
ROContext::get_nbi_wave(T *dest, const T *source, size_t nelems, int pe) {
    size_t size = sizeof(T) * nelems;
    getmem_nbi_wave((void*)dest, (void*) source, size, pe);
}

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_GPU_TEMPLATES_HPP
