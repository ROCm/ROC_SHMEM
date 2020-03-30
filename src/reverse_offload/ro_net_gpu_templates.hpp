/******************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef RO_NET_GPU_TEMPLATES_H
#define RO_NET_GPU_TEMPLATES_H

#include "config.h"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include "ro_net_internal.hpp"

template<typename T> struct GetROType { };

/* Add specializations here! */
template<> struct GetROType<float>
{ static constexpr ro_net_types Type = RO_NET_FLOAT; };

template<> struct GetROType<double>
{ static constexpr ro_net_types Type = RO_NET_DOUBLE; };

template<> struct GetROType<int>
{ static constexpr ro_net_types Type = RO_NET_INT; };

template<> struct GetROType<short>
{ static constexpr ro_net_types Type = RO_NET_SHORT; };

template<> struct GetROType<long>
{ static constexpr ro_net_types Type = RO_NET_LONG; };

template<> struct GetROType<long long>
{ static constexpr ro_net_types Type = RO_NET_LONG_LONG; };

template<> struct GetROType<long double>
{ static constexpr ro_net_types Type = RO_NET_LONG_DOUBLE; };

template <typename T, ROC_SHMEM_OP Op>
__device__ void
ROContext::to_all(T *dest, const T *source, int nreduce, int PE_start,
                  int logPE_stride, int PE_size, T *pWrk, long *pSync)
{
    if (!is_thread_zero_in_block()) {
        __syncthreads();
        return;
    }

    /*
     * Need to get out of template land here, since we must pack the type
     * and op info into the command queue at runtime.
     */
    build_queue_element(RO_NET_TO_ALL,
                        dest, (void *) source, nreduce, PE_start, logPE_stride,
                        PE_size, pWrk, pSync,
                        (struct ro_net_wg_handle *) backend_ctx, true,
                        Op, GetROType<T>::Type);

    __syncthreads();
}

template <typename T>
__device__ void
ROContext::put(T *dest, const T *source, size_t nelems, int pe)
{
    size_t size = sizeof(T) * nelems;
    build_queue_element(RO_NET_PUT, dest, (void *) source, size, pe, 0, 0,
                        nullptr, nullptr,
                        (struct ro_net_wg_handle *) backend_ctx, true);
}

template <typename T>
__device__ void
ROContext::put_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    size_t size = sizeof(T) * nelems;
    build_queue_element(RO_NET_PUT_NBI, dest, (void *) source, size, pe, 0,
                        0, nullptr, nullptr,
                        (struct ro_net_wg_handle *) backend_ctx, true);
}

template <typename T>
__device__ void
ROContext::p(T *dest, T value, int pe)
{
    build_queue_element(RO_NET_P, dest, &value, sizeof(T), pe, 0, 0, nullptr,
                        nullptr, (struct ro_net_wg_handle *) backend_ctx,
                        true);
}

template <typename T>
__device__ T
ROContext::g(T *source, int pe)
{
    assert("RO _g unimplemented\n");
}

template <typename T>
__device__ void
ROContext::get(T *dest, const T *source, size_t nelems, int pe)
{
    size_t size = sizeof(T) * nelems;
    build_queue_element(RO_NET_GET, dest, (void *) source, size, pe, 0, 0,
                        nullptr, nullptr,
                        (struct ro_net_wg_handle *) backend_ctx, true);
}

template <typename T>
__device__ void
ROContext::get_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    size_t size = sizeof(T) * nelems;
    build_queue_element(RO_NET_GET_NBI, dest, (void *) source, size, pe, 0,
                        0, nullptr, nullptr,
                        (struct ro_net_wg_handle *) backend_ctx, true);
}

#endif // RO_NET_GPU_TEMPLATES_H
