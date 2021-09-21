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
#ifndef RO_HOST_TEMPLATES_H
#define RO_HOST_TEMPLATES_H

#include "config.h"

#include "host/host_templates.hpp"

template <typename T>
__host__ void
ROHostContext::p(T *dest, T value, int pe)
{
    DPRINTF(("Function: gpu_ib_host_p\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dest);

    host_interface->p<T>(dest, value, pe, window_info);
}

template <typename T>
__host__ T
ROHostContext::g(const T *source, int pe)
{
    DPRINTF(("Function: gpu_ib_host_g\n"));

    WindowInfo *window_info = list_of_windows.get_window_info((void*) source);

    return host_interface->g<T>(source, pe, window_info);
}

template <typename T>
__host__ void
ROHostContext::put(T *dest, const T *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_put\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dest);

    host_interface->put<T>(dest, source, nelems, pe, window_info);
}

template <typename T>
__host__ void
ROHostContext::get(T *dest, const T *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_get\n"));

    WindowInfo *window_info = list_of_windows.get_window_info((void*) source);

    host_interface->get<T>(dest, source, nelems, pe, window_info);
}

template <typename T>
__host__ void
ROHostContext::put_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_put_nbi\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dest);

    host_interface->put_nbi<T>(dest, source, nelems, pe, window_info);
}

template <typename T>
__host__ void
ROHostContext::get_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_get_nbi\n"));

    WindowInfo *window_info = list_of_windows.get_window_info((void*) source);

    host_interface->get_nbi<T>(dest, source, nelems, pe, window_info);
}

template <typename T>
__host__ void
ROHostContext::broadcast(T *dest,
                         const T *source,
                         int nelems,
                         int pe_root,
                         int pe_start,
                         int log_pe_stride,
                         int pe_size,
                         long *p_sync)
{
    DPRINTF(("Function: gpu_ib_host_broadcast\n"));

    host_interface->broadcast<T>(dest, source, nelems, pe_root, pe_start, log_pe_stride, pe_size, p_sync);
}

template <typename T, ROC_SHMEM_OP Op>
__host__ void
ROHostContext::to_all(T *dest,
                      const T *source,
                      int nreduce,
                      int pe_start,
                      int log_pe_stride,
                      int pe_size,
                      T *p_wrk,
                      long *p_sync)
{
    DPRINTF(("Function: gpu_ib_host_to_all\n"));

    host_interface->to_all<T, Op>(dest, source, nreduce, pe_start, log_pe_stride, pe_size, p_wrk, p_sync);
}

template <typename T> __host__ void
ROHostContext::wait_until(T *ptr, roc_shmem_cmps cmp, T val)
{
    DPRINTF(("Function: gpu_ib_host_wait_until\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(ptr);

    host_interface->wait_until<T>(ptr, cmp, val, window_info);
}

template <typename T> __host__ int
ROHostContext::test(T *ptr, roc_shmem_cmps cmp, T val)
{
    DPRINTF(("Function: gpu_ib_host_test\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(ptr);

    return host_interface->test<T>(ptr, cmp, val, window_info);
}

#endif // RO_HOST_TEMPLATES_H
