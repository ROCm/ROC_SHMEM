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
#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_HOST_TEMPLATES_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_HOST_TEMPLATES_HPP_

#include "config.h"  // NOLINT(build/include_subdir)
#include "src/host/host_templates.hpp"

namespace rocshmem {

template <typename T>
__host__ void ROHostContext::p(T *dest, T value, int pe) {
  DPRINTF("Function: gpu_ib_host_p\n");

  host_interface->p<T>(dest, value, pe, context_window_info);
}

template <typename T>
__host__ T ROHostContext::g(const T *source, int pe) {
  DPRINTF("Function: gpu_ib_host_g\n");

  return host_interface->g<T>(source, pe, context_window_info);
}

template <typename T>
__host__ void ROHostContext::put(T *dest, const T *source, size_t nelems,
                                 int pe) {
  DPRINTF("Function: gpu_ib_host_put\n");

  host_interface->put<T>(dest, source, nelems, pe, context_window_info);
}

template <typename T>
__host__ void ROHostContext::get(T *dest, const T *source, size_t nelems,
                                 int pe) {
  DPRINTF("Function: gpu_ib_host_get\n");

  host_interface->get<T>(dest, source, nelems, pe, context_window_info);
}

template <typename T>
__host__ void ROHostContext::put_nbi(T *dest, const T *source, size_t nelems,
                                     int pe) {
  DPRINTF("Function: gpu_ib_host_put_nbi\n");

  host_interface->put_nbi<T>(dest, source, nelems, pe, context_window_info);
}

template <typename T>
__host__ void ROHostContext::get_nbi(T *dest, const T *source, size_t nelems,
                                     int pe) {
  DPRINTF("Function: gpu_ib_host_get_nbi\n");

  host_interface->get_nbi<T>(dest, source, nelems, pe, context_window_info);
}

template <typename T>
__host__ void ROHostContext::amo_add(void *dst, T value, int pe) {
  DPRINTF("Function: ro_net_host_amo_add\n");

  host_interface->amo_add(dst, value, pe, context_window_info);
}

template <typename T>
__host__ void ROHostContext::amo_cas(void *dst, T value, T cond, int pe) {
  DPRINTF("Function: ro_net_host_amo_cas\n");

  host_interface->amo_cas(dst, value, cond, pe, context_window_info);
}

template <typename T>
__host__ T ROHostContext::amo_fetch_add(void *dst, T value, int pe) {
  DPRINTF("Function: ro_net_host_amo_fetch_add\n");

  return host_interface->amo_fetch_add(dst, value, pe, context_window_info);
}

template <typename T>
__host__ T ROHostContext::amo_fetch_cas(void *dst, T value, T cond, int pe) {
  DPRINTF("Function: ro_net_host_amo_fetch_cas\n");

  return host_interface->amo_fetch_cas(dst, value, cond, pe,
                                       context_window_info);
}

template <typename T>
__host__ void ROHostContext::broadcast(T *dest, const T *source, int nelems,
                                       int pe_root, int pe_start,
                                       int log_pe_stride, int pe_size,
                                       long *p_sync) {
  DPRINTF("Function: gpu_ib_host_broadcast\n");

  host_interface->broadcast<T>(dest, source, nelems, pe_root, pe_start,
                               log_pe_stride, pe_size, p_sync);
}

template <typename T>
__host__ void ROHostContext::broadcast(roc_shmem_team_t team, T *dest,
                                       const T *source, int nelems,
                                       int pe_root) {
  DPRINTF("Function: Team-based ro_net_host_broadcast\n");

  host_interface->broadcast<T>(team, dest, source, nelems, pe_root);
}

template <typename T, ROC_SHMEM_OP Op>
__host__ void ROHostContext::to_all(T *dest, const T *source, int nreduce,
                                    int pe_start, int log_pe_stride,
                                    int pe_size, T *p_wrk, long *p_sync) {
  DPRINTF("Function: gpu_ib_host_to_all\n");

  host_interface->to_all<T, Op>(dest, source, nreduce, pe_start, log_pe_stride,
                                pe_size, p_wrk, p_sync);
}

template <typename T, ROC_SHMEM_OP Op>
__host__ void ROHostContext::to_all(roc_shmem_team_t team, T *dest,
                                    const T *source, int nreduce) {
  DPRINTF("Function: Team-based ro_net_host_to_all\n");

  host_interface->to_all<T, Op>(team, dest, source, nreduce);
}

template <typename T>
__host__ void ROHostContext::wait_until(T *ptr, roc_shmem_cmps cmp, T val) {
  DPRINTF("Function: gpu_ib_host_wait_until\n");

  host_interface->wait_until<T>(ptr, cmp, val, context_window_info);
}

template <typename T>
__host__ int ROHostContext::test(T *ptr, roc_shmem_cmps cmp, T val) {
  DPRINTF("Function: gpu_ib_host_test\n");

  return host_interface->test<T>(ptr, cmp, val, context_window_info);
}

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_HOST_TEMPLATES_HPP_
