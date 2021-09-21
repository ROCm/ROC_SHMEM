/******************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEMPLATES_HOST_H
#define TEMPLATES_HOST_H

#include <roc_shmem.hpp>

/**
 * @file templates_host.hpp
 * @brief Internal header that declares templates for ROC_SHMEM's implentation
 * of the user-facing host APIs.
 *
 * This file contains templates for the OpenSHMEM APIs that take have
 * hardcoded data types into the function name.
 */

/******************************************************************************
 **************************** HOST FUNCTIONS **********************************
 *****************************************************************************/

template <typename T>
__host__ void roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
                            size_t nelems, int pe);

template <typename T>
__host__ void roc_shmem_put(T *dest, const T *source, size_t nelems, int pe);

template <typename T>
__host__ void roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe);

template <typename T>
__host__ void roc_shmem_p(T *dest, T value, int pe);

template <typename T>
__host__ void roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source,
                            size_t nelems, int pe);

template <typename T>
__host__ void roc_shmem_get(T *dest, const T *source, size_t nelems, int pe);

template <typename T>
__host__ T roc_shmem_g(roc_shmem_ctx_t ctx, const T *source, int pe);

template <typename T>
__host__ T roc_shmem_g(const T *source, int pe);

template <typename T>
__host__ void roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *src,
                                size_t nelems, int pe);

template <typename T>
__host__ void roc_shmem_put_nbi(T *dest, const T *src,
                                size_t nelems, int pe);

template <typename T>
__host__ void roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest,
                                const T *source, size_t nelems, int pe);

template <typename T>
__host__ void roc_shmem_get_nbi(T *dest,
                                const T *source, size_t nelems, int pe);

template <typename T>
__host__ T roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest,
                                      T val, int pe);

template <typename T>
__host__ T roc_shmem_atomic_fetch_add(T *dest, T val, int pe);

template <typename T>
__host__ T roc_shmem_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest,
                                        T cond, T val, int pe);

template <typename T>
__host__ T roc_shmem_atomic_compare_swap(T *dest, T cond, T val, int pe);

template <typename T>
__host__ T roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__host__ T roc_shmem_atomic_fetch_inc(T *dest, int pe);

template <typename T>
__host__ T roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__host__ T roc_shmem_atomic_fetch(T *dest, int pe);

template <typename T>
__host__ void roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest,
                                   T val, int pe);

template <typename T>
__host__ void roc_shmem_atomic_add(T *dest, T val, int pe);

template <typename T>
__host__ void roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe);

template <typename T>
__host__ void roc_shmem_atomic_inc(T *dest, int pe);

template <typename T>
__host__ void
roc_shmem_broadcast(roc_shmem_ctx_t ctx,
                    T *dest,
                    const T *source,
                    int nelement,
                    int PE_root,
                    int PE_start,
                    int logPE_stride,
                    int PE_size,
                    long *pSync);

template<typename T, ROC_SHMEM_OP Op>
__host__ void
roc_shmem_to_all(roc_shmem_ctx_t ctx,
                 T *dest,
                 const T *source,
                 int nreduce,
                 int PE_start,
                 int logPE_stride,
                 int PE_size,
                 T *pWrk,
                 long *pSync);

template <typename T>
__host__ void roc_shmem_wait_until(T *ptr, roc_shmem_cmps cmp, T val);

template <typename T>
__device__ int roc_shmem_test(T *ptr, roc_shmem_cmps cmp, T val);

#endif
