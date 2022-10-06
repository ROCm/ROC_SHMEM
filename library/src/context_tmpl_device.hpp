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

#ifndef ROCSHMEM_LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP
#define ROCSHMEM_LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP

#include "backend_type.hpp"
#include "context_ib_device.hpp"
#include "context_ro_device.hpp"

namespace rocshmem {

/*
 * Context dispatch implementations for the template functions. Needs to
 * be in a header and not cpp because it is a template.
 */
template <typename T>
__device__ void
Context::p(T *dest,
           T value,
           int pe) {
    ctxStats.incStat(NUM_P);

    /*
     * TODO: Need to handle _p a bit differently for coalescing, since the
     * owner of a coalesced message needs val from all absorbed messages.
     */
     DISPATCH(p(dest, value, pe));
}

template <typename T>
__device__ T
Context::g(T *source,
           int pe) {
    ctxStats.incStat(NUM_G);

    /*
     * TODO: Need to handle _g a bit differently for coalescing, since the
     * owner of a coalesced message needs val from all absorbed messages.
     */
    DISPATCH_RET(g(source, pe));
}

// The only way to get multi-arg templates to feed into a macro
template <typename T, ROC_SHMEM_OP Op>
__device__ void
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

    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_TO_ALL);
    }

    DISPATCH(to_all<PAIR(T, Op)>(dest,
                                 source,
                                 nreduce,
                                 PE_start,
                                 logPE_stride,
                                 PE_size,
                                 pWrk,
                                 pSync));
}

template <typename T, ROC_SHMEM_OP Op>
__device__ void
Context::to_all(roc_shmem_team_t team,
                T *dest,
                const T *source,
                int nreduce) {
    if (nreduce == 0) {
        return;
    }

    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_TO_ALL);
    }

    DISPATCH(to_all<PAIR(T, Op)>(team,
                                 dest,
                                 source,
                                 nreduce));
}

template <typename T>
__device__ void
Context::put(T *dest,
             const T *source,
             size_t nelems,
             int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT);

    DISPATCH(put(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::put_nbi(T *dest,
                 const T *source,
                 size_t nelems,
                 int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_NBI);

    DISPATCH(put_nbi(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::get(T *dest,
             const T *source,
             size_t nelems,
             int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET);

    DISPATCH(get(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::get_nbi(T *dest,
                 const T *source,
                 size_t nelems,
                 int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_NBI);

    DISPATCH(get_nbi(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::alltoall(roc_shmem_team_t team,
                  T *dest,
                  const T *source,
                  int nelems) {
    if (nelems == 0) {
        return;
    }

    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_ALLTOALL);
    }

    DISPATCH(alltoall<T>(team,
                         dest,
                         source,
                         nelems));
}

template <typename T>
__device__ void
Context::fcollect(roc_shmem_team_t team,
                  T *dest,
                  const T *source,
                  int nelems) {
    if (nelems == 0) {
        return;
    }

    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_FCOLLECT);
    }

    DISPATCH(fcollect<T>(team,
                         dest,
                         source,
                         nelems));
}

template <typename T>
__device__ void
Context::broadcast(roc_shmem_team_t team,
                   T *dest,
                   const T *source,
                   int nelems,
                   int pe_root) {
    if (nelems == 0) {
        return;
    }

    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_BROADCAST);
    }

    DISPATCH(broadcast<T>(team,
                          dest,
                          source,
                          nelems,
                          pe_root));
}

template <typename T>
__device__ void
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

    if (is_thread_zero_in_block()) {
        ctxStats.incStat(NUM_BROADCAST);
    }

    DISPATCH(broadcast<T>(dest,
                          source,
                          nelems,
                          pe_root,
                          pe_start,
                          log_pe_stride,
                          pe_size,
                          p_sync));
}

template <typename T>
__device__ void
Context::wait_until(T *ptr,
                    roc_shmem_cmps cmp,
                    T val) {
    while (!test(ptr, cmp, val)) {
    }
}

template <typename T>
__device__ int
Context::test(T *ptr,
              roc_shmem_cmps cmp,
              T val) {
    int ret = 0;
    volatile T * vol_ptr = reinterpret_cast<T*>(ptr);
    switch (cmp) {
        case ROC_SHMEM_CMP_EQ:
            if (uncached_load(vol_ptr) == val) {
                ret = 1;
            }
            break;
        case ROC_SHMEM_CMP_NE:
            if (uncached_load(vol_ptr) != val) {
                ret = 1;
            }
            break;
        case ROC_SHMEM_CMP_GT:
            if (uncached_load(vol_ptr) > val) {
                ret = 1;
            }
            break;
        case ROC_SHMEM_CMP_GE:
            if (uncached_load(vol_ptr) >= val) {
                ret = 1;
            }
            break;
        case ROC_SHMEM_CMP_LT:
            if (uncached_load(vol_ptr) < val) {
                ret = 1;
            }
            break;
        case ROC_SHMEM_CMP_LE:
            if (uncached_load(vol_ptr) <= val) {
                ret = 1;
            }
            break;
        default:
            break;
    }
    return ret;
}

template <typename T>
__device__ void
Context::put_wg(T *dest,
                const T *source,
                size_t nelems,
                int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_WG);

    DISPATCH_NO_LOCK(put_wg(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::put_nbi_wg(T *dest,
                    const T *source,
                    size_t nelems,
                    int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_NBI_WG);

    DISPATCH_NO_LOCK(put_nbi_wg(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::get_wg(T *dest,
                const T *source,
                size_t nelems,
                int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_WG);

    DISPATCH_NO_LOCK(get_wg(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::get_nbi_wg(T *dest,
                    const T *source,
                    size_t nelems,
                    int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_NBI_WG);

    DISPATCH_NO_LOCK(get_nbi_wg(dest, source, nelems, pe));
}


template <typename T>
__device__ void
Context::put_wave(T *dest,
                  const T *source,
                  size_t nelems,
                  int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_WAVE);

    DISPATCH(put_wave(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::put_nbi_wave(T *dest,
                      const T *source,
                      size_t nelems,
                      int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_PUT_NBI_WAVE);

    DISPATCH(put_nbi_wave(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::get_wave(T *dest,
                  const T *source,
                  size_t nelems,
                  int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_WAVE);

    DISPATCH(get_wave(dest, source, nelems, pe));
}

template <typename T>
__device__ void
Context::get_nbi_wave(T *dest,
                      const T *source,
                      size_t nelems,
                      int pe) {
    if (nelems == 0) {
        return;
    }

    ctxStats.incStat(NUM_GET_NBI_WAVE);

    DISPATCH(get_nbi_wave(dest, source, nelems, pe));
}

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_CONTEXT_TMPL_DEVICE_HPP
