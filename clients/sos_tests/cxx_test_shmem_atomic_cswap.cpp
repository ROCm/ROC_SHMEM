/*
 *  This test program is derived from a unit test created by Nick Park.
 *  The original unit test is a work of the U.S. Government and is not subject
 *  to copyright protection in the United States.  Foreign copyrights may
 *  apply.
 *
 *  Copyright (c) 2017 Intel Corporation. All rights reserved.
 *  This software is available to you under the BSD license below:
 *
 *      Redistribution and use in source and binary forms, with or
 *      without modification, are permitted provided that the following
 *      conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <roc_shmem.hpp>

using namespace rocshmem;

enum op { CSWAP = 0, ATOMIC_COMPARE_SWAP, CTX_ATOMIC_COMPARE_SWAP,
          ATOMIC_COMPARE_SWAP_NBI, CTX_ATOMIC_COMPARE_SWAP_NBI };

#ifdef ENABLE_DEPRECATED_TESTS
#define DEPRECATED_CSWAP(TYPENAME,...) roc_shmem_##TYPENAME##_cswap(__VA_ARGS__)
#else
#define DEPRECATED_CSWAP(TYPENAME,...) roc_shmem_##TYPENAME##_atomic_compare_swap(__VA_ARGS__)
#endif /* ENABLE_DEPRECATED_TESTS */

#define SHMEM_NBI_OPS_CASES(OP, TYPE, TYPENAME)                                         \
        case ATOMIC_COMPARE_SWAP_NBI:                                                   \
            roc_shmem_##TYPENAME##_atomic_compare_swap_nbi(&old, remote,                \
                            (TYPE)npes, (TYPE)mype, (mype + 1) % npes);                 \
            break;                                                                      \
        case CTX_ATOMIC_COMPARE_SWAP_NBI:                                               \
            roc_shmem_ctx_##TYPENAME##_atomic_compare_swap_nbi(ROC_SHMEM_CTX_DEFAULT,   \
              &old, remote, (TYPE)npes, (TYPE)mype, (mype + 1) % npes);                 \
            break;

#define TEST_SHMEM_CSWAP(OP, TYPE, TYPENAME)                                    \
  do {                                                                          \
    TYPE *remote;                                                               \
    TYPE old;                                                                   \
    const int mype = roc_shmem_my_pe();                                         \
    const int npes = roc_shmem_n_pes();                                         \
    remote = (TYPE *) roc_shmem_malloc(sizeof(TYPE));                           \
    *remote = npes;                                                             \
    roc_shmem_barrier_all();                                                    \
    switch (OP) {                                                               \
        case CSWAP:                                                             \
            old = DEPRECATED_CSWAP(TYPENAME, remote, (TYPE)npes,                \
                                (TYPE)mype, (mype + 1) % npes);                 \
            break;                                                              \
        case ATOMIC_COMPARE_SWAP:                                               \
            old = roc_shmem_##TYPENAME##_atomic_compare_swap(remote,            \
                            (TYPE)npes, (TYPE)mype, (mype + 1) % npes);         \
            break;                                                              \
        case CTX_ATOMIC_COMPARE_SWAP:                                           \
            old = roc_shmem_ctx_##TYPENAME##_atomic_compare_swap(               \
                                            ROC_SHMEM_CTX_DEFAULT, remote,      \
                                            (TYPE)npes, (TYPE)mype,             \
                                            (mype + 1) % npes);                 \
            break;                                                              \
        /* SHMEM_NBI_OPS_CASES(OP, TYPE, TYPENAME)  */                          \
        default:                                                                \
          printf("invalid operation (%d)\n", OP);                               \
          roc_shmem_global_exit(1);                                             \
    }                                                                           \
    roc_shmem_barrier_all();                                                    \
    if ((*remote) != (TYPE)((mype + npes - 1) % npes)) {                        \
      printf("PE %i observed error with TEST_SHMEM_CSWAP(%s, %s)\n",            \
             mype, #OP, #TYPE);                                                 \
      rc = EXIT_FAILURE;                                                        \
    }                                                                           \
    if (old != (TYPE) npes) {                                                   \
      printf("PE %i error inconsistent value of old (%s, %s)\n",                \
             mype, #OP, #TYPE);                                                 \
      rc = EXIT_FAILURE;                                                        \
    }                                                                           \
    roc_shmem_free(remote);                                                     \
    if (rc == EXIT_FAILURE) roc_shmem_global_exit(1);                           \
  } while (false)


int main(int argc, char* argv[]) {
  roc_shmem_init(1);

  int rc = EXIT_SUCCESS;

#ifdef ENABLE_DEPRECATED_TESTS
  TEST_SHMEM_CSWAP(CSWAP, int, int);
  TEST_SHMEM_CSWAP(CSWAP, long, long);
  TEST_SHMEM_CSWAP(CSWAP, long long, longlong);
  TEST_SHMEM_CSWAP(CSWAP, unsigned int, uint);
  TEST_SHMEM_CSWAP(CSWAP, unsigned long, ulong);
  TEST_SHMEM_CSWAP(CSWAP, unsigned long long, ulonglong);
  TEST_SHMEM_CSWAP(CSWAP, int32_t, int32);
  TEST_SHMEM_CSWAP(CSWAP, int64_t, int64);
  TEST_SHMEM_CSWAP(CSWAP, uint32_t, uint32);
  TEST_SHMEM_CSWAP(CSWAP, uint64_t, uint64);
  TEST_SHMEM_CSWAP(CSWAP, size_t, size);
  TEST_SHMEM_CSWAP(CSWAP, ptrdiff_t, ptrdiff);
#endif /* ENABLE_DEPRECATED_TESTS */

  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, int, int);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, long, long);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, long long, longlong);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, unsigned int, uint);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, unsigned long, ulong);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, unsigned long long, ulonglong);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, int32_t, int32);
  TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, int64_t, int64);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, uint32_t, uint32);
  TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, uint64_t, uint64);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, size_t, size);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP, ptrdiff_t, ptrdiff);

  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, int, int);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, long, long);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, long long, longlong);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, unsigned int, uint);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, unsigned long, ulong);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, unsigned long long, ulonglong);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, int32_t, int32);
  TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, int64_t, int64);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, uint32_t, uint32);
  TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, uint64_t, uint64);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, size_t, size);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP, ptrdiff_t, ptrdiff);

  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, int, int);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, long, long);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, long long, longlong);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, unsigned int, uint);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, unsigned long, ulong);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, unsigned long long, ulonglong);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, int32_t, int32);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, int64_t, int64);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, uint32_t, uint32);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, uint64_t, uint64);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, size_t, size);
  //TEST_SHMEM_CSWAP(ATOMIC_COMPARE_SWAP_NBI, ptrdiff_t, ptrdiff);

  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, int, int);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, long, long);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, long long, longlong);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, unsigned int, uint);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, unsigned long, ulong);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, unsigned long long, ulonglong);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, int32_t, int32);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, int64_t, int64);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, uint32_t, uint32);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, uint64_t, uint64);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, size_t, size);
  //TEST_SHMEM_CSWAP(CTX_ATOMIC_COMPARE_SWAP_NBI, ptrdiff_t, ptrdiff);

  roc_shmem_finalize();
  return rc;
}
