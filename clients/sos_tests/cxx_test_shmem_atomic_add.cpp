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

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

enum op {
  ADD = 0,
  ATOMIC_ADD,
  CTX_ATOMIC_ADD,
  FADD,
  ATOMIC_FETCH_ADD,
  CTX_ATOMIC_FETCH_ADD,
  ATOMIC_FETCH_ADD_NBI,
  CTX_ATOMIC_FETCH_ADD_NBI
};

#ifdef ENABLE_DEPRECATED_TESTS
#define DEPRECATED_ADD(TYPENAME, ...) roc_shmem_##TYPENAME##_add(__VA_ARGS__)
#define DEPRECATED_FADD(TYPENAME, ...) roc_shmem_##TYPENAME##_fadd(__VA_ARGS__)
#else
#define DEPRECATED_ADD(TYPENAME, ...) \
  roc_shmem_##TYPENAME##_atomic_add(__VA_ARGS__)
#define DEPRECATED_FADD(TYPENAME, ...) \
  roc_shmem_##TYPENAME##_atomic_fetch_add(__VA_ARGS__)
#endif /* ENABLE_DEPRECATED_TESTS */

#define SHMEM_NBI_OPS_CASES(OP, TYPE, TYPENAME)                             \
  case ATOMIC_FETCH_ADD_NBI:                                                \
    roc_shmem_##TYPENAME##_atomic_fetch_add_nbi(&old, remote,               \
                                                (TYPE)(mype + 1), i);       \
    roc_shmem_quiet();                                                      \
    if (old > (TYPE)(npes * (npes + 1) / 2)) {                              \
      printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, \
             #TYPE);                                                        \
      rc = EXIT_FAILURE;                                                    \
    }                                                                       \
    break;                                                                  \
  case CTX_ATOMIC_FETCH_ADD_NBI:                                            \
    roc_shmem_ctx_##TYPENAME##_atomic_fetch_add_nbi(                        \
        ROC_SHMEM_CTX_DEFAULT, &old, remote, (TYPE)(mype + 1), i);          \
    roc_shmem_quiet();                                                      \
    if (old > (TYPE)(npes * (npes + 1) / 2)) {                              \
      printf("PE %i error inconsistent value of old (%s, %s)\n", mype, #OP, \
             #TYPE);                                                        \
      rc = EXIT_FAILURE;                                                    \
    }                                                                       \
    break;

#define TEST_SHMEM_ADD(OP, TYPE, TYPENAME)                                     \
  do {                                                                         \
    TYPE *remote;                                                              \
    TYPE old;                                                                  \
    const int mype = roc_shmem_my_pe();                                        \
    const int npes = roc_shmem_n_pes();                                        \
    remote = (TYPE *)roc_shmem_malloc(sizeof(TYPE));                           \
    *remote = (TYPE)0;                                                         \
    roc_shmem_barrier_all();                                                   \
    for (int i = 0; i < npes; i++) switch (OP) {                               \
        case ADD:                                                              \
          DEPRECATED_ADD(TYPENAME, remote, (TYPE)(mype + 1), i);               \
          break;                                                               \
        case ATOMIC_ADD:                                                       \
          roc_shmem_##TYPENAME##_atomic_add(remote, (TYPE)(mype + 1), i);      \
          break;                                                               \
        case CTX_ATOMIC_ADD:                                                   \
          roc_shmem_ctx_##TYPENAME##_atomic_add(ROC_SHMEM_CTX_DEFAULT, remote, \
                                                (TYPE)(mype + 1), i);          \
          break;                                                               \
        case FADD:                                                             \
          old = DEPRECATED_FADD(TYPENAME, remote, (TYPE)(mype + 1), i);        \
          if (old > (TYPE)(npes * (npes + 1) / 2)) {                           \
            printf("PE %i error inconsistent value of old (%s, %s)\n", mype,   \
                   #OP, #TYPE);                                                \
            rc = EXIT_FAILURE;                                                 \
          }                                                                    \
          break;                                                               \
        case ATOMIC_FETCH_ADD:                                                 \
          old = roc_shmem_##TYPENAME##_atomic_fetch_add(remote,                \
                                                        (TYPE)(mype + 1), i);  \
          if (old > (TYPE)(npes * (npes + 1) / 2)) {                           \
            printf("PE %i error inconsistent value of old (%s, %s)\n", mype,   \
                   #OP, #TYPE);                                                \
            rc = EXIT_FAILURE;                                                 \
          }                                                                    \
          break;                                                               \
        case CTX_ATOMIC_FETCH_ADD:                                             \
          old = roc_shmem_ctx_##TYPENAME##_atomic_fetch_add(                   \
              ROC_SHMEM_CTX_DEFAULT, remote, (TYPE)(mype + 1), i);             \
          if (old > (TYPE)(npes * (npes + 1) / 2)) {                           \
            printf("PE %i error inconsistent value of old (%s, %s)\n", mype,   \
                   #OP, #TYPE);                                                \
            rc = EXIT_FAILURE;                                                 \
          }                                                                    \
          break;                                                               \
        /*SHMEM_NBI_OPS_CASES(OP, TYPE, TYPENAME)*/                            \
        default:                                                               \
          printf("Invalid operation (%d)\n", OP);                              \
          roc_shmem_global_exit(1);                                            \
      }                                                                        \
    roc_shmem_barrier_all();                                                   \
    if ((*remote) != (TYPE)(npes * (npes + 1) / 2)) {                          \
      printf("PE %i observed error with TEST_SHMEM_ADD(%s, %s)\n", mype, #OP,  \
             #TYPE);                                                           \
      rc = EXIT_FAILURE;                                                       \
    }                                                                          \
    roc_shmem_free(remote);                                                    \
    if (rc == EXIT_FAILURE) roc_shmem_global_exit(1);                          \
  } while (false)

int main(int argc, char *argv[]) {
  roc_shmem_init();

  int rc = EXIT_SUCCESS;

#ifdef ENABLE_DEPRECATED_TESTS
  TEST_SHMEM_ADD(ADD, int, int);
  TEST_SHMEM_ADD(ADD, long, long);
  TEST_SHMEM_ADD(ADD, long long, longlong);
  TEST_SHMEM_ADD(ADD, unsigned int, uint);
  TEST_SHMEM_ADD(ADD, unsigned long, ulong);
  TEST_SHMEM_ADD(ADD, unsigned long long, ulonglong);
  TEST_SHMEM_ADD(ADD, int32_t, int32);
  TEST_SHMEM_ADD(ADD, int64_t, int64);
  TEST_SHMEM_ADD(ADD, uint32_t, uint32);
  TEST_SHMEM_ADD(ADD, uint64_t, uint64);
  TEST_SHMEM_ADD(ADD, size_t, size);
  TEST_SHMEM_ADD(ADD, ptrdiff_t, ptrdiff);

  TEST_SHMEM_ADD(FADD, int, int);
  TEST_SHMEM_ADD(FADD, long, long);
  TEST_SHMEM_ADD(FADD, long long, longlong);
  TEST_SHMEM_ADD(FADD, unsigned int, uint);
  TEST_SHMEM_ADD(FADD, unsigned long, ulong);
  TEST_SHMEM_ADD(FADD, unsigned long long, ulonglong);
  TEST_SHMEM_ADD(FADD, int32_t, int32);
  TEST_SHMEM_ADD(FADD, int64_t, int64);
  TEST_SHMEM_ADD(FADD, uint32_t, uint32);
  TEST_SHMEM_ADD(FADD, uint64_t, uint64);
  TEST_SHMEM_ADD(FADD, size_t, size);
  TEST_SHMEM_ADD(FADD, ptrdiff_t, ptrdiff);
#endif /* ENABLE_DEPRECATED_TESTS */

  // TEST_SHMEM_ADD(ATOMIC_ADD, int, int);
  // TEST_SHMEM_ADD(ATOMIC_ADD, long, long);
  // TEST_SHMEM_ADD(ATOMIC_ADD, long long, longlong);
  // TEST_SHMEM_ADD(ATOMIC_ADD, unsigned int, uint);
  // TEST_SHMEM_ADD(ATOMIC_ADD, unsigned long, ulong);
  // TEST_SHMEM_ADD(ATOMIC_ADD, unsigned long long, ulonglong);
  // TEST_SHMEM_ADD(ATOMIC_ADD, int32_t, int32);
  TEST_SHMEM_ADD(ATOMIC_ADD, int64_t, int64);
  // TEST_SHMEM_ADD(ATOMIC_ADD, uint32_t, uint32);
  TEST_SHMEM_ADD(ATOMIC_ADD, uint64_t, uint64);
  // TEST_SHMEM_ADD(ATOMIC_ADD, size_t, size);
  // TEST_SHMEM_ADD(ATOMIC_ADD, ptrdiff_t, ptrdiff);

  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, int, int);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, long, long);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, long long, longlong);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, unsigned int, uint);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, unsigned long, ulong);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, unsigned long long, ulonglong);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, int32_t, int32);
  TEST_SHMEM_ADD(CTX_ATOMIC_ADD, int64_t, int64);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, uint32_t, uint32);
  TEST_SHMEM_ADD(CTX_ATOMIC_ADD, uint64_t, uint64);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, size_t, size);
  // TEST_SHMEM_ADD(CTX_ATOMIC_ADD, ptrdiff_t, ptrdiff);

  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, int, int);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, long, long);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, long long, longlong);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, unsigned int, uint);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, unsigned long, ulong);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, unsigned long long, ulonglong);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, int32_t, int32);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, int64_t, int64);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, uint32_t, uint32);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, uint64_t, uint64);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, size_t, size);
  // TEST_SHMEM_ADD(ATOMIC_FETCH_ADD, ptrdiff_t, ptrdiff);

  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, int, int);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, long, long);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, long long, longlong);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, unsigned int, uint);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, unsigned long, ulong);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, unsigned long long, ulonglong);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, int32_t, int32);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, int64_t, int64);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, uint32_t, uint32);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, uint64_t, uint64);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, size_t, size);
  // TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD, ptrdiff_t, ptrdiff);

  /*
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, int, int);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, long, long);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, long long, longlong);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, unsigned int, uint);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, unsigned long, ulong);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, unsigned long long, ulonglong);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, int32_t, int32);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, int64_t, int64);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, uint32_t, uint32);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, uint64_t, uint64);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, size_t, size);
  TEST_SHMEM_ADD(ATOMIC_FETCH_ADD_NBI, ptrdiff_t, ptrdiff);

  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, int, int);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, long, long);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, long long, longlong);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, unsigned int, uint);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, unsigned long, ulong);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, unsigned long long, ulonglong);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, int32_t, int32);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, int64_t, int64);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, uint32_t, uint32);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, uint64_t, uint64);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, size_t, size);
  TEST_SHMEM_ADD(CTX_ATOMIC_FETCH_ADD_NBI, ptrdiff_t, ptrdiff);
  */

  roc_shmem_finalize();
  return rc;
}
