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

#define TEST_SHMEM_WAIT_UNTIL(TYPE, TYPENAME)                            \
  do {                                                                   \
    TYPE *remote = 0;                                                    \
    remote = (TYPE *)roc_shmem_malloc(sizeof(TYPE));                     \
    *remote = 0;                                                         \
    const int mype = roc_shmem_my_pe();                                  \
    const int npes = roc_shmem_n_pes();                                  \
    roc_shmem_##TYPENAME##_p(remote, (TYPE)mype + 1, (mype + 1) % npes); \
    roc_shmem_##TYPENAME##_wait_until(remote, ROC_SHMEM_CMP_NE, 0);      \
    if ((*remote) != (TYPE)((mype + npes - 1) % npes) + 1) {             \
      printf(                                                            \
          "PE %i received incorrect value with "                         \
          "TEST_SHMEM_WAIT_UNTIL(%s)\n",                                 \
          mype, #TYPE);                                                  \
      rc = EXIT_FAILURE;                                                 \
      roc_shmem_global_exit(1);                                          \
    }                                                                    \
    roc_shmem_free(remote);                                              \
  } while (false)

int main(int argc, char *argv[]) {
  roc_shmem_init();

  int rc = EXIT_SUCCESS;
  TEST_SHMEM_WAIT_UNTIL(short, short);
  TEST_SHMEM_WAIT_UNTIL(int, int);
  TEST_SHMEM_WAIT_UNTIL(long, long);
  TEST_SHMEM_WAIT_UNTIL(long long, longlong);
  TEST_SHMEM_WAIT_UNTIL(unsigned short, ushort);
  TEST_SHMEM_WAIT_UNTIL(unsigned int, uint);
  TEST_SHMEM_WAIT_UNTIL(unsigned long, ulong);
  TEST_SHMEM_WAIT_UNTIL(unsigned long long, ulonglong);
  // TEST_SHMEM_WAIT_UNTIL(int32_t, int32);
  // TEST_SHMEM_WAIT_UNTIL(int64_t, int64);
  // TEST_SHMEM_WAIT_UNTIL(uint32_t, uint32);
  // TEST_SHMEM_WAIT_UNTIL(uint64_t, uint64);
  // TEST_SHMEM_WAIT_UNTIL(size_t, size);
  // TEST_SHMEM_WAIT_UNTIL(ptrdiff_t, ptrdiff);

  roc_shmem_finalize();
  return rc;
}
