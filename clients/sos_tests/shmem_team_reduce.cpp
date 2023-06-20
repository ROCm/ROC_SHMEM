/*
 *  This test program is derived from a unit test created by Nick Park.
 *  The original unit test is a work of the U.S. Government and is not subject
 *  to copyright protection in the United States.  Foreign copyrights may
 *  apply.
 *
 *  Copyright (c) 2021 Intel Corporation. All rights reserved.
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

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
//#include <complex.h>
//#include <math.h>
#include <stdbool.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

#define MAX_NPES 32

#define STRINGIFY(x) #x

#define REDUCTION(OP, TYPE)                                            \
  do {                                                                 \
    roc_shmem_ctx_##TYPE##_##OP##_to_all(                              \
        ROC_SHMEM_CTX_DEFAULT, ROC_SHMEM_TEAM_WORLD, dest, src, npes); \
  } while (0)

#define INIT_SRC_BUFFER(TYPE)            \
  do {                                   \
    for (int i = 0; i < MAX_NPES; i++) { \
      src[i] = (TYPE)1ULL;               \
    }                                    \
  } while (0)

#define CHECK_DEST_BUFFER(OP, TYPE, CORRECT_VAL)   \
  do {                                             \
    for (int i = 0; i < npes; i++) {               \
      if (dest[i] != (TYPE)CORRECT_VAL) {          \
        printf(                                    \
            "PE %i received incorrect value with " \
            "TEST_SHMEM_REDUCE(%s, %s)\n",         \
            mype, #OP, #TYPE);                     \
        rc = EXIT_FAILURE;                         \
      }                                            \
    }                                              \
  } while (0)

#define TEST_SHMEM_REDUCE(OP, TYPENAME, TYPE)                      \
  do {                                                             \
    TYPE *src, *dest;                                              \
    src = dest = nullptr;                                          \
    src = (TYPE *)roc_shmem_malloc(sizeof(TYPE) * MAX_NPES);       \
    dest = (TYPE *)roc_shmem_malloc(sizeof(TYPE) * MAX_NPES);      \
                                                                   \
    INIT_SRC_BUFFER(TYPE);                                         \
                                                                   \
    REDUCTION(OP, TYPENAME);                                       \
                                                                   \
    roc_shmem_barrier_all();                                       \
                                                                   \
    std::string op = STRINGIFY(OP);                                \
    if (op.compare("and") == 0) {                                  \
      CHECK_DEST_BUFFER(OP, TYPE, 1ULL);                           \
    } else if (op.compare("or") == 0) {                            \
      CHECK_DEST_BUFFER(OP, TYPE, 1ULL);                           \
    } else if (op.compare("xor") == 0) {                           \
      CHECK_DEST_BUFFER(OP, TYPE, (TYPE)(npes % 2 ? 1ULL : 0ULL)); \
    } else if (op.compare("max") == 0) {                           \
      CHECK_DEST_BUFFER(OP, TYPE, 1ULL);                           \
    } else if (op.compare("min") == 0) {                           \
      CHECK_DEST_BUFFER(OP, TYPE, 1ULL);                           \
    } else if (op.compare("sum") == 0) {                           \
      CHECK_DEST_BUFFER(OP, TYPE, npes);                           \
    } else if (op.compare("prod") == 0) {                          \
      CHECK_DEST_BUFFER(OP, TYPE, 1ULL);                           \
    } else {                                                       \
      printf("Invalid operation (%s)\n", STRINGIFY(OP));           \
      roc_shmem_global_exit(1);                                    \
    }                                                              \
                                                                   \
    roc_shmem_free(src);                                           \
    roc_shmem_free(dest);                                          \
                                                                   \
  } while (0)

int main(void) {
  roc_shmem_init();

  int rc = EXIT_SUCCESS;

  const int mype = roc_shmem_my_pe();
  const int npes = roc_shmem_n_pes();

  if (npes > MAX_NPES) {
    if (mype == 0)
      fprintf(stderr, "ERR - Requires less than %d PEs\n", MAX_NPES);
    roc_shmem_global_exit(1);
  }

  // TEST_SHMEM_REDUCE(and, uchar, unsigned char);
  // TEST_SHMEM_REDUCE(and, ushort, unsigned short);
  TEST_SHMEM_REDUCE(and, short, short);
  // TEST_SHMEM_REDUCE(and, uint, unsigned int);
  // TEST_SHMEM_REDUCE(and, ulong, unsigned long);
  TEST_SHMEM_REDUCE(and, long, long);
  // TEST_SHMEM_REDUCE(and, ulonglong, unsigned long long);
  TEST_SHMEM_REDUCE(and, longlong, long long);
  TEST_SHMEM_REDUCE(and, int, int);
  // TEST_SHMEM_REDUCE(and, int8, int8_t);
  // TEST_SHMEM_REDUCE(and, int16, int16_t);
  // TEST_SHMEM_REDUCE(and, int32, int32_t);
  // TEST_SHMEM_REDUCE(and, int64, int64_t);
  // TEST_SHMEM_REDUCE(and, uint8, uint8_t);
  // TEST_SHMEM_REDUCE(and, uint16, uint16_t);
  // TEST_SHMEM_REDUCE(and, uint32, uint32_t);
  // TEST_SHMEM_REDUCE(and, uint64, uint64_t);
  // TEST_SHMEM_REDUCE(and, size, size_t);

  // TEST_SHMEM_REDUCE(or, uchar, unsigned char);
  // TEST_SHMEM_REDUCE(or, ushort, unsigned short);
  TEST_SHMEM_REDUCE(or, short, short);
  // TEST_SHMEM_REDUCE(or, uint, unsigned int);
  TEST_SHMEM_REDUCE(or, int, int);
  // TEST_SHMEM_REDUCE(or, ulong, unsigned long);
  TEST_SHMEM_REDUCE(or, long, long);
  // TEST_SHMEM_REDUCE(or, ulonglong, unsigned long long);
  TEST_SHMEM_REDUCE(or, longlong, long long);
  // TEST_SHMEM_REDUCE(or, int8, int8_t);
  // TEST_SHMEM_REDUCE(or, int16, int16_t);
  // TEST_SHMEM_REDUCE(or, int32, int32_t);
  // TEST_SHMEM_REDUCE(or, int64, int64_t);
  // TEST_SHMEM_REDUCE(or, uint8, uint8_t);
  // TEST_SHMEM_REDUCE(or, uint16, uint16_t);
  // TEST_SHMEM_REDUCE(or, uint32, uint32_t);
  // TEST_SHMEM_REDUCE(or, uint64, uint64_t);
  // TEST_SHMEM_REDUCE(or, size, size_t);

  // TEST_SHMEM_REDUCE(xor, uchar, unsigned char);
  // TEST_SHMEM_REDUCE(xor, ushort, unsigned short);
  TEST_SHMEM_REDUCE(xor, short, short);
  // TEST_SHMEM_REDUCE(xor, uint, unsigned int);
  TEST_SHMEM_REDUCE(xor, int, int);
  // TEST_SHMEM_REDUCE(xor, ulong, unsigned long);
  TEST_SHMEM_REDUCE(xor, long, long);
  // TEST_SHMEM_REDUCE(xor, ulonglong, unsigned long long);
  // TEST_SHMEM_REDUCE(xor, int8, int8_t);
  TEST_SHMEM_REDUCE(xor, longlong, long long);
  // TEST_SHMEM_REDUCE(xor, int16, int16_t);
  // TEST_SHMEM_REDUCE(xor, int32, int32_t);
  // TEST_SHMEM_REDUCE(xor, int64, int64_t);
  // TEST_SHMEM_REDUCE(xor, uint8, uint8_t);
  // TEST_SHMEM_REDUCE(xor, uint16, uint16_t);
  // TEST_SHMEM_REDUCE(xor, uint32, uint32_t);
  // TEST_SHMEM_REDUCE(xor, uint64, uint64_t);
  // TEST_SHMEM_REDUCE(xor, size, size_t);

  // TEST_SHMEM_REDUCE(max, char, char);
  // TEST_SHMEM_REDUCE(max, schar, signed char);
  TEST_SHMEM_REDUCE(max, short, short);
  TEST_SHMEM_REDUCE(max, int, int);
  TEST_SHMEM_REDUCE(max, long, long);
  TEST_SHMEM_REDUCE(max, longlong, long long);
  // TEST_SHMEM_REDUCE(max, ptrdiff, ptrdiff_t);
  // TEST_SHMEM_REDUCE(max, uchar, unsigned char);
  // TEST_SHMEM_REDUCE(max, ushort, unsigned short);
  // TEST_SHMEM_REDUCE(max, uint, unsigned int);
  // TEST_SHMEM_REDUCE(max, ulong, unsigned long);
  // TEST_SHMEM_REDUCE(max, ulonglong, unsigned long long);
  // TEST_SHMEM_REDUCE(max, int8, int8_t);
  // TEST_SHMEM_REDUCE(max, int16, int16_t);
  // TEST_SHMEM_REDUCE(max, int32, int32_t);
  // TEST_SHMEM_REDUCE(max, int64, int64_t);
  // TEST_SHMEM_REDUCE(max, uint8, uint8_t);
  // TEST_SHMEM_REDUCE(max, uint16, uint16_t);
  // TEST_SHMEM_REDUCE(max, uint32, uint32_t);
  // TEST_SHMEM_REDUCE(max, uint64, uint64_t);
  // TEST_SHMEM_REDUCE(max, size, size_t);
  TEST_SHMEM_REDUCE(max, float, float);
  TEST_SHMEM_REDUCE(max, double, double);
  // TEST_SHMEM_REDUCE(max, longdouble, long double);

  // TEST_SHMEM_REDUCE(min, char, char);
  // TEST_SHMEM_REDUCE(min, schar, signed char);
  TEST_SHMEM_REDUCE(min, short, short);
  TEST_SHMEM_REDUCE(min, int, int);
  TEST_SHMEM_REDUCE(min, long, long);
  TEST_SHMEM_REDUCE(min, longlong, long long);
  // TEST_SHMEM_REDUCE(min, ptrdiff, ptrdiff_t);
  // TEST_SHMEM_REDUCE(min, uchar, unsigned char);
  // TEST_SHMEM_REDUCE(min, ushort, unsigned short);
  // TEST_SHMEM_REDUCE(min, uint, unsigned int);
  // TEST_SHMEM_REDUCE(min, ulong, unsigned long);
  // TEST_SHMEM_REDUCE(min, ulonglong, unsigned long long);
  // TEST_SHMEM_REDUCE(min, int8, int8_t);
  // TEST_SHMEM_REDUCE(min, int16, int16_t);
  // TEST_SHMEM_REDUCE(min, int32, int32_t);
  // TEST_SHMEM_REDUCE(min, int64, int64_t);
  // TEST_SHMEM_REDUCE(min, uint8, uint8_t);
  // TEST_SHMEM_REDUCE(min, uint16, uint16_t);
  // TEST_SHMEM_REDUCE(min, uint32, uint32_t);
  // TEST_SHMEM_REDUCE(min, uint64, uint64_t);
  // TEST_SHMEM_REDUCE(min, size, size_t);
  TEST_SHMEM_REDUCE(min, float, float);
  TEST_SHMEM_REDUCE(min, double, double);
  // TEST_SHMEM_REDUCE(min, longdouble, long double);

  // TEST_SHMEM_REDUCE(sum, char, char);
  // TEST_SHMEM_REDUCE(sum, schar, signed char);
  TEST_SHMEM_REDUCE(sum, short, short);
  TEST_SHMEM_REDUCE(sum, int, int);
  TEST_SHMEM_REDUCE(sum, long, long);
  TEST_SHMEM_REDUCE(sum, longlong, long long);
  // TEST_SHMEM_REDUCE(sum, ptrdiff, ptrdiff_t);
  // TEST_SHMEM_REDUCE(sum, uchar, unsigned char);
  // TEST_SHMEM_REDUCE(sum, ushort, unsigned short);
  // TEST_SHMEM_REDUCE(sum, uint, unsigned int);
  // TEST_SHMEM_REDUCE(sum, ulong, unsigned long);
  // TEST_SHMEM_REDUCE(sum, ulonglong, unsigned long long);
  // TEST_SHMEM_REDUCE(sum, int8, int8_t);
  // TEST_SHMEM_REDUCE(sum, int16, int16_t);
  // TEST_SHMEM_REDUCE(sum, int32, int32_t);
  // TEST_SHMEM_REDUCE(sum, int64, int64_t);
  // TEST_SHMEM_REDUCE(sum, uint8, uint8_t);
  // TEST_SHMEM_REDUCE(sum, uint16, uint16_t);
  // TEST_SHMEM_REDUCE(sum, uint32, uint32_t);
  // TEST_SHMEM_REDUCE(sum, uint64, uint64_t);
  // TEST_SHMEM_REDUCE(sum, size, size_t);
  TEST_SHMEM_REDUCE(sum, float, float);
  TEST_SHMEM_REDUCE(sum, double, double);
  // TEST_SHMEM_REDUCE(sum, longdouble, long double);
  // TEST_SHMEM_REDUCE(sum, complexd, double _Complex);
  // TEST_SHMEM_REDUCE(sum, complexf, float _Complex);

  // TEST_SHMEM_REDUCE(prod, char, char);
  // TEST_SHMEM_REDUCE(prod, schar, signed char);
  TEST_SHMEM_REDUCE(prod, short, short);
  TEST_SHMEM_REDUCE(prod, int, int);
  TEST_SHMEM_REDUCE(prod, long, long);
  TEST_SHMEM_REDUCE(prod, longlong, long long);
  // TEST_SHMEM_REDUCE(prod, ptrdiff, ptrdiff_t);
  // TEST_SHMEM_REDUCE(prod, uchar, unsigned char);
  // TEST_SHMEM_REDUCE(prod, ushort, unsigned short);
  // TEST_SHMEM_REDUCE(prod, uint, unsigned int);
  // TEST_SHMEM_REDUCE(prod, ulong, unsigned long);
  // TEST_SHMEM_REDUCE(prod, ulonglong, unsigned long long);
  // TEST_SHMEM_REDUCE(prod, int8, int8_t);
  // TEST_SHMEM_REDUCE(prod, int16, int16_t);
  // TEST_SHMEM_REDUCE(prod, int32, int32_t);
  // TEST_SHMEM_REDUCE(prod, int64, int64_t);
  // TEST_SHMEM_REDUCE(prod, uint8, uint8_t);
  // TEST_SHMEM_REDUCE(prod, uint16, uint16_t);
  // TEST_SHMEM_REDUCE(prod, uint32, uint32_t);
  // TEST_SHMEM_REDUCE(prod, uint64, uint64_t);
  // TEST_SHMEM_REDUCE(prod, size, size_t);
  TEST_SHMEM_REDUCE(prod, float, float);
  TEST_SHMEM_REDUCE(prod, double, double);
  // TEST_SHMEM_REDUCE(prod, longdouble, long double);
  // TEST_SHMEM_REDUCE(prod, complexd, double _Complex);
  // TEST_SHMEM_REDUCE(prod, complexf, float _Complex);

  roc_shmem_finalize();
  return rc;
}
