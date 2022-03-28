/*
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
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

/*
* to_all - exercise SHMEM max,min,or,prod,sum,or,xor_to_all() reduction calls.
*       Each reduction is invoked for all data types:
*           short, int, long, float, double, long double, long long.
*       Point being numerous SHMEM atomics and synchronizations in flight.
*       From OpenSHMEM_specification_v1.0-final doc:
*           The pWrk and pSync arrays on all PEs in the active set must not be
*           in use from a prior call to a collective OpenSHMEM routine.
*
* frank @ SystemFabric Works identified an interesting overflow issue in the
* prod_to_all test. In the presence of slightly larger PE counts (>=14),
* overflow is encountered in short, int and float, double and long double.
* The short and int both wrap correctly and are both uniformly wrong...uniformly
* being the salient point. float, double and long double all suffer from
* floating point rounding errors, hence the FP test results are ignored
* (assumed to pass)when FP rounding is encountered. FP*_prod_to_all() calls are
* still made so as not to upset the pSync ordering.
*
* usage: to_all {-amopsSv|h}
*   where:
*       -a  do not run and_to_all
*       -m  do not run min_to_all, max_to_all() always run.
*       -o  do not run or_to_all
*       -p  do not run prod_to_all
*       -s  do not run sum_to_all
*       -x  do not run xor_to_all
*       -S  Serialize *_to_all() calls with barriers.
*       -v  verbose(additional -v, more verbose)
*       -h  this text.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <complex.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

#define Rprintf if (roc_shmem_my_pe() == 0) printf
#define Rfprintf if (roc_shmem_my_pe() == 0) fprintf
#define Vprintf if (Verbose > 1) printf

int sum_to_all(int me, int npes);
int and_to_all(int me, int npes);
int min_to_all(int me, int npes);
int max_to_all(int me, int npes);
int prod_to_all(int me, int npes);
int or_to_all(int me, int npes);
int xor_to_all(int me, int npes);

int Verbose;
int Serialize;
int Min, And, Sum, Prod, Or, Xor;
int Passed;

long *pSync;
long *pSync1;

#define N 128

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define WRK_SIZE MAX(N/2+1, ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE)

short *src0, *dst0, *pWrk0;
int *src1, *dst1, *pWrk1;
long *src2, *dst2, *pWrk2;
float *src3, *dst3, *pWrk3;
double *src4, *dst4, *pWrk4;
long double *src5, *dst5, *pWrk5;
long long *src6, *dst6, *pWrk6;

short expected_result0;
int expected_result1;
long expected_result2;
float expected_result3;
double expected_result4;
long double expected_result5;
long long expected_result6;

int ok[7];

int
max_to_all(int me, int npes)
{
    int i, j, pass=0;

    memset(ok,0,sizeof(ok));

    for (i = 0; i < N; i++) {
        src0[i] = src1[i] = src2[i] = src3[i] = src4[i] = src5[i] = src6[i]
            = me + i;
    }
    roc_shmem_barrier_all();

    roc_shmem_ctx_short_max_to_all(ROC_SHMEM_CTX_DEFAULT,      dst0, src0, N, 0, 0, npes, pWrk0, pSync);
    roc_shmem_ctx_int_max_to_all(ROC_SHMEM_CTX_DEFAULT,        dst1, src1, N, 0, 0, npes, pWrk1, pSync1);
    roc_shmem_ctx_long_max_to_all(ROC_SHMEM_CTX_DEFAULT,       dst2, src2, N, 0, 0, npes, pWrk2, pSync);
    roc_shmem_ctx_float_max_to_all(ROC_SHMEM_CTX_DEFAULT,      dst3, src3, N, 0, 0, npes, pWrk3, pSync1);
    roc_shmem_ctx_double_max_to_all(ROC_SHMEM_CTX_DEFAULT,     dst4, src4, N, 0, 0, npes, pWrk4, pSync);
    //roc_shmem_ctx_longdouble_max_to_all(ROC_SHMEM_CTX_DEFAULT, dst5, src5, N, 0, 0, npes, pWrk5, pSync1);
    roc_shmem_ctx_longlong_max_to_all(ROC_SHMEM_CTX_DEFAULT,   dst6, src6, N, 0, 0, npes, pWrk6, pSync);

    if (me == 0) {
        for (i = 0,j=-1; i < N; i++,j++) {
          if(dst0[i] != npes+j) ok[0] = 1;
          if(dst1[i] != npes+j) ok[1] = 1;
          if(dst2[i] != npes+j) ok[2] = 1;
          if(dst3[i] != npes+j) ok[3] = 1;
          if(dst4[i] != npes+j) ok[4] = 1;
          if(dst5[i] != npes+j) ok[5] = 1;
          if(dst6[i] != npes+j) ok[6] = 1;
        }

        if(ok[0]==1){
            printf("Reduction operation roc_shmem_short_max_to_all: Failed\n");
        }
        else{
            Vprintf("Reduction operation roc_shmem_short_max_to_all: Passed\n");
            pass++;
        }
        if(ok[1]==1){
            printf("Reduction operation roc_shmem_int_max_to_all: Failed\n");
        }
        else{
            Vprintf("Reduction operation roc_shmem_int_max_to_all: Passed\n");
            pass++;
        }
        if(ok[2]==1){
            printf("Reduction operation roc_shmem_long_max_to_all: Failed\n");
        }
        else{
            Vprintf("Reduction operation roc_shmem_long_max_to_all: Passed\n");
            pass++;
        }
        if(ok[3]==1){
            printf("Reduction operation roc_shmem_float_max_to_all: Failed\n");
        }
        else{
            Vprintf("Reduction operation roc_shmem_float_max_to_all: Passed\n");
            pass++;
        }
        if(ok[4]==1){
            printf("Reduction operation roc_shmem_double_max_to_all: Failed\n");
        }
        else{
            Vprintf("Reduction operation roc_shmem_double_max_to_all: Passed\n");
            pass++;
        }
        /*
        if(ok[5]==1){
          printf("Reduction operation roc_shmem_longdouble_max_to_all: Failed\n");
        }
        else{
           Vprintf("Reduction operation roc_shmem_longdouble_max_to_all: Passed\n");
           pass++;
        }
        */
        pass++;
        if(ok[6]==1){
            printf("Reduction operation roc_shmem_longlong_max_to_all: Failed\n");
        }
        else{
            Vprintf("Reduction operation roc_shmem_longlong_max_to_all: Passed\n");
            pass++;
        }
        Vprintf("\n");
    }
    if (Serialize) roc_shmem_barrier_all();

    return (pass == 7 ? 1 : 0);
}

int
min_to_all(int me, int npes)
{
    int i, pass=0;

    memset(ok,0,sizeof(ok));

    for (i = 0; i < N; i++) {
      src0[i] = src1[i] = src2[i] = src3[i] = src4[i] = src5[i] = src6[i]
        = me + i;
      dst0[i] = -9;
      dst1[i] = -9;
      dst2[i] = -9;
      dst3[i] = -9;
      dst4[i] = -9;
      dst5[i] = -9;
      dst6[i] = -9;
    }

    roc_shmem_barrier_all();

    roc_shmem_ctx_short_min_to_all(ROC_SHMEM_CTX_DEFAULT, dst0, src0, N, 0, 0, npes, pWrk0, pSync);
    roc_shmem_ctx_int_min_to_all(ROC_SHMEM_CTX_DEFAULT, dst1, src1, N, 0, 0, npes, pWrk1, pSync1);
    roc_shmem_ctx_long_min_to_all(ROC_SHMEM_CTX_DEFAULT, dst2, src2, N, 0, 0, npes, pWrk2, pSync);
    roc_shmem_ctx_float_min_to_all(ROC_SHMEM_CTX_DEFAULT, dst3, src3, N, 0, 0, npes, pWrk3, pSync1);
    roc_shmem_ctx_double_min_to_all(ROC_SHMEM_CTX_DEFAULT, dst4, src4, N, 0, 0, npes, pWrk4, pSync);
    //roc_shmem_ctx_longdouble_min_to_all(ROC_SHMEM_CTX_DEFAULT, dst5, src5, N, 0, 0, npes, pWrk5, pSync1);
    roc_shmem_ctx_longlong_min_to_all(ROC_SHMEM_CTX_DEFAULT, dst6, src6, N, 0, 0, npes, pWrk6, pSync);

    if(me == 0) {
      for (i = 0; i < N; i++) {
         if(dst0[i] != i) ok[0] = 1;
           if(dst1[i] != i) ok[1] = 1;
           if(dst2[i] != i) ok[2] = 1;
           if(dst3[i] != i) ok[3] = 1;
           if(dst4[i] != i) ok[4] = 1;
           if(dst5[i] != i) ok[5] = 1;
           if(dst6[i] != i) ok[6] = 1;
      }
      if(ok[0]==1){
        printf("Reduction operation roc_shmem_short_min_to_all: Failed\n");
      }
      else{
        Vprintf("Reduction operation roc_shmem_short_min_to_all: Passed\n");
        pass++;
        }
        if(ok[1]==1){
        printf("Reduction operation roc_shmem_int_min_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_int_min_to_all: Passed\n");
        pass++;
        }
        if(ok[2]==1){
        printf("Reduction operation roc_shmem_long_min_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_long_min_to_all: Passed\n");
        pass++;
        }
        if(ok[3]==1){
        printf("Reduction operation roc_shmem_float_min_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_float_min_to_all: Passed\n");
        pass++;
        }
        if(ok[4]==1){
        printf("Reduction operation roc_shmem_double_min_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_double_min_to_all: Passed\n");
        pass++;
        }
        /*
        if(ok[5]==1){
        printf("Reduction operation roc_shmem_longdouble_min_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_longdouble_min_to_all: Passed\n");
        pass++;
        }
        */
        pass++;
        if(ok[6]==1){
        printf("Reduction operation roc_shmem_longlong_min_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_longlong_min_to_all: Passed\n");
        pass++;
        }
        Vprintf("\n");
    }
    if (Serialize) roc_shmem_barrier_all();

    return (pass == 7 ? 1 : 0);
}


int
sum_to_all(int me, int npes)
{
  int i, pass=0;

  memset(ok,0,sizeof(ok));

  for (i = 0; i < N; i++) {
    src0[i] = src1[i] = src2[i] = src3[i] = src4[i] = src5[i] = src6[i] = me;
    dst0[i] = -9;
    dst1[i] = -9;
    dst2[i] = -9;
    dst3[i] = -9;
    dst4[i] = -9;
    dst5[i] = -9;
    dst6[i] = -9;
  }

  roc_shmem_barrier_all();

  roc_shmem_ctx_short_sum_to_all(ROC_SHMEM_CTX_DEFAULT, dst0, src0, N, 0, 0, npes, pWrk0, pSync);
  roc_shmem_ctx_int_sum_to_all(ROC_SHMEM_CTX_DEFAULT, dst1, src1, N, 0, 0, npes, pWrk1, pSync1);
  roc_shmem_ctx_long_sum_to_all(ROC_SHMEM_CTX_DEFAULT, dst2, src2, N, 0, 0, npes, pWrk2, pSync);
  roc_shmem_ctx_float_sum_to_all(ROC_SHMEM_CTX_DEFAULT, dst3, src3, N, 0, 0, npes, pWrk3, pSync1);
  roc_shmem_ctx_double_sum_to_all(ROC_SHMEM_CTX_DEFAULT, dst4, src4, N, 0, 0, npes, pWrk4, pSync);
  //roc_shmem_ctx_longdouble_sum_to_all(ROC_SHMEM_CTX_DEFAULT, dst5, src5, N, 0, 0, npes, pWrk5, pSync1);
  roc_shmem_ctx_longlong_sum_to_all(ROC_SHMEM_CTX_DEFAULT, dst6, src6, N, 0, 0, npes, pWrk6, pSync);

  if(me == 0) {
    for (i = 0; i < N; i++) {
      if(dst0[i] != (short) (npes * (npes-1)/2)) ok[0] = 1;
      if(dst1[i] != (int) (npes * (npes-1)/2)) ok[1] = 1;
      if(dst2[i] != (long) (npes * (npes-1)/2)) ok[2] = 1;
      if(dst3[i] != (float) (npes * (npes-1)/2)) ok[3] = 1;
      if(dst4[i] != (double) (npes * (npes-1)/2)) ok[4] = 1;
      if(dst5[i] != (long double) (npes * (npes-1)/2)) ok[5] = 1;
      if(dst6[i] != (long long) (npes * (npes-1)/2)) ok[6] = 1;
    }
    if(ok[0]==1){
      printf("Reduction operation roc_shmem_short_sum_to_all: Failed\n");
    }
    else{
      Vprintf("Reduction operation roc_shmem_short_sum_to_all: Passed\n");
      pass++;
    }
    if(ok[1]==1){
      printf("Reduction operation roc_shmem_int_sum_to_all: Failed\n");
    }
    else{
      Vprintf("Reduction operation roc_shmem_int_sum_to_all: Passed\n");
      pass++;
    }
    if(ok[2]==1){
      printf("Reduction operation roc_shmem_long_sum_to_all: Failed\n");
    }
    else{
      Vprintf("Reduction operation roc_shmem_long_sum_to_all: Passed\n");
      pass++;
    }
    if(ok[3]==1){
      printf("Reduction operation roc_shmem_float_sum_to_all: Failed\n");
    }
    else{
      Vprintf("Reduction operation roc_shmem_float_sum_to_all: Passed\n");
      pass++;
    }
    if(ok[4]==1){
      printf("Reduction operation roc_shmem_double_sum_to_all: Failed\n");
    }
    else{
      Vprintf("Reduction operation roc_shmem_double_sum_to_all: Passed\n");
      pass++;
    }
    /*
    if(ok[5]==1){
      printf("Reduction operation roc_shmem_longdouble_sum_to_all: Failed\n");
    }
    else{
      Vprintf("Reduction operation roc_shmem_longdouble_sum_to_all: Passed\n");
      pass++;
    }
    */
    pass++;
    if(ok[6]==1){
      printf("Reduction operation roc_shmem_longlong_sum_to_all: Failed\n");
    }
    else{
      Vprintf("Reduction operation roc_shmem_longlong_sum_to_all: Passed\n");
      pass++;
    }
    Vprintf("\n"); fflush(stdout);
  }
    if (Serialize) roc_shmem_barrier_all();

    return (pass == 7 ? 1 : 0);
}


int
and_to_all(int me, int num_pes)
{
    int i, pass=0;

    memset(ok,0,sizeof(ok));

    for (i = 0; i < N; i++) {
        src0[i] = src1[i] = src2[i] = src6[i] = me;
        dst0[i] = dst1[i] = dst2[i] = dst6[i] = -9;
    }

    roc_shmem_barrier_all();

    roc_shmem_ctx_short_and_to_all(ROC_SHMEM_CTX_DEFAULT, dst0, src0, N, 0, 0, num_pes, pWrk0, pSync);
    roc_shmem_ctx_int_and_to_all(ROC_SHMEM_CTX_DEFAULT, dst1, src1, N, 0, 0, num_pes, pWrk1, pSync1);
    roc_shmem_ctx_long_and_to_all(ROC_SHMEM_CTX_DEFAULT, dst2, src2, N, 0, 0, num_pes, pWrk2, pSync);
    roc_shmem_ctx_longlong_and_to_all(ROC_SHMEM_CTX_DEFAULT, dst6, src6, N, 0, 0, num_pes, pWrk6, pSync1);

    if (me==0) {
      for (i = 0; i < N; i++) {
          if(dst0[i] != 0) ok[0] = 1;
          if(dst1[i] != 0) ok[1] = 1;
          if(dst2[i] != 0) ok[2] = 1;
          if(dst6[i] != 0) ok[3] = 1;
      }

      if(ok[0]==1){
        printf("Reduction operation roc_shmem_short_and_to_all: Failed\n");
      }
      else{
        Vprintf("Reduction operation roc_shmem_short_and_to_all: Passed\n");
        pass++;
        }
        if(ok[1]==1){
        printf("Reduction operation roc_shmem_int_and_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_int_and_to_all: Passed\n");
        pass++;
        }
        if(ok[2]==1){
        printf("Reduction operation roc_shmem_long_and_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_long_and_to_all: Passed\n");
        pass++;
        }
        if(ok[3]==1){
        printf("Reduction operation roc_shmem_longlong_and_to_all: Failed\n");
        }
      else{
        Vprintf("Reduction operation roc_shmem_longlong_and_to_all: Passed\n");
        pass++;
        }
      Vprintf("\n"); fflush(stdout);
    }
    if (Serialize) roc_shmem_barrier_all();

    return (pass == 4 ? 1 : 0);
}


int
prod_to_all(int me, int npes)
{
    int i, pass=0;
    int float_rounding_err=0;
    int double_rounding_err=0;
    int ldouble_rounding_err=0;

    memset(ok,0,sizeof(ok));

    for (i = 0; i < N; i++) {
        src0[i] = src1[i] = src2[i] = src3[i] = src4[i] = src5[i] = src6[i]
            = me + 1;
        dst0[i] = -9;
        dst1[i] = -9;
        dst2[i] = -9;
        dst3[i] = -9;
        dst4[i] = -9;
        dst5[i] = -9;
        dst6[i] = -9;
    }

    expected_result0 = expected_result1 = expected_result2 =
    expected_result6 = 1;
    expected_result3 = expected_result4 = expected_result5 = 1.0;


    for(i=1; i <= npes; i++) {
        expected_result0 *= i;
        expected_result1 *= i;
        expected_result2 *= i;
        expected_result3 *= (float)i;
        expected_result4 *= (double)i;
        if ((double)expected_result3 != expected_result4) {
            if (!float_rounding_err && Verbose > 2 && me == 0)
                printf("float_err @ npes %d\n",i);
            float_rounding_err = 1;
        }
        expected_result5 *= (long double)i;
        if ((long double)expected_result4 != expected_result5) {
            if (!double_rounding_err && Verbose > 2 && me == 0)
                printf("double_err @ npes %d\n",i);
            ldouble_rounding_err = double_rounding_err = 1;
        }
        expected_result6 *= i;
    }

    roc_shmem_barrier_all();

    roc_shmem_ctx_short_prod_to_all(ROC_SHMEM_CTX_DEFAULT, dst0, src0, N, 0, 0, npes, pWrk0, pSync);
    roc_shmem_ctx_int_prod_to_all(ROC_SHMEM_CTX_DEFAULT, dst1, src1, N, 0, 0, npes, pWrk1, pSync1);
    roc_shmem_ctx_long_prod_to_all(ROC_SHMEM_CTX_DEFAULT, dst2, src2, N, 0, 0, npes, pWrk2, pSync);
    roc_shmem_ctx_float_prod_to_all(ROC_SHMEM_CTX_DEFAULT, dst3, src3, N, 0, 0, npes, pWrk3, pSync1);
    roc_shmem_ctx_double_prod_to_all(ROC_SHMEM_CTX_DEFAULT, dst4, src4, N, 0, 0, npes, pWrk4, pSync);
    //roc_shmem_ctx_longdouble_prod_to_all(ROC_SHMEM_CTX_DEFAULT, dst5, src5, N, 0, 0, npes, pWrk5, pSync1);
    roc_shmem_ctx_longlong_prod_to_all(ROC_SHMEM_CTX_DEFAULT, dst6, src6, N, 0, 0, npes, pWrk6, pSync);

    if(me == 0) {
      for (i = 0; i < N; i++) {
        if(dst0[i] != expected_result0) ok[0] = 1;
        if(dst1[i] != expected_result1) ok[1] = 1;
        if(dst2[i] != expected_result2) ok[2] = 1;

        /* check for overflow */
        if(!float_rounding_err && dst3[i] != expected_result3) { ok[3] = 1;
            printf("dst3[%d]: %f, expected val: %f\n",i, dst3[i], expected_result3);
        }
        if(!double_rounding_err && dst4[i] != expected_result4) {ok[4] = 1;
            printf("dst4[%d]: %f, expected val: %f\n",i, dst4[i], expected_result4);
        }
        /*
        if(!ldouble_rounding_err && dst5[i] != expected_result5) {ok[5] = 1;
            printf("dst5[%d]: %Lf, expected val: %Lf T4 %f\n",i, dst5[i], expected_result5,dst4[i]);
        }
        */
        if(dst6[i] != expected_result6) ok[6] = 1;
      }

      if(ok[0]==1)
        printf("Reduction operation roc_shmem_short_prod_to_all: Failed\n");
      else {
        Vprintf("Reduction operation roc_shmem_short_prod_to_all: Passed\n");
        pass++;
      }

      if(ok[1]==1)
        printf("Reduction operation roc_shmem_int_prod_to_all: Failed\n");
      else {
        Vprintf("Reduction operation roc_shmem_int_prod_to_all: Passed\n");
        pass++;
      }

      if(ok[2]==1)
        printf("Reduction operation roc_shmem_long_prod_to_all: Failed\n");
      else {
        Vprintf("Reduction operation roc_shmem_long_prod_to_all: Passed\n");
        pass++;
      }

      if(ok[3]==1)
        printf("Reduction operation roc_shmem_float_prod_to_all: Failed\n");
      else {
        if (float_rounding_err) {
            Vprintf("Reduction operation roc_shmem_float_prod_to_all: skipped due to float rounding error\n");
        }
        else {
            Vprintf("Reduction operation roc_shmem_float_prod_to_all: Passed\n");
        }
        pass++;
      }

      if(ok[4]==1)
        printf("Reduction operation roc_shmem_double_prod_to_all: Failed\n");
      else {
        if (double_rounding_err) {
            Vprintf("Reduction operation roc_shmem_double_prod_to_all: skipped due to double rounding error\n");
        }
        else {
            Vprintf("Reduction operation roc_shmem_double_prod_to_all: Passed\n");
        }
        pass++;
      }

      /*
      if(ok[5]==1)
        printf("Reduction operation roc_shmem_longdouble_prod_to_all: Failed\n");
      else {
        if (double_rounding_err) {
            Vprintf("Reduction operation roc_shmem_longdouble_prod_to_all: skipped due to long double rounding error\n");
        }
        else {
            Vprintf("Reduction operation roc_shmem_longdouble_prod_to_all: Passed\n");
        }
        pass++;
      }
      */
      pass++;

      if(ok[6]==1)
        printf("Reduction operation roc_shmem_longlong_prod_to_all: Failed\n");
      else {
        Vprintf("Reduction operation roc_shmem_longlong_prod_to_all: Passed\n");
        pass++;
      }
      Vprintf("\n");
    }
    if (Serialize) roc_shmem_barrier_all();

    return (pass == 7 ? 1 : 0);
}


int
or_to_all(int me, int npes)
{
    int i, pass=0;

    memset(ok,0,sizeof(ok));

    for (i = 0; i < N; i++) {
      src0[i] = src1[i] = src2[i] = src6[i] = (me + 1)%4;
      dst0[i] = -9;
      dst1[i] = -9;
      dst2[i] = -9;
      dst6[i] = -9;
    }

    roc_shmem_barrier_all();

    roc_shmem_ctx_short_or_to_all(ROC_SHMEM_CTX_DEFAULT, dst0, src0, N, 0, 0, npes, pWrk0, pSync);
    roc_shmem_ctx_int_or_to_all(ROC_SHMEM_CTX_DEFAULT, dst1, src1, N, 0, 0, npes, pWrk1, pSync1);
    roc_shmem_ctx_long_or_to_all(ROC_SHMEM_CTX_DEFAULT, dst2, src2, N, 0, 0, npes, pWrk2, pSync);
    roc_shmem_ctx_longlong_or_to_all(ROC_SHMEM_CTX_DEFAULT, dst6, src6, N, 0, 0, npes, pWrk6, pSync1);

    if (me==0) {
        for (i = 0; i < N; i++) {
            int expected = (npes == 1) ? 1 : 3;

            if(dst0[i] != expected) ok[0] = 1;
            if(dst1[i] != expected) ok[1] = 1;
            if(dst2[i] != expected) ok[2] = 1;
            if(dst6[i] != expected) ok[6] = 1;
        }

        if(ok[0]==1)
            printf("Reduction operation roc_shmem_short_or_to_all: Failed\n");
        else {
            Vprintf("Reduction operation roc_shmem_short_or_to_all: Passed\n");
            pass++;
        }

        if(ok[1]==1)
            printf("Reduction operation roc_shmem_int_or_to_all: Failed\n");
        else {
            Vprintf("Reduction operation roc_shmem_int_or_to_all: Passed\n");
            pass++;
        }

        if(ok[2]==1)
            printf("Reduction operation roc_shmem_long_or_to_all: Failed\n");
        else {
            Vprintf("Reduction operation roc_shmem_long_or_to_all: Passed\n");
            pass++;
        }

        if(ok[6]==1)
            printf("Reduction operation roc_shmem_longlong_or_to_all: Failed\n");
        else {
            Vprintf("Reduction operation roc_shmem_longlong_or_to_all: Passed\n");
            pass++;
        }
        Vprintf("\n");
    }
    if (Serialize) roc_shmem_barrier_all();

    return (pass == 4 ? 1 : 0);
}


int
xor_to_all(int me, int npes)
{
    int i, pass=0;
    int expected_result = ((int)(npes/2) % 2);

    memset(ok,0,sizeof(ok));

    for (i = 0; i < N; i++) {
        src0[i] = src1[i] = src2[i] = src6[i] = me%2;
        dst0[i] = -9;
        dst1[i] = -9;
        dst2[i] = -9;
        dst6[i] = -9;
    }

    roc_shmem_barrier_all();

    roc_shmem_ctx_short_xor_to_all(ROC_SHMEM_CTX_DEFAULT, dst0, src0, N, 0, 0, npes, pWrk0, pSync);
    roc_shmem_ctx_int_xor_to_all(ROC_SHMEM_CTX_DEFAULT, dst1, src1, N, 0, 0, npes, pWrk1, pSync1);
    roc_shmem_ctx_long_xor_to_all(ROC_SHMEM_CTX_DEFAULT, dst2, src2, N, 0, 0, npes, pWrk2, pSync);
    roc_shmem_ctx_longlong_xor_to_all(ROC_SHMEM_CTX_DEFAULT, dst6, src6, N, 0, 0, npes, pWrk6, pSync1);

    if (me==0) {
      for (i = 0; i < N; i++) {
          if(dst0[i] != expected_result) ok[0] = 1;
          if(dst1[i] != expected_result) ok[1] = 1;
          if(dst2[i] != expected_result) ok[2] = 1;
          if(dst6[i] != expected_result) ok[6] = 1;
      }

      if(ok[0]==1)
          printf("Reduction operation roc_shmem_short_xor_to_all: Failed\n");
      else {
          Vprintf("Reduction operation roc_shmem_short_xor_to_all: Passed\n");
          pass++;
      }

      if(ok[1]==1)
          printf("Reduction operation roc_shmem_int_xor_to_all: Failed\n");
       else {
          Vprintf("Reduction operation roc_shmem_int_xor_to_all: Passed\n");
          pass++;
      }

       if(ok[2]==1)
          printf("Reduction operation roc_shmem_long_xor_to_all: Failed\n");
        else {
          Vprintf("Reduction operation roc_shmem_long_xor_to_all: Passed\n");
          pass++;
      }

        if(ok[6]==1)
          printf("Reduction operation roc_shmem_longlong_xor_to_all: Failed\n");
        else {
          Vprintf("Reduction operation roc_shmem_longlong_xor_to_all: Passed\n");
          pass++;
      }

      Vprintf("\n");
    }
    if (Serialize) roc_shmem_barrier_all();

    return (pass == 4 ? 1 : 0);
}


int
main(int argc, char* argv[])
{
    int c, i, mype, num_pes, tests, passed;
    char *pgm;

    roc_shmem_init(1);
    mype = roc_shmem_my_pe();
    num_pes = roc_shmem_n_pes();

    if ((pgm=strrchr(argv[0],'/'))) {
        pgm++;
    } else {
        pgm = argv[0];
    }

    while((c=getopt(argc,argv,"ampsSoxhv")) != -1) {
        switch(c) {
          case 'a':
            And++;  // do not run and_to_all
            break;
          case 'm':
            Min++;  // do not run min_to_all
            break;
          case 'o':
            Or++;  // do not run or_to_all
            break;
          case 'p':
            Prod++;  // do not run prod_to_all
            break;
          case 's':
            Sum++;  // do not run sum_to_all
            break;
          case 'x':
            Xor++;  // do not run xor_to_all
            break;
          case 'S':
            Serialize++;
            break;
          case 'v':
            Verbose++;
            break;
          case 'h':
          default:
            Rfprintf(stderr,"usage: %s {-v(verbose)|h(help)}\n",pgm);
            roc_shmem_finalize();
            return 1;
        }
    }

    tests = passed = 0;

    pSync = (long *)    roc_shmem_malloc(ROC_SHMEM_BCAST_SYNC_SIZE * sizeof(long));
    pSync1 = (long *)   roc_shmem_malloc(ROC_SHMEM_BCAST_SYNC_SIZE * sizeof(long));
    if (!pSync || !pSync1) {
        fprintf(stderr,"ERR: cannot allocate one of the pSync arrays\n");
    }

    for (i = 0; i < ROC_SHMEM_REDUCE_SYNC_SIZE; i++) {
        pSync[i] = ROC_SHMEM_SYNC_VALUE;
        pSync1[i] = ROC_SHMEM_SYNC_VALUE;
    }

    pWrk0 = (short *)           roc_shmem_malloc(WRK_SIZE * sizeof(short));
    pWrk1 = (int *)             roc_shmem_malloc(WRK_SIZE * sizeof(int));
    pWrk2 = (long *)            roc_shmem_malloc(WRK_SIZE * sizeof(long));
    pWrk3 = (float *)           roc_shmem_malloc(WRK_SIZE * sizeof(float));
    pWrk4 = (double *)          roc_shmem_malloc(WRK_SIZE * sizeof(double));
    pWrk5 = (long double *)     roc_shmem_malloc(WRK_SIZE * sizeof(long double));
    pWrk6 = (long long *)       roc_shmem_malloc(WRK_SIZE * sizeof(long long));
    if (!pWrk0 || !pWrk1 || !pWrk2 || !pWrk3 || !pWrk4 || !pWrk5 || !pWrk6) {
        fprintf(stderr,"ERR: cannot allocate one of the pWrk arrays\n");
    }

    src0 = (short *)           roc_shmem_malloc(N * sizeof(short));
    src1 = (int *)             roc_shmem_malloc(N * sizeof(int));
    src2 = (long *)            roc_shmem_malloc(N * sizeof(long));
    src3 = (float *)           roc_shmem_malloc(N * sizeof(float));
    src4 = (double *)          roc_shmem_malloc(N * sizeof(double));
    src5 = (long double *)     roc_shmem_malloc(N * sizeof(long double));
    src6 = (long long *)       roc_shmem_malloc(N * sizeof(long long));
    if (!src0 || !src1 || !src2 || !src3 || !src4 || !src5 || !src6) {
        fprintf(stderr,"ERR: cannot allocate one of the src arrays\n");
    }

    dst0 = (short *)           roc_shmem_malloc(N * sizeof(short));
    dst1 = (int *)             roc_shmem_malloc(N * sizeof(int));
    dst2 = (long *)            roc_shmem_malloc(N * sizeof(long));
    dst3 = (float *)           roc_shmem_malloc(N * sizeof(float));
    dst4 = (double *)          roc_shmem_malloc(N * sizeof(double));
    dst5 = (long double *)     roc_shmem_malloc(N * sizeof(long double));
    dst6 = (long long *)       roc_shmem_malloc(N * sizeof(long long));
    if (!dst0 || !dst1 || !dst2 || !dst3 || !dst4 || !dst5 || !dst6) {
        fprintf(stderr,"ERR: cannot allocate one of the dst arrays\n");
    }

    roc_shmem_barrier_all();

    passed += max_to_all(mype, num_pes);
    tests++;

    if (!Min) {
        passed += min_to_all(mype, num_pes);
        tests++;
    }

    if (!Sum) {
        passed += sum_to_all(mype, num_pes);
        tests++;
    }

    if (!And) {
        passed += and_to_all(mype, num_pes);
        tests++;
    }

    if (!Prod) {
        passed += prod_to_all(mype, num_pes);
        tests++;
    }

    if (!Or) {
        passed += or_to_all(mype, num_pes);
        tests++;
    }

    if (!Xor) {
        passed += xor_to_all(mype, num_pes);
        tests++;
    }

    c = 0;
    if (mype == 0) {
        if ((Verbose || tests != passed))
            fprintf(stderr,"to_all[%d] %d of %d tests passed\n",
                    mype,passed,tests);
        c = (tests == passed ? 0 : 1);
    }

    roc_shmem_free(pSync);
    roc_shmem_free(pSync1);

    roc_shmem_free(pWrk0);
    roc_shmem_free(pWrk1);
    roc_shmem_free(pWrk2);
    roc_shmem_free(pWrk3);
    roc_shmem_free(pWrk4);
    roc_shmem_free(pWrk5);
    roc_shmem_free(pWrk6);

    roc_shmem_free(src0);
    roc_shmem_free(src1);
    roc_shmem_free(src2);
    roc_shmem_free(src3);
    roc_shmem_free(src4);
    roc_shmem_free(src5);
    roc_shmem_free(src6);

    roc_shmem_free(dst0);
    roc_shmem_free(dst1);
    roc_shmem_free(dst2);
    roc_shmem_free(dst3);
    roc_shmem_free(dst4);
    roc_shmem_free(dst5);
    roc_shmem_free(dst6);

    roc_shmem_finalize();

    return c;
}
