/*
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

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <roc_shmem.hpp>

using namespace rocshmem;

#define NELEM 10

/* Note: Need to alternate psync arrays because the active set changes */

int main(void)
{
    int i, me, npes;
    int errors = 0;
    long *bcast_psync;
    //long *barrier_psync0, *barrier_psync1;
    long long *src, *dst;

    roc_shmem_init(1);

    me = roc_shmem_my_pe();
    npes = roc_shmem_n_pes();

    src = (long long *) roc_shmem_malloc(NELEM * sizeof(long long));
    dst = (long long *) roc_shmem_malloc(NELEM * sizeof(long long));
    for (i = 0; i < NELEM; i++) {
        src[i] = me;
        dst[i] = -1;
    }

    bcast_psync = (long *) roc_shmem_malloc(ROC_SHMEM_BCAST_SYNC_SIZE * sizeof(long));
    for (i = 0; i < ROC_SHMEM_BCAST_SYNC_SIZE; i++)
        bcast_psync[i] = ROC_SHMEM_SYNC_VALUE;

    /*
    barrier_psync0 = (long *) roc_shmem_malloc(ROC_SHMEM_BCAST_SYNC_SIZE * sizeof(long));
    barrier_psync1 = (long *) roc_shmem_malloc(ROC_SHMEM_BCAST_SYNC_SIZE * sizeof(long));
    for (i = 0; i < ROC_SHMEM_BARRIER_SYNC_SIZE; i++) {
        barrier_psync0[i] = ROC_SHMEM_SYNC_VALUE;
        barrier_psync1[i] = ROC_SHMEM_SYNC_VALUE;
    }
    */

    if (me == 0)
        printf("Shrinking active set test\n");

    roc_shmem_barrier_all();

    /* A total of npes tests are performed, where the active set in each test
     * includes PEs i..npes-1 */
    for (i = 0; i <= me; i++) {
        int j;

        if (me == i)
            printf(" + active set size %d\n", npes-i);

        roc_shmem_ctx_longlong_broadcast(ROC_SHMEM_CTX_DEFAULT, dst, src, NELEM, 0, i, 0, npes-i, bcast_psync);

        /* Validate broadcasted data */
        for (j = 0; j < NELEM; j++) {
            long long expected = (me == i) ? i-1 : i;
            if (dst[j] != expected) {
                printf("%d: Expected dst[%d] = %lld, got dst[%d] = %lld, iteration %d\n",
                       me, j, expected, j, dst[j], i);
                errors++;
            }
        }

        //roc_shmem_barrier(i, 0, npes-i, (i % 2) ? barrier_psync0 : barrier_psync1);
    }

    roc_shmem_barrier_all();

    for (i = 0; i < NELEM; i++)
        dst[i] = -1;

    if (me == 0)
        printf("Changing root test\n");

    roc_shmem_barrier_all();

    /* A total of npes tests are performed, where the root changes each time */
    for (i = 0; i < npes; i++) {
        int j;

        if (me == i)
            printf(" + root %d\n", i);

        roc_shmem_ctx_longlong_broadcast(ROC_SHMEM_CTX_DEFAULT, dst, src, NELEM, i, 0, 0, npes, bcast_psync);

        /* Validate broadcasted data */
        for (j = 0; j < NELEM; j++) {
            long long expected = (me == i) ? i-1 : i;
            if (dst[j] != expected) {
                printf("%d: Expected dst[%d] = %lld, got dst[%d] = %lld, iteration %d\n",
                       me, j, expected, j, dst[j], i);
                errors++;
            }
        }

        //roc_shmem_barrier(0, 0, npes, barrier_psync0);
    }

    roc_shmem_free(src);
    roc_shmem_free(dst);

    roc_shmem_free(bcast_psync);

    roc_shmem_finalize();

    return errors != 0;
}
