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
#include <roc_shmem.hpp>

using namespace rocshmem;

#define NELEM 10

int main(void)
{
    int me, npes;
    int errors = 0;
    long *psync, *pwrk, *src;

    roc_shmem_init(1);

    me = roc_shmem_my_pe();
    npes = roc_shmem_n_pes();

    src = (long *) roc_shmem_malloc(NELEM * sizeof(long));
    for (int i = 0; i < NELEM; i++)
        src[i] = me;

    psync = (long *) roc_shmem_malloc(ROC_SHMEM_REDUCE_SYNC_SIZE * sizeof(long));
    for (int i = 0; i < ROC_SHMEM_REDUCE_SYNC_SIZE; i++)
        psync[i] = ROC_SHMEM_SYNC_VALUE;

    pwrk = (long *) roc_shmem_malloc((NELEM/2 + ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE) * sizeof(long));

    roc_shmem_barrier_all();

    roc_shmem_ctx_long_max_to_all(ROC_SHMEM_CTX_DEFAULT, src, src, NELEM, 0, 0, npes, pwrk, psync);

    /* Validate reduced data */
    for (int j = 0; j < NELEM; j++) {
        long expected = npes-1;
        if (src[j] != expected) {
            printf("%d: Expected src[%d] = %ld, got src[%d] = %ld\n", me, j, expected, j, src[j]);
            errors++;
        }
    }

    roc_shmem_free(src);
    roc_shmem_free(psync);
    roc_shmem_free(pwrk);

    roc_shmem_finalize();

    return errors != 0;
}
