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

/* Synopsis: Test that a single asymmetric allocation works correctly.
 *
 * This semantic is provided in OpenSHMEM 1.1 and some versions of Cray SHMEM.
 * It was removed from OpenSHMEM in 1.2, but we maintain it for backward
 * compatibility.
 */

#include <stdio.h>
#include <stdlib.h>
#include <roc_shmem.hpp>

using namespace rocshmem;

long bufsize;

int main(int argc, char **argv) {
    int *buf, *buf_in;
    int me, npes, i, target;

    roc_shmem_init(1);
    me = roc_shmem_my_pe();
    npes = roc_shmem_n_pes();

    /* Each PE allocates space for "me + 1" integers */
    bufsize = me + 1;
    buf = (int*) roc_shmem_malloc(sizeof(int) * bufsize);

    if (NULL == buf)
        roc_shmem_global_exit(1);

    for (i = 0; i < bufsize; i++)
        buf[i] = -1;

    roc_shmem_barrier_all();

    /* Write to neighbor's buffer */
    target = (me + 1) % npes;
    buf_in = (int*) malloc(sizeof(int) * (target + 1));
    if (!buf_in) {
        fprintf(stderr, "ERR - null buf_in pointer\n");
        roc_shmem_global_exit(1);
    }

    for (i = 0; i < target + 1; i++)
        buf_in[i] = target;

    roc_shmem_int_put(buf, buf_in, target + 1, target);

    roc_shmem_barrier_all();

    /* Validate data was written correctly */
    for (i = 0; i < me + 1; i++) {
        if (buf[i] != me) {
            printf("Error [%3d]: buf[%d] == %d, expected %d\n", me, i, buf[i], me);
            roc_shmem_global_exit(2);
        }
    }

    free(buf_in);
    roc_shmem_free(buf);
    roc_shmem_finalize();
    return 0;
}
