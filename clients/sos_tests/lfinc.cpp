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

/* long_finc neighbor - Perf test roc_shmem_atomic_fetch_inc(); */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

#define LOOPS 25000

static double shmemx_wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
}

int Verbose;
double elapsed;

int main( int argc, char *argv[])
{
    int rc=0, my_pe, npes, neighbor;
    int loops=LOOPS;
    int j;
    size_t data_sz=sizeof(long) * 3;
    double start_time;
    long *data, lval=0;

    if (argc > 1)
        loops = atoi(argv[1]);

    roc_shmem_init(1);

    my_pe = roc_shmem_my_pe();
    npes = roc_shmem_n_pes();

    if (loops <= 0) {
        if (my_pe == 0)
            printf("Error: loops must be greater than 0\n");

        roc_shmem_finalize();
        return 1;
    }

    data = (long *) roc_shmem_malloc(data_sz);
    if (!data) {
        fprintf(stderr,"[%d] roc_shmem_malloc(%ld) failure? %d\n",
                my_pe,data_sz,errno);
        roc_shmem_global_exit(1);
    }
    memset((void*)data,0,data_sz);

    roc_shmem_barrier_all();

    neighbor = (my_pe + 1) % npes;
    start_time = shmemx_wtime();
    for(j=0,elapsed=0.0; j < loops; j++) {
        start_time = shmemx_wtime();
        lval = roc_shmem_int64_atomic_fetch_inc( (int64_t*)&data[1], neighbor );
        elapsed += shmemx_wtime() - start_time;
        if (lval != (long) j) {
            fprintf(stderr,"[%d] Test: FAIL previous val %ld != %d Exit.\n",
                    my_pe, lval, j);
            roc_shmem_global_exit(1);
        }
    }
    roc_shmem_barrier_all();

    rc = 0;
    if (data[1] != (long)loops) {
        fprintf(stderr,"[%d] finc neighbot: FAIL data[1](%p) %ld != %d Exit.\n",
                    my_pe, (void*)&data[1], data[1], loops);
        rc--;
    }

    /* check if adjancent memory locations distrubed */
    assert(data[0] == 0);
    assert(data[2] == 0);

    if (my_pe == 0 ) {
        if (rc == 0 && Verbose)
            fprintf(stderr,"[%d] finc neighbor: PASSED.\n",my_pe);
        fprintf(stderr,"[%d] %d loops of roc_shmem_int64_atomic_fetch_inc() in %6.4f secs\n"
                "  %2.6f usecs per roc_shmem_int64_atomic_fetch_inc()\n",
                    my_pe,loops,elapsed,((elapsed*100000.0)/(double)loops));
    }
    roc_shmem_free(data);

    roc_shmem_finalize();

    return rc;
}

