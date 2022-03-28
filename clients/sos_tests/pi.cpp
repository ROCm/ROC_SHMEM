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

#include <roc_shmem.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>

using namespace rocshmem;

#define NUM_POINTS 10000


int
main(int argc, char* argv[], char *envp[])
{
    int me, myroc_shmem_n_pes;
    long long *inside, *total;

    /*
    ** Starts/Initializes SHMEM/OpenSHMEM
    */
    roc_shmem_init(1);
    /*
    ** Fetch the number or processes
    ** Some implementations use num_pes();
    */
    myroc_shmem_n_pes = roc_shmem_n_pes();
    /*
    ** Assign my process ID to me
    */
    me = roc_shmem_my_pe();

    inside = (long long *) roc_shmem_malloc(sizeof(long long));
    total = (long long *) roc_shmem_malloc(sizeof(long long));
    *inside = *total = 0;

    srand(1+me);

    for((*total) = 0; (*total) < NUM_POINTS; ++(*total)) {
        double x,y;
        x = rand()/(double)RAND_MAX;
        y = rand()/(double)RAND_MAX;

        if(x*x + y*y < 1) {
            ++(*inside);
        }
    }

    roc_shmem_barrier_all();

    int errors = 0;

    if(me == 0) {
        for(int i = 1; i < myroc_shmem_n_pes; ++i) {
            long long remoteInside,remoteTotal;
            roc_shmem_longlong_get(&remoteInside,inside,1,i);
            roc_shmem_longlong_get(&remoteTotal,total,1,i);
            (*total) += remoteTotal;
            (*inside) += remoteInside;
        }

        double approx_pi = 4.0*(*inside)/(double)(*total);

        if(fabs(M_PI-approx_pi) > 0.1) {
            ++errors;
        }

        if (NULL == getenv("MAKELEVEL")) {
            printf("Pi from %llu points on %d PEs: %lf\n", *total, myroc_shmem_n_pes, approx_pi);
        }
    }

    roc_shmem_free(inside);
    roc_shmem_free(total);

    roc_shmem_finalize();

    return errors;
}
