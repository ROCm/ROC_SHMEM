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

/*
 *for back to back operation testing: independent buffers for each operation
 *as well as alternating PE waiting
 * */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <roc_shmem.hpp>
#include <assert.h>
#include <string.h>
#include <ctype.h>

typedef enum {
    NUM_WRITE = 8,
    NUM_READ =  5,
    NUM_SYNC = 3
} max_ops;

long *target;
long *source;
long *sync_pes;

int verbose;
int debug;


static inline void wait_until(long *wait_var, int iterations, int pe)
{

    if (debug)
        printf("PE %d waiting...%ld\n", pe, *wait_var);

    roc_shmem_long_wait_until(wait_var, ROC_SHMEM_CMP_EQ, iterations);

    if (debug)
        printf("PE %d wait_until passed\n", pe);

}

static inline void pre_op_check(const char *op,
                                int check_var, int iterations, int pe)
{
    if (verbose)
        printf("SHMEM %s, performing %d iterations\n",
               op, iterations);

    if (debug)
        printf("BEFORE operation PE %d target = %d\n",
               pe, check_var);
}

static inline void post_op_check(const char *op,
                                 int check_var, int iterations, int pe)
{

    if (check_var != iterations) {
        fprintf(stderr, "%s ERR: PE %d source = %d != %d\n",
                op, pe, check_var, iterations);
        roc_shmem_global_exit(EXIT_FAILURE);
    }
}

static inline void putfence(int me, int iterations, int T)
{
    int i;

    if (me == 1)
        pre_op_check(__func__, target[T], iterations, 1);

    if (me == 0) {
        for (i = 1; i < iterations; i++) {
            roc_shmem_long_p(&target[T], i, 1);
            roc_shmem_fence();
        }

        roc_shmem_long_p(&target[T], i, 1);

    } else
        wait_until(&target[T], iterations, 1);

    if (verbose)
        if (me == 0)
            printf("SHMEM %s finished\n", __func__);

}


static inline void gettest(int me, int iterations, int T, int S, int P)
{

    int i;

    if (me == 1) {
        pre_op_check(__func__, target[T], iterations, 1);

        roc_shmem_long_p(&source[S], iterations, 0);
        roc_shmem_fence();

        for (i = 0; i < iterations; i++)
            target[T] = roc_shmem_long_g(&source[S], 0);

        roc_shmem_long_p(&sync_pes[P], iterations, 0);

        post_op_check("get", target[T], iterations, 1);

    } else
        wait_until(&sync_pes[P], iterations, 0);

    if (verbose) {
        if (me == 0)
            printf("SHMEM %s finished\n", __func__);
    }
}

static inline void atomic_inc(int me, int iterations, int T)
{

    int i;

    if (me == 1)
        pre_op_check(__func__, target[T], iterations, 1);

    target[T] = 0;
    roc_shmem_barrier_all();

    if (me == 0) {
        for (i = 0; i < iterations; i++) {
            roc_shmem_int64_atomic_inc( (int64_t*) &target[T], 1);
            roc_shmem_fence();
        }
        roc_shmem_int64_atomic_inc( (int64_t*) &target[T], 1);

        if (debug)
            printf("PE 0 done with operation\n");

    } else
        wait_until(&target[T], (iterations+1), 1);

    if (verbose) {
        if (me == 1)
            printf("SHMEM %s finished\n", __func__);
    }
}

static inline void atomic_add(int me, int iterations, int T)
{

    int i;

    if (me == 0)
        pre_op_check(__func__, target[T], iterations, 0);

    target[T] = 0;
    roc_shmem_barrier_all();

    if (me == 1) {
        for (i = 0; i < iterations; i++) {
            roc_shmem_int64_atomic_add( (int64_t*) &target[T], 1, 0);
            roc_shmem_fence();
        }
        roc_shmem_int64_atomic_add( (int64_t*) &target[T], 1, 0);

        if (debug)
            printf("PE 1 done with operation\n");

    } else
        wait_until(&target[T], (iterations+1), 0);

    if (verbose) {
        if (me == 1)
            printf("SHMEM %s finished\n", __func__);
    }
}

/*
static inline void swaptest(int me, int iterations, int T, int S, int P)
{

    int i;
    const int tswap = 5, sswap = 2;
    target[T] = tswap;
    source[S] = sswap;

    roc_shmem_barrier_all(); /* Ensure target/source initialization completed */
/*

    if (me == 0)
        pre_op_check(__func__, source[S], iterations, 0);

    if (me == 0) {
        for (i = 0; i < iterations; i++)
            source[S] = roc_shmem_long_atomic_swap(&target[T], source[S], 1);

        roc_shmem_long_p(&sync_pes[P], i, 1);

        if (debug)
            printf("AFTER flag PE 0 value of source is %d"
                   " = 5?\n", source[S]);

        if (((iterations % 2 == 1) && (source[S] != tswap)) ||
            ((iterations % 2 == 0) &&
             (source[S] != sswap))) {
            fprintf(stderr, "swap ERR: PE 0 source = %d\n",
                    source[S]);
            roc_shmem_global_exit(EXIT_FAILURE);
        }

    } else {
        wait_until(&sync_pes[P], iterations, 1);

        if (((iterations % 2 == 1) && (target[T] != sswap)) ||
            ((iterations % 2 == 0) &&
             (target[T] != tswap))) {
            fprintf(stderr, "swap ERR: PE 0 target = %d \n",
                    target[T]);
            roc_shmem_global_exit(EXIT_FAILURE);
        }

    }

    if (verbose) {
        if (me == 0)
            printf("SHMEM %s finished\n", __func__);
    }
}
*/

static inline void cswaptest(int me, int iterations, int T, int S, int P)
{

    int i;
    source[S] = -100;

    target[T] = 0;
    roc_shmem_barrier_all();

    if (me == 1) {
        pre_op_check(__func__, source[S], iterations, 1);

        for (i = 0; i < iterations; i++)
            source[S] = roc_shmem_int64_atomic_compare_swap( (int64_t*) &(target[T]), i, (i+1), 0);

        roc_shmem_long_p(&sync_pes[P], i, 0);

        post_op_check("compare_swap", source[S], (iterations-1), 1);

    } else {
        wait_until(&sync_pes[P], iterations, 0);

        if (target[T] != iterations) {
            fprintf(stderr, "compare_swap ERR: PE 1 target = %ld != %d\n",
                    target[T], iterations);
            roc_shmem_global_exit(EXIT_FAILURE);
        }
    }

    if (verbose) {
        if (me == 1)
            printf("SHMEM %s finished\n", __func__);
    }
}

static inline void fetchatomic_add(int me, int iterations, int T, int S)
{

    int i;

    if (me == 1)
        pre_op_check(__func__, target[T], iterations, 1);

    target[T] = 0;
    roc_shmem_barrier_all();

    if (me == 0) {
        if (debug) {
            printf("BEFORE flag PE 0 value of source is"
                   " %ld = 0?\n", source[S]);
        }

        for (i = 0; i < iterations; i++) {
            source[S] = roc_shmem_int64_atomic_fetch_add( (int64_t*) &target[T], 1, 1);
            roc_shmem_fence();
        }
        source[S] = roc_shmem_int64_atomic_fetch_add( (int64_t*) &target[T], 1, 1);

        post_op_check("fetch_add", source[S], iterations, 0);

    } else
        wait_until(&target[T], (iterations+1), 1);

    if (verbose) {
        if (me == 0)
            printf("SHMEM %s finished\n", __func__);
    }
}

static inline void fetchatomic_inc(int me, int iterations, int T, int S)
{

    int i;

    if (me == 0)
        pre_op_check(__func__, target[T], iterations, 0);

    target[T] = 0;
    roc_shmem_barrier_all();

    if (me == 1) {
        if (debug) {
            printf("BEFORE flag PE 1 value of source is %ld\n",
                   source[S]);
        }

        for (i = 0; i < iterations; i++) {
            source[S] = roc_shmem_int64_atomic_fetch_inc( (int64_t*) &target[T], 0);
            roc_shmem_fence();
        }

        post_op_check("fetch_inc", source[S], (iterations-1), 1);
    } else
        wait_until(&target[T], iterations, 0);

    if (verbose) {
        if (me == 1)
            printf("SHMEM %s finished\n", __func__);
    }

}

int main(int argc, char **argv)
{
    int        me, nproc;
    int        c, all_ops = 1;
    int        T = 0, S = 0, P = 0;
    const int  DEFAULT_ITR = 7;
    int        iterations = DEFAULT_ITR;

    roc_shmem_init();

    me = roc_shmem_my_pe();
    nproc = roc_shmem_n_pes();

    target      = (long *) roc_shmem_malloc(NUM_WRITE * sizeof(long));
    source      = (long *) roc_shmem_malloc(NUM_READ  * sizeof(long));
    sync_pes    = (long *) roc_shmem_malloc(NUM_SYNC  * sizeof(long));

    memset(target, -1, NUM_WRITE * sizeof(int));
    memset(source, -1, NUM_READ * sizeof(int));
    memset(sync_pes, -1, NUM_SYNC * sizeof(int));

    roc_shmem_barrier_all();

    if (nproc != 2) {
        if (me == 0) {
            fprintf(stderr, "This is a micro test and is only "
                    "intended to run on exactly two processes you"
                    " are using %d\n", nproc);
        }
        roc_shmem_finalize();
        return 0;
    }

    while ((c = getopt(argc, argv, "i:vdpgaAscfFh")) != -1) {
        switch (c) {
            case 'i':
                iterations = atoi(optarg);
                assert(iterations > 0);
                all_ops += 2;
                break;
            case 'v':
                verbose = 1;
                all_ops++;
                break;
            case 'd':
                debug = 1;
                break;
            case 'p':
                putfence(me, iterations, T++);
                break;
            case 'g':
                gettest(me, iterations, T++, S++, P++);
                break;
            case 'a':
                atomic_add(me, iterations, T++);
                break;
            case 'A':
                atomic_inc(me, iterations, T++);
                break;
            /*
            case 's':
                swaptest(me, iterations, T++, S++, P++);
                break;
            */
            case 'c':
                cswaptest(me, iterations, T++, S++, P++);
                break;
            case 'f':
                fetchatomic_add(me, iterations, T++, S++);
                break;
            case 'F':
                fetchatomic_inc(me, iterations, T++, S++);
                break;
            case 'h':
            default:
                if (me == 0) {
                    fprintf(stderr, "input options:\n 1) single"
                            " argument option will run all tests by default"
                            "and additionally request:  -v (verbose) | "
                            "-i <number of interations>\n");
                    fprintf(stderr, " 2) two argument options "
                            "choose any combination of the following "
                            "to run individual tests:  -i <iterations>, -v"
                            ", -d, -p, -g, -a, -A, -s, -c, -f, -F, -h\n");
                }
                roc_shmem_finalize();
                return 1;
        }
    }

    if (argc == all_ops || argc == 1) {
        putfence(me, iterations,  T++);
        gettest(me, iterations, T++, S++, P++);
        atomic_add(me, iterations, T++);
        atomic_inc(me, iterations, T++);
        //swaptest(me, iterations, T++, S++, P++);
        cswaptest(me, iterations, T++, S++, P++);
        fetchatomic_add(me, iterations, T++, S++);
        fetchatomic_inc(me, iterations, T++, S++);
    }

    if (verbose) {
        if (me == 1)
            printf("PE 1: PASS: %8d iterations\n", iterations);
        else
            printf("PE 0 Successful exit\n");
    }

    roc_shmem_free(target);
    roc_shmem_free(source);
    roc_shmem_free(sync_pes);

    roc_shmem_finalize();

    return 0;
}
