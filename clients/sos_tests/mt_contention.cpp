/*
 *  Copyright (c) 2018 Intel Corporation. All rights reserved.
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

/* Multithreaded Contention Test: Overlapping AMO/quiet on a shared (default)
 * context */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <roc_shmem.hpp>

using namespace rocshmem;

#define T 8

long *dest;

int me, npes;
int errors = 0;

static void * thread_main(void *arg) {
    int tid = * (int *) arg;
    int i;

    /* Threads increment the counter on each PE and then performs a quiet.
     * All threads use the default context; thus, this checks that quiet
     * with overlapping AMOs behaves correctly. */

    for (i = 1; i <= npes; i++)
        roc_shmem_int64_atomic_add(dest, tid, (me + i) % npes);

    roc_shmem_quiet();

    return NULL;
}


int main(int argc, char **argv) {
    int tl, i, ret;
    pthread_t threads[T];
    int       t_arg[T];

    ret = roc_shmem_init_thread(ROC_SHMEM_THREAD_MULTIPLE, &tl, 1);

    if (tl != ROC_SHMEM_THREAD_MULTIPLE || ret != 0) {
        printf("Init failed (requested thread level %d, got %d, ret %d)\n",
               ROC_SHMEM_THREAD_MULTIPLE, tl, ret);

        if (ret == 0) {
            roc_shmem_global_exit(1);
        } else {
            return ret;
        }
    }

    me = roc_shmem_my_pe();
    npes = roc_shmem_n_pes();

    if (me == 0) printf("Starting multithreaded test on %d PEs, %d threads/PE\n", npes, T);

    dest = (long *) roc_shmem_malloc(sizeof(long));
    *dest = 0;

    for (i = 0; i < T; i++) {
        int err;
        t_arg[i] = i;
        err = pthread_create(&threads[i], NULL, thread_main, (void*) &t_arg[i]);
        assert(0 == err);
    }

    for (i = 0; i < T; i++) {
        int err;
        err = pthread_join(threads[i], NULL);
        assert(0 == err);
    }

    roc_shmem_sync_all();

    if ((*dest) != ((T-1)*T/2)*npes) {
        printf("%d: dest = %ld, expected %d\n", me, *dest, ((T-1)*T/2)*npes);
        errors++;
    }

    roc_shmem_free(dest);

    roc_shmem_finalize();
    return (errors == 0) ? 0 : 1;
}
