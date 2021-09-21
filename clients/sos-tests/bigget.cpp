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
 * (big get) each PE gets N elements (1 MB) from (my_pe()+1 mod num_pes()).
 */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>

#include <roc_shmem.hpp>

#define NUM_ELEMENTS 4194304 // 32 MB by longs
//#define DFLT_LOOPS 10000
#define DFLT_LOOPS 1000

int Verbose;
int Sync;
int Track;
int elements = NUM_ELEMENTS;
double *total_time;

static int
atoi_scaled(char *s)
{
    long val;
    char *e;

    val = strtol(s,&e,0);
    if (e == NULL || *e =='\0')
        return (int)val;

    if (*e == 'k' || *e == 'K')
        val *= 1024;
    else if (*e == 'm' || *e == 'M')
        val *= 1024*1024;
    else if (*e == 'g' || *e == 'G')
        val *= 1024*1024*1024;

    return (int)val;
}

static void
usage(char *pgm)
{
    fprintf(stderr,
        "usage: %s -{hvclst}\n"
        "  where: (big gets)\n"
        "    -v              be verbose, multiple 'v' more verbose\n"
        "    -e element-cnt (%d)  # of int sized elements to get\n"
        "    -l loops (%d)  loop count.\n"
        "    -s             synchronize: barrier after each roc_shmem_get()\n"
        "    -t             track: output '.' for every 200 roc_shmem_get()s\n",
        pgm,NUM_ELEMENTS,DFLT_LOOPS);
}

static inline double shmemx_wtime(void)
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)((tv.tv_usec / 1000000.0) + tv.tv_sec);
}

int
main(int argc, char **argv)
{
    int loops=DFLT_LOOPS;
    char *pgm;
    int *Target;
    int *Source;
    int i, me, npes;
    int target_pe;
    long bytes;
    double time_taken=0.0, start_time;

    roc_shmem_init();
    me = roc_shmem_my_pe();
    npes = roc_shmem_n_pes();

    if ((pgm=strrchr(argv[0],'/')))
        pgm++;
    else
        pgm = argv[0];

    while ((i = getopt (argc, argv, "hve:l:st")) != EOF) {
        switch (i)
        {
            case 'v':
                Verbose++;
                break;
            case 'e':
                if ((elements = atoi_scaled(optarg)) <= 0) {
                    fprintf(stderr,"ERR: Bad elements count %d\n",elements);
                    roc_shmem_finalize();
                    return 1;
                }
                break;
            case 'l':
                if ((loops = atoi_scaled(optarg)) <= 0) {
                    fprintf(stderr,"ERR: Bad loop count %d\n",loops);
                    roc_shmem_finalize();
                    return 1;
                }
                break;
            case 's':
                Sync++;
                break;
            case 't':
                Track++;
                break;
            case 'h':
                if (me == 0)
                    usage(pgm);
                return 0;
            default:
                if (me == 0) {
                    fprintf(stderr,"%s: unknown switch '-%c'?\n",pgm,i);
                    usage(pgm);
                }
                roc_shmem_finalize();
                return 1;
        }
    }

    target_pe = (me+1) % npes;

    total_time = (double *) roc_shmem_malloc( npes * sizeof(double) );
    if (!total_time) {
        fprintf(stderr,"ERR: bad total_time roc_shmem_malloc(%ld)\n",
                (elements * sizeof(double)));
        roc_shmem_global_exit(1);
    }

    Source = (int *) roc_shmem_malloc( elements * sizeof(*Source) );
    if (!Source) {
        fprintf(stderr,"ERR: bad Source roc_shmem_malloc(%ld)\n",
                (elements * sizeof(*Target)));
        roc_shmem_free(total_time);
        roc_shmem_global_exit(1);
    }

    Target = (int *) roc_shmem_malloc( elements * sizeof(*Target) );
    if (!Target) {
        fprintf(stderr,"ERR: bad Target roc_shmem_malloc(%ld)\n",
                (elements * sizeof(*Target)));
        roc_shmem_free(Source);
        roc_shmem_free(total_time);
        roc_shmem_global_exit(1);
    }

    for (i = 0; i < elements; i++) {
        Target[i] = -90;
        Source[i] = i + 1;
    }

    bytes = loops * sizeof(int) * elements;

    if (Verbose && me==0)
        fprintf(stderr, "%s: INFO - %d loops, get %d (int) elements from PE+1\n",
                pgm, loops, elements);

    roc_shmem_barrier_all();

    for(i=0; i < loops; i++) {

        start_time = shmemx_wtime();

        roc_shmem_int_get( Target, Source, elements, target_pe );

        time_taken += shmemx_wtime() - start_time;

        if (me==0) {
            if ( Track && i > 0 && ((i % 200) == 0))
                fprintf(stderr,".%d",i);
        }
        if (Sync)
            roc_shmem_barrier_all();
    }

    // collect time per node elapsed time.
    roc_shmem_double_put( &total_time[me], &time_taken, 1, 0 );

    roc_shmem_barrier_all();

    for (i = 0; i < elements; i++) {
        if (Target[i] != i + 1) {
            printf("%d: Error Target[%d] = %d, expected %d\n",
                   me, i, Target[i], i + 1);
            roc_shmem_global_exit(1);
        }
    }

    if ( Track && me == 0 )
        fprintf(stderr,"\n");

    if (Verbose && me == 0) {
        double rate,secs;

        // average time
        for(i=0,secs=0.0; i < npes; i++)
            secs += total_time[i];
        secs /= (double)npes;
        rate = ((double)bytes/(1024.0*1024.0)) / secs;
        printf("%s: ave %5.3f MB/sec (bytes %ld secs %5.3f)\n",
               pgm, rate, bytes, secs);
    }

    roc_shmem_free(total_time);
    roc_shmem_free(Target);
    roc_shmem_free(Source);

    roc_shmem_finalize();

    return 0;
}
