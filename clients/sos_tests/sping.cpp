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
 * SHMEM PingPong test
 * :vim:sw=4:ts=4:
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>

int getSize (char *);
double gettime(void);
double dt (double *, double *);
void usage (char *);
void help (char *);
void printStats (int, int, int, int, double);

#define dprint if (Verbose) printf
#define DFLT_MIN_WORDS 128
#define DFLT_REPS 1000;

int Verbose=0;

#include <roc_shmem.hpp>

using namespace rocshmem;

int getSize (char *str)
{
    int size;
    char mod[32];

    switch (sscanf (str, "%d%1[mMkK]", &size, mod))
    {
        case 1:
            return (size);

        case 2:
            switch (*mod)
            {
                case 'm':
                case 'M':
                    return (size << 20);

                case 'k':
                case 'K':
                    return (size << 10);

                default:
                    return (size);
            }

        default:
            return (-1);
    }
}

double gettime()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (tv.tv_sec * 1000000 + tv.tv_usec);
}

double dt (double *tv1, double *tv2)
{
    return (*tv1 - *tv2);
}

void usage (char *name)
{
    fprintf (stderr, "Usage: %s [flags] nwords [maxWords] [incWords]\n", name);
    fprintf (stderr, " %s -h\n", name);
    exit (1);
}

void help (char *name)
{
    if (roc_shmem_my_pe() == 0) {
        printf ("Usage: %s [flags] nwords [maxWords] [incWords]\n\n", name);
        printf (" Flags may be any of\n");
        printf (" -n number repititions\n");
        printf (" -e everyone print timing info\n");
        printf (" -h print this info\n\n");
        printf (" Numbers may be postfixed with 'k' or 'm'\n\n");
    }
    roc_shmem_barrier_all();
    exit (0);
}

void printStats (int proc, int peer, int doprint, int now, double t)
{
    if (doprint || (proc & 1))
        printf("%3d pinged %3d: %8d words %9.2f uSec %8.2f MB/s\n",
               proc, peer, now, t, sizeof(long)*now/(t));
}

int main (int argc, char *argv[])
{
    double t,tv[2];
    int reps = DFLT_REPS;
    int doprint = 1/*0*/;
    char *progName;
    int minWords;
    int maxWords;
    int incWords, nwords, nproc, proc, peer, c, r, i;
    long *rbuf; /* remote buffer - sink */
    long *tbuf; /* transmit buffer - src */

    roc_shmem_init(1);
    proc = roc_shmem_my_pe();
    nproc = roc_shmem_n_pes();
    if (nproc == 1) {
        fprintf(stderr, "ERR - Requires > 1 Processing Elements\n");
        roc_shmem_finalize();
        return 0;
    }

    for (progName = argv[0] + strlen(argv[0]);
         progName > argv[0] && *(progName - 1) != '/';
         progName--)
        ;

    while ((c = getopt (argc, argv, "n:evh")) != -1)
        switch (c)
        {
            case 'n':
                if ((reps = getSize (optarg)) <= 0)
                    usage (progName);
                break;
            case 'e':
                doprint++;
                break;
            case 'v':
                Verbose++;
                break;
            case 'h':
                help (progName);
            default:
                usage (progName);
        }

    if (optind == argc)
        minWords = DFLT_MIN_WORDS;
    else if ((minWords = getSize (argv[optind++])) <= 0)
        usage (progName);

    if (optind == argc)
        maxWords = minWords;
    else if ((maxWords = getSize (argv[optind++])) < minWords)
        usage (progName);

    if (optind == argc)
        incWords = 0;
    else if ((incWords = getSize (argv[optind++])) < 0)
        usage (progName);

    if (!(rbuf = (long *)roc_shmem_malloc(maxWords * sizeof(long))))
    {
        perror ("Failed memory allocation");
        exit (1);
    }
    memset (rbuf, 0, maxWords * sizeof (long));

    if (!(tbuf = (long *)roc_shmem_malloc(maxWords * sizeof(long))))
    {
        perror ("Failed memory allocation");
        exit (1);
    }

    for (i = 0; i < maxWords; i++)
        tbuf[i] = 1000 + (i & 255);

    if (doprint)
        printf ("%d(%d): Shmem PING reps %d minWords %d maxWords %d "
                "incWords %d\n",
                proc, nproc, reps, minWords, maxWords, incWords);

    dprint("[%d] rbuf: %ld\n", proc, (unsigned long) rbuf);

    roc_shmem_barrier_all();

    peer = proc ^ 1;
    if (peer >= nproc)
        doprint = 0;

    for (nwords = minWords;
         nwords <= maxWords;
         nwords = incWords ? nwords + incWords : nwords ? 2 * nwords : 1)
    {
        r = reps;
        roc_shmem_barrier_all();
        tv[0] = gettime();
        if (peer < nproc)
        {
            if (proc & 1)
            {
                r--;
                roc_shmem_long_wait_until(&rbuf[nwords-1], ROC_SHMEM_CMP_NE, 0);
                rbuf[nwords-1] = 0;
            }

            while (r-- > 0)
            {
                roc_shmem_long_put(rbuf, tbuf, nwords, peer);
                roc_shmem_long_wait_until(&rbuf[nwords-1], ROC_SHMEM_CMP_NE, 0);
                rbuf[nwords-1] = 0;
            }

            if (proc & 1)
            {
                roc_shmem_long_put(rbuf, tbuf, nwords, peer);
            }
        }
        tv[1] = gettime();
        t = dt (&tv[1], &tv[0]) / (2 * reps);

        roc_shmem_barrier_all();

        printStats (proc, peer, doprint, nwords, t);
    }

    roc_shmem_free(rbuf);
    roc_shmem_free(tbuf);

    roc_shmem_finalize();

    return 0;
}
