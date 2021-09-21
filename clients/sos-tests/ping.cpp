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
 * ping test  usage: ping {-V} {longs-per-put(128)} {loop-cnt(10)}
 *
 * For yod -c X ./ping
 * Proc0 puts nWords to Proc1 ... Proc(X-1)
 */
#include <roc_shmem.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>

#define Rfprintf if (roc_shmem_my_pe() == 0) fprintf
#define Rprintf if (roc_shmem_my_pe() == 0)  printf

#define RDprintf if (Verbose && roc_shmem_my_pe() == 0)  printf
#define RDfprintf if (Verbose && roc_shmem_my_pe() == 0) fprintf

/* option flags */
#define OUTPUT_MOD 1 // output debug every X loops
int output_mod = OUTPUT_MOD;
int Verbose;
int Slow;

#define DFLT_NWORDS 128
#define TARGET_SZ 8192
#define VAL 0xCafeBabe

long src[TARGET_SZ];

int
main(int argc, char* argv[])
{
    int c, j, loops, k;
    int proc, num_procs;
    int  nWords=1;
    int  failures=0;
    char *prog_name;
    long *Target;

    roc_shmem_init();
    proc = roc_shmem_my_pe();
    num_procs = roc_shmem_n_pes();

    if (num_procs == 1) {
        Rfprintf(stderr,
                 "ERR - Requires > 1 PEs\n");
        roc_shmem_finalize();
        return 0;
    }

    prog_name = strrchr(argv[0],'/');
    if ( prog_name )
        prog_name++;
    else
        prog_name = argv[0];

    while((c=getopt(argc,argv,"hVM:s")) != -1) {
        switch(c) {
            case 's':
                Slow++;
                break;
            case 'V':
                Verbose++;
                break;
            case 'M':
                output_mod = atoi(optarg);
                if (output_mod <= 0) {
                    Rfprintf(stderr, "ERR - output modulo arg out of "
                             "bounds '%d'?]\n", output_mod);
                    roc_shmem_finalize();
                    return 1;
                }
                Rfprintf(stderr,"%s: output modulo %d\n",
                         prog_name,output_mod);
                break;
            case 'h':
                Rfprintf(stderr,
                         "usage: %s {nWords-2-put} {Loop-count}\n",
                         prog_name);
                roc_shmem_finalize();
                return 1;
            default:
                roc_shmem_finalize();
                return 1;
        }
    }

    if (optind == argc)
        nWords = DFLT_NWORDS;
    else {
        nWords = atoi(argv[optind++]);
        if (nWords <= 0 || nWords > TARGET_SZ) {
            Rfprintf(stderr,
                     "ERR - nWords arg out of bounds '%d' [1..%d]?\n",
                     nWords, TARGET_SZ);
            roc_shmem_finalize();
            return 1;
        }
    }

    if (optind == argc)
        loops = 10;
    else {
        loops = atoi(argv[optind++]);
        if (loops <= 0 || loops > 1000000) {
            Rfprintf(stderr,
                     "ERR - loops arg out of bounds '%d'?\n", loops);
            roc_shmem_finalize();
            return 1;
        }
    }

    //Rprintf("%s: %d loops of %d longs per put\n",prog_name,loops,nWords);

    for(j=0; j < nWords; j++)
        src[j] = VAL;

    Target = (long *) roc_shmem_malloc(TARGET_SZ * sizeof(long));

    for(j=0; j < loops; j++) {

        roc_shmem_barrier_all();

        if ( Verbose && (j==0 || (j % output_mod) == 0) )
            fprintf(stderr,"[%d] +(%d)\n", roc_shmem_my_pe(),j);

        if ( proc == 0 ) {
            int p;
            for(p=1; p < num_procs; p++)
                roc_shmem_long_put(Target, src, nWords, p);
        }
        else {
            if (Slow) {
                /* wait for each put to complete */
                for(k=0; k < nWords; k++)
                    roc_shmem_long_wait_until(&Target[k], ROC_SHMEM_CMP_NE, proc);
            } else {
                /* wait for last word to be written */
                roc_shmem_long_wait_until(&Target[nWords-1], ROC_SHMEM_CMP_NE, proc);
            }
        }

        if ( Verbose && (j==0 || (j % output_mod) == 0) )
            fprintf(stderr,"[%d] -(%d)\n", roc_shmem_my_pe(),j);

        roc_shmem_barrier_all();

        if ( proc != 0 ) {
            for(k=0; k < nWords; k++) {
                if (Target[k] != VAL) {
                    fprintf(stderr, "[%d] Target[%d] %#lx "
                            "!= %#x\?\n",
                            proc,k,Target[k],VAL);
                    failures++;
                }
                //assert(Target[proc] == VAL);
                Target[k] = 0;
            }
        }
        else
            memset(Target, 0, TARGET_SZ);
    }

    roc_shmem_barrier_all();

    if (failures || Verbose)
        Rprintf ("%d(%d) Exit(%d)\n", proc, num_procs, failures);

    roc_shmem_free(Target);

    roc_shmem_finalize();

    return failures;
}
