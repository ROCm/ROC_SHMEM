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
 * roc_shmem_barrier() test  barrier {-V} {loop-cnt}
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

#define Rfprintf \
  if (roc_shmem_my_pe() == 0) fprintf
#define Rprintf \
  if (roc_shmem_my_pe() == 0) printf
#define RDfprintf \
  if (Verbose && roc_shmem_my_pe() == 0) fprintf
#define RDprintf \
  if (Verbose && roc_shmem_my_pe() == 0) printf

int Verbose;

int main(int argc, char* argv[]) {
  int c, j, loops;
  int rank, num_ranks;
  char* prog_name;

  roc_shmem_init();
  rank = roc_shmem_my_pe();
  num_ranks = roc_shmem_n_pes();
  if (num_ranks == 1) {
    Rfprintf(stderr, "ERR - Requires > 1 PEs\n");
    roc_shmem_finalize();
    return 0;
  }
  prog_name = strrchr(argv[0], '/');
  if (prog_name)
    prog_name++;
  else
    prog_name = argv[0];

  while ((c = getopt(argc, argv, "v")) != -1) {
    switch (c) {
      case 'V':
        Verbose++;
        break;
      default:
        Rfprintf(stderr, "ERR - unknown -%c ?\n", c);
        roc_shmem_finalize();
        return 1;
    }
  }

  if (optind == argc)
    loops = 30;
  else {
    loops = atoi(argv[optind++]);
    if (loops <= 0 || loops > 1000000) {
      Rfprintf(stderr, "ERR - loops arg out of bounds '%d'?\n", loops);
      roc_shmem_finalize();
      return 1;
    }
  }

  for (j = 0; j < loops; j++) {
    // if ( j==0 || (j % 10) == 0 )
    RDfprintf(stderr, "[%d] pre-barrier(%d)\n", rank, j);

    roc_shmem_barrier_all(); /* sync sender and receiver */

    // if ( j==0 || (j % 10) == 0 )
    RDfprintf(stderr, "[%d] post barrier(%d)\n", rank, j);
  }

  RDprintf("%d(%d) Exit\n", rank, num_ranks);

  roc_shmem_finalize();

  return 0;
}
