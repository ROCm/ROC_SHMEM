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
 * broadcast [0...num_pes]
 *
 * usage: bcast {-v|h}
 *
 * Loop - roc_shmem_broadcast_all() with increasing data amount.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

#define START_BCAST_SIZE 16
#define BCAST_INCR 1024

int main(int argc, char *argv[]) {
  int i, Verbose = 0;
  int mpe, num_pes, loops = 10, cloop;
  char *pgm;
  long *dst, *src;
  int nBytes = START_BCAST_SIZE;
  int nLongs = 0;
  long *pSync;

  roc_shmem_init();
  mpe = roc_shmem_my_pe();
  num_pes = roc_shmem_n_pes();

  if (num_pes == 1) {
    printf("%s: Requires number of PEs > 1\n", argv[0]);
    roc_shmem_finalize();
    return 0;
  }

  if (sizeof(long) != 8) {
    printf("Test assumes 64-bit long (%zd)\n", sizeof(long));
    roc_shmem_global_exit(1);
    return 0;
  }

  if ((pgm = strrchr(argv[0], '/'))) {
    pgm++;
  } else {
    pgm = argv[0];
  }

  if (argc > 1) {
    if (strncmp(argv[1], "-v", 3) == 0) {
      Verbose = 1;
    } else if (strncmp(argv[1], "-h", 3) == 0) {
      fprintf(stderr, "usage: %s {-v(verbose)|h(help)}\n", pgm);
      roc_shmem_finalize();
      exit(1);
    }
  }

  pSync = (long *)roc_shmem_malloc(ROC_SHMEM_BCAST_SYNC_SIZE);

  for (i = 0; i < ROC_SHMEM_BCAST_SYNC_SIZE; i += 1) {
    pSync[i] = ROC_SHMEM_SYNC_VALUE;
  }

  if (mpe == 0 && Verbose) {
    fprintf(stderr, "%d loops\n", loops);
  }

  for (cloop = 1; cloop <= loops; cloop++) {
    nLongs = nBytes / sizeof(long);
    dst = (long *)roc_shmem_malloc(nBytes * 2);
    if (!dst) {
      fprintf(stderr, "[%d] roc_shmem_malloc(%d) failed %s\n", mpe, nBytes,
              strerror(errno));
      return 0;
    }
    memset((void *)dst, 0, nBytes);
    src = &dst[nLongs];
    for (i = 1; i < nLongs; i++) {
      src[i] = i + 1;
    }

    roc_shmem_barrier_all();

    roc_shmem_ctx_long_broadcast(ROC_SHMEM_CTX_DEFAULT, dst, src, nLongs, 1, 0,
                                 0, num_pes, pSync);

    for (i = 0; i < nLongs; i++) {
      /* the root node shouldn't have the result into dst (cf specification).*/
      if (1 != mpe && dst[i] != src[i]) {
        fprintf(stderr, "[%d] dst[%d] %ld != expected %ld\n", mpe, i, dst[i],
                src[i]);
        roc_shmem_global_exit(1);
      } else if (1 == mpe && dst[i] != 0) {
        fprintf(stderr, "[%d] dst[%d] %ld != expected 0\n", mpe, i, dst[i]);
        roc_shmem_global_exit(1);
      }
    }
    roc_shmem_barrier_all();

    roc_shmem_free(dst);
    if (Verbose && mpe == 0)
      fprintf(stderr, "loop %2d Bcast %d, Done.\n", cloop, nBytes);
    nBytes += BCAST_INCR;
  }

  roc_shmem_finalize();

  return 0;
}
