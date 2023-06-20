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

/* Non-Blocking Get Test
 * Tom St. John <tom.st.john@intel.com>
 * January, 2016
 *
 * PE 0 uses a non-blocking get to copy an array from
 * every remote PE.
 */

#include <stdio.h>
#include <string.h>

#include <roc_shmem.hpp>

using namespace rocshmem;

static long target[10];

int main(int argc, char *argv[]) {
  int i, j, num_pes;
  int failed = 0;
  long *source;

  roc_shmem_init();

  source = (long *)roc_shmem_malloc(10 * sizeof(long));
  for (i = 0; i < 10; i++) {
    source[i] = i + 1;
  }

  roc_shmem_barrier_all();

  if (roc_shmem_my_pe() == 0) {
    num_pes = roc_shmem_n_pes();

    for (j = 0; j < num_pes; j++) {
      memset(target, 0, sizeof(long) * 10);
      roc_shmem_long_get_nbi(target, source, 10, j);
      roc_shmem_quiet();

      for (i = 0; i < 10; i++) {
        if (source[i] != target[i]) {
          fprintf(stderr,
                  "[%d] get_nbi from PE %d: target[%d] = %ld, expected %ld\n",
                  roc_shmem_my_pe(), j, i, target[i], source[i]);
          failed = 1;
        }
      }

      if (failed) roc_shmem_global_exit(1);
    }
  }

  roc_shmem_free(source);

  roc_shmem_finalize();

  return 0;
}
