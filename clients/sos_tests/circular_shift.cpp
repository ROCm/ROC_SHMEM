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

/* circular shift bbb into aaa */

#include <roc_shmem.hpp>

using namespace rocshmem;

int
main(int argc, char* argv[])
{
    int me, neighbor;
    int ret = 0;
    int aaa, *bbb;

    roc_shmem_init(1);

    bbb = (int *) roc_shmem_malloc(sizeof(int));

    *bbb = me = roc_shmem_my_pe();
    neighbor = (me + 1) % roc_shmem_n_pes();

    roc_shmem_barrier_all();

    roc_shmem_int_get( &aaa, bbb, 1, neighbor );

    roc_shmem_barrier_all();

    if (aaa != neighbor ) ret = 1;

    roc_shmem_free(bbb);

    roc_shmem_finalize();

    return ret;
}
