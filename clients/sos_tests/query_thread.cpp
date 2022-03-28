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

#include <stdio.h>
#include <roc_shmem.hpp>

using namespace rocshmem;

int
main(int argc, char* argv[])
{
    int provided;

    int tl, ret;
    ret = roc_shmem_init_thread(ROC_SHMEM_THREAD_FUNNELED, &tl, 1);

    if (tl < ROC_SHMEM_THREAD_FUNNELED || ret != 0) {
        printf("Init failed (requested thread level %d, got %d, ret %d)\n",
               ROC_SHMEM_THREAD_FUNNELED, tl, ret);
        if (ret == 0) {
            roc_shmem_global_exit(1);
        } else {
            return ret;
        }
    }

    roc_shmem_query_thread(&provided);
    printf("%d: Query result for thread level %d\n", roc_shmem_my_pe(), provided);

    if (provided < ROC_SHMEM_THREAD_FUNNELED) {
        printf("Error: thread support changed to an invalid level after init\n");
        roc_shmem_global_exit(1);
    }

    roc_shmem_finalize();
    return 0;
}
