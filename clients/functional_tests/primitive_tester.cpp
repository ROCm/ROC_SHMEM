/******************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "primitive_tester.hpp"

#include <roc_shmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void
PrimitiveTest(int loop,
              int skip,
              uint64_t *timer,
              char *s_buf,
              char *r_buf,
              int size,
              TestType type,
              ShmemContextType ctx_type)
{
    __shared__ roc_shmem_ctx_t ctx;
    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);

    if (hipThreadIdx_x == 0) {
        uint64_t start;

        for (int i = 0; i < loop + skip; i++) {
            if (i == skip)
                start = roc_shmem_timer();

            switch (type) {
                case GetTestType:
                    roc_shmem_ctx_getmem(ctx, r_buf, s_buf, size, 1);
                    break;
                case GetNBITestType:
                    roc_shmem_ctx_getmem_nbi(ctx, r_buf, s_buf, size, 1);
                    break;
                case PutTestType:
                    roc_shmem_ctx_putmem(ctx, r_buf, s_buf, size, 1);
                    break;
                case PutNBITestType:
                    roc_shmem_ctx_putmem_nbi(ctx, r_buf, s_buf, size, 1);
                    break;
                case PTestType:
                    for(int s = 0; s <size; s++ ){
                        char val = s_buf[s];
                        roc_shmem_ctx_char_p(ctx, &r_buf[s], val , 1);
                    }
                    break;
                case GTestType:
                    for(int s = 0; s <size; s++ ){
                        char ret= roc_shmem_ctx_char_g(ctx, &s_buf[s], 1);
                        r_buf[s]=ret;
                    }
                    break;
                default:
                    break;
            }
        }

        roc_shmem_ctx_quiet(ctx);

        timer[hipBlockIdx_x] =  roc_shmem_timer() - start;
    }

    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PrimitiveTester::PrimitiveTester(TesterArguments args)
    : Tester(args)
{
    s_buf = (char *)roc_shmem_malloc(args.max_msg_size * args.wg_size);
    r_buf = (char *)roc_shmem_malloc(args.max_msg_size * args.wg_size);
}

PrimitiveTester::~PrimitiveTester()
{
    roc_shmem_free(s_buf);
    roc_shmem_free(r_buf);
}

void
PrimitiveTester::resetBuffers(uint64_t size)
{
    memset(s_buf, '0', args.max_msg_size * args.wg_size);
    memset(r_buf, '1', args.max_msg_size * args.wg_size);
}

void
PrimitiveTester::launchKernel(dim3 gridSize,
                              dim3 blockSize,
                              int loop,
                              uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(PrimitiveTest,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       s_buf,
                       r_buf,
                       size,
                       _type,
                       _shmem_context);

    num_msgs = (loop + args.skip) * gridSize.x;
    num_timed_msgs = loop ;
}

void
PrimitiveTester::verifyResults(uint64_t size)
{
    int check_id = (_type == GetTestType    ||
                    _type == GetNBITestType ||
                    _type == GTestType)
                    ? 0 : 1;

    if (args.myid == check_id) {
        for (int i = 0; i < size; i++) {
            if (r_buf[i] != '0') {
                fprintf(stderr, "Data validation error at idx %d\n", i);
                fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
                exit(-1);
            }
        }
    }
}
