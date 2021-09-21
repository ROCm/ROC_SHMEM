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

#include "primitive_amo_tester.hpp"

#include <roc_shmem.hpp>

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void
PrimitiveAMOTest(int loop,
                 int skip,
                 uint64_t *timer,
                 char *r_buf,
                 int64_t *ret_val,
                 TestType type,
                 ShmemContextType ctx_type)
{
    __shared__ roc_shmem_ctx_t ctx;
    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);

    if (hipThreadIdx_x == 0) {
        uint64_t start;
        int64_t ret;
        int64_t cond = 0;

        for(int i = 0; i < loop+skip; i++) {
            if(i == skip)
                start = roc_shmem_timer();

            switch (type) {
                case AMO_FAddTestType:
                    ret = roc_shmem_ctx_int64_atomic_fetch_add(
                                ctx, (int64_t*)r_buf, 2, 1);
                    break;
                case AMO_FIncTestType:
                    ret = roc_shmem_ctx_int64_atomic_fetch_inc(
                                ctx, (int64_t*)r_buf, 1);
                    break;
                case AMO_FetchTestType:
                    ret = roc_shmem_ctx_int64_atomic_fetch(
                                ctx, (int64_t*)r_buf, 1);
                    break;
                case AMO_FCswapTestType:
                    ret = roc_shmem_ctx_int64_atomic_compare_swap(
                                ctx, (int64_t*)r_buf, cond, (int64_t)i, 1);
                    cond = i;
                    break;
                case AMO_AddTestType:
                    roc_shmem_ctx_int64_atomic_add(
                            ctx, (int64_t*)r_buf, 2, 1);
                    break;
                case AMO_IncTestType:
                    roc_shmem_ctx_int64_atomic_inc(
                            ctx, (int64_t*)r_buf, 1);
                    break;
                default:
                    break;
            }
        }

        roc_shmem_ctx_quiet(ctx);

        timer[hipBlockIdx_x] =  roc_shmem_timer() - start;

        *ret_val = ret;

        // do get to check the result for no-fetch ops
        roc_shmem_ctx_getmem(ctx, r_buf, r_buf, sizeof(int64_t), 1);
    }

    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PrimitiveAMOTester::PrimitiveAMOTester(TesterArguments args)
    : Tester(args)
{
    hipMalloc((void**)&_ret_val, args.max_msg_size );
    r_buf = (char *)roc_shmem_malloc(args.max_msg_size);
}

PrimitiveAMOTester::~PrimitiveAMOTester()
{
    roc_shmem_free(r_buf);
    hipFree(_ret_val);
}

void
PrimitiveAMOTester::resetBuffers()
{
    memset(r_buf, 0, args.max_msg_size );
    memset(_ret_val, 0, args.max_msg_size);
}

void
PrimitiveAMOTester::launchKernel(dim3 gridsize,
                                 dim3 blocksize,
                                 int loop,
                                 uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(PrimitiveAMOTest,
                       gridsize,
                       blocksize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       r_buf,
                       _ret_val,
                       _type,
                       _shmem_context);

    num_msgs = (loop + args.skip) * gridsize.x;
    num_timed_msgs = loop * gridsize.x;
}

void
PrimitiveAMOTester::verifyResults(uint64_t size)
{
    if (args.myid == 0) {
        int64_t expected_val = 0;

        switch (_type) {
            case AMO_FAddTestType:
                expected_val = 2 * (num_msgs - 1);
                break;
            case AMO_FIncTestType:
                expected_val = num_msgs - 1;
                break;
            case AMO_AddTestType:
                expected_val = 2 * num_msgs;
                break;
            case AMO_IncTestType:
                expected_val = num_msgs;
                break;
            case AMO_FCswapTestType:
                expected_val = num_msgs - 2;
                break;
            default:
                break;
        }

        int fetch_op = (_type == AMO_FAddTestType  ||
                        _type == AMO_FIncTestType  ||
                        _type == AMO_FetchTestType ||
                        _type == AMO_FCswapTestType) ? 1 : 0;

        if (fetch_op == 1) {
            if (*_ret_val != expected_val) {
                fprintf(stderr, "data validation error\n");
                fprintf(stderr, "got %ld, expected %ld\n",
                        *_ret_val, expected_val);
                exit(-1);
            }
        } else {
            if (((int64_t*)r_buf)[0] != expected_val) {
                fprintf(stderr, "data validation error\n");
                fprintf(stderr, "got %ld, expected %ld\n",
                    ((int64_t*)r_buf)[0], expected_val);
                exit(-1);
            }
        }
    }
}
