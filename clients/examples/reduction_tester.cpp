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

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template<typename T1, ROC_SHMEM_OP T2>
__global__
void ReductionTest(int loop,
                   int skip,
                   uint64_t *timer,
                   T1 *s_buf,
                   T1 *r_buf,
                   T1 *pWrk,
                   long *pSync,
                   int size,
                   TestType type,
                   ShmemContextType ctx_type)
{
    __shared__ roc_shmem_ctx_t ctx;

    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);

    int n_pes = roc_shmem_n_pes(ctx);

    __syncthreads();

    uint64_t start;
    for (int i = 0; i < loop + skip; i++) {
        if (i == skip && hipThreadIdx_x == 0) {
            start = roc_shmem_timer();
        }
        roc_shmem_wg_to_all<T1, T2>(ctx,
                                    r_buf,
                                    s_buf,
                                    size,
                                    0,
                                    0,
                                    n_pes,
                                    pWrk,
                                    pSync);
        roc_shmem_wg_barrier_all(ctx);
    }

    __syncthreads();

    if (hipThreadIdx_x == 0) {
        timer[hipBlockIdx_x] = roc_shmem_timer() - start;
    }

    roc_shmem_wg_ctx_destroy(ctx);
    roc_shmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template<typename T1, ROC_SHMEM_OP T2>
ReductionTester<T1, T2>::ReductionTester(TesterArguments args,
                                         std::function<void(T1&, T1&)> f1,
                                         std::function<std::pair<bool, std::string>(const T1&, const T1&)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2}
{
    s_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));
    r_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));

    size_t p_wrk_size =
        std::max(args.max_msg_size / 2 + 1, SHMEM_REDUCE_MIN_WRKDATA_SIZE);
    pWrk = (T1 *)roc_shmem_malloc(p_wrk_size * sizeof(T1));

    size_t p_sync_size = SHMEM_REDUCE_SYNC_SIZE;
    pSync = (long *)roc_shmem_malloc(p_sync_size * sizeof(long));

    for (int i = 0; i < p_sync_size; i++) {
        pSync[i] = SHMEM_SYNC_VALUE;
    }
}

template<typename T1, ROC_SHMEM_OP T2>
ReductionTester<T1, T2>::~ReductionTester()
{
    roc_shmem_free(s_buf);
    roc_shmem_free(r_buf);
    roc_shmem_free(pWrk);
    roc_shmem_free(pSync);
}

template<typename T1, ROC_SHMEM_OP T2>
void
ReductionTester<T1, T2>::launchKernel(dim3 gridSize,
                                      dim3 blockSize,
                                      int loop,
                                      uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(ReductionTest<T1, T2>,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       s_buf,
                       r_buf,
                       pWrk,
                       pSync,
                       size,
                       _type,
                       _shmem_context);

    num_msgs = loop + args.skip;
    num_timed_msgs = loop;
}

template<typename T1, ROC_SHMEM_OP T2>
void
ReductionTester<T1, T2>::resetBuffers()
{
    for (int i = 0; i < args.max_msg_size; i++) {
        init_buf(s_buf[i], r_buf[i]);
    }
}

template<typename T1, ROC_SHMEM_OP T2>
void
ReductionTester<T1, T2>::verifyResults(uint64_t size)
{
    int n_pes = roc_shmem_n_pes();
    for (int i = 0; i < size; i++) {
        auto r = verify_buf(r_buf[i], (T1)n_pes);
        if (r.first == false) {
            fprintf(stderr, "Data validation error at idx %d\n", i);
            fprintf(stderr, "%s.\n", r.second.c_str());
            exit(-1);
        }
    }
}

