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

using namespace rocshmem;

/* Declare the template with a generic implementation */
template <typename T>
__device__ void
wg_broadcast(roc_shmem_ctx_t ctx,
             T *dest,
             const T *source,
             int nelem,
             int pe_root,
             int pe_start,
             int log_pe_stride,
             int pe_size,
             long *p_sync)
{
    return;
}

/* Define templates to call ROC_SHMEM */
#define BROADCAST_DEF_GEN(T, TNAME) \
    template <> __device__ void \
    wg_broadcast<T>(roc_shmem_ctx_t ctx, \
                    T *dest, \
                    const T *source, \
                    int nelem, \
                    int pe_root, \
                    int pe_start, \
                    int log_pe_stride, \
                    int pe_size, \
                    long *p_sync) \
    { \
        roc_shmem_ctx_##TNAME##_wg_broadcast(ctx, dest, source, nelem, pe_root, pe_start, \
                                             log_pe_stride, pe_size, p_sync); \
    }

BROADCAST_DEF_GEN(float, float)
BROADCAST_DEF_GEN(double, double)
BROADCAST_DEF_GEN(char, char)
//BROADCAST_DEF_GEN(long double, longdouble)
BROADCAST_DEF_GEN(signed char, schar)
BROADCAST_DEF_GEN(short, short)
BROADCAST_DEF_GEN(int, int)
BROADCAST_DEF_GEN(long, long)
BROADCAST_DEF_GEN(long long, longlong)
BROADCAST_DEF_GEN(unsigned char, uchar)
BROADCAST_DEF_GEN(unsigned short, ushort)
BROADCAST_DEF_GEN(unsigned int, uint)
BROADCAST_DEF_GEN(unsigned long, ulong)
BROADCAST_DEF_GEN(unsigned long long, ulonglong)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template<typename T1>
__global__
void BroadcastTest(int loop,
                   int skip,
                   uint64_t *timer,
                   T1 *source_buf,
                   T1 *dest_buf,
                   long *pSync,
                   int size,
                   ShmemContextType ctx_type)
{
    __shared__ roc_shmem_ctx_t ctx;

    roc_shmem_wg_init();
    roc_shmem_wg_ctx_create(ctx_type, &ctx);

    int n_pes = roc_shmem_ctx_n_pes(ctx);

    __syncthreads();

    uint64_t start;
    for (int i = 0; i < loop; i++) {
        if (i == skip && hipThreadIdx_x == 0) {
            start = roc_shmem_timer();
        }

        wg_broadcast<T1>(ctx,
                         dest_buf,    // T* dest
                         source_buf,  // const T* source
                         size,        // int nelement
                         0,           // int PE_root
                         0,           // int PE_start
                         0,           // int logPE_stride
                         n_pes,       // int PE_size
                         pSync);      // long *pSync
        roc_shmem_ctx_wg_barrier_all(ctx);
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
template<typename T1>
BroadcastTester<T1>::BroadcastTester(TesterArguments args, std::function<void(T1&, T1&)> f1,
    std::function<std::pair<bool, std::string>(const T1&)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2}
{
    source_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));
    dest_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));

    size_t p_sync_size = ROC_SHMEM_BCAST_SYNC_SIZE;
    pSync = (long *)roc_shmem_malloc(p_sync_size * sizeof(long));

    for (int i = 0; i < p_sync_size; i++) {
        pSync[i] = ROC_SHMEM_SYNC_VALUE;
    }
}

template<typename T1>
BroadcastTester<T1>::~BroadcastTester()
{
    roc_shmem_free(source_buf);
    roc_shmem_free(dest_buf);
    roc_shmem_free(pSync);
}

template<typename T1>
void
BroadcastTester<T1>::launchKernel(dim3 gridSize,
                                  dim3 blockSize,
                                  int loop,
                                  uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(BroadcastTest<T1>,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       source_buf,
                       dest_buf,
                       pSync,
                       size,
                       _shmem_context);

    num_msgs = loop + args.skip;
    num_timed_msgs = loop;
}

template<typename T1>
void
BroadcastTester<T1>::resetBuffers()
{
    for (int i = 0; i < args.max_msg_size; i++) {
        init_buf(source_buf[i], dest_buf[i]);
    }
}

template<typename T1>
void
BroadcastTester<T1>::verifyResults(uint64_t size)
{
    for (int i = 0; i < size; i++) {
        auto r = verify_buf(dest_buf[i]);
        if (r.first == false) {
            fprintf(stderr, "Data validation error at idx %d\n", i);
            fprintf(stderr, "%s.\n", r.second.c_str());
            exit(-1);
        }
    }
}
