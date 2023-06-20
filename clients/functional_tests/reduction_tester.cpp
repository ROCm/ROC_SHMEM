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
template <typename T, ROC_SHMEM_OP Op>
__device__ void wg_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source,
                          int nreduce, int PE_start, int logPE_stride,
                          int PE_size, T *pWrk, long *pSync) {
  return;
}

/* Define templates to call ROC_SHMEM */
#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op)                              \
  template <>                                                                \
  __device__ void wg_to_all<T, Op>(                                          \
      roc_shmem_ctx_t ctx, T * dest, const T *source, int nreduce,           \
      int PE_start, int logPE_stride, int PE_size, T *pWrk, long *pSync) {   \
    roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(ctx, dest, source, nreduce, \
                                                 PE_start, logPE_stride,     \
                                                 PE_size, pWrk, pSync);      \
  }

#define ARITH_REDUCTION_DEF_GEN(T, TNAME)         \
  REDUCTION_DEF_GEN(T, TNAME, sum, ROC_SHMEM_SUM) \
  REDUCTION_DEF_GEN(T, TNAME, min, ROC_SHMEM_MIN) \
  REDUCTION_DEF_GEN(T, TNAME, max, ROC_SHMEM_MAX) \
  REDUCTION_DEF_GEN(T, TNAME, prod, ROC_SHMEM_PROD)

#define BITWISE_REDUCTION_DEF_GEN(T, TNAME)       \
  REDUCTION_DEF_GEN(T, TNAME, or, ROC_SHMEM_OR)   \
  REDUCTION_DEF_GEN(T, TNAME, and, ROC_SHMEM_AND) \
  REDUCTION_DEF_GEN(T, TNAME, xor, ROC_SHMEM_XOR)

#define INT_REDUCTION_DEF_GEN(T, TNAME) \
  ARITH_REDUCTION_DEF_GEN(T, TNAME)     \
  BITWISE_REDUCTION_DEF_GEN(T, TNAME)

#define FLOAT_REDUCTION_DEF_GEN(T, TNAME) ARITH_REDUCTION_DEF_GEN(T, TNAME)

INT_REDUCTION_DEF_GEN(int, int)
INT_REDUCTION_DEF_GEN(short, short)
INT_REDUCTION_DEF_GEN(long, long)
INT_REDUCTION_DEF_GEN(long long, longlong)
FLOAT_REDUCTION_DEF_GEN(float, float)
FLOAT_REDUCTION_DEF_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_DEF_GEN(long double, longdouble)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1, ROC_SHMEM_OP T2>
__global__ void ReductionTest(int loop, int skip, uint64_t *timer, T1 *s_buf,
                              T1 *r_buf, T1 *pWrk, long *pSync, int size,
                              TestType type, ShmemContextType ctx_type) {
  __shared__ roc_shmem_ctx_t ctx;

  roc_shmem_wg_init();
  roc_shmem_wg_ctx_create(ctx_type, &ctx);

  int n_pes = roc_shmem_ctx_n_pes(ctx);

  __syncthreads();

  uint64_t start;
  for (int i = 0; i < loop + skip; i++) {
    if (i == skip && hipThreadIdx_x == 0) {
      start = roc_shmem_timer();
    }
    wg_to_all<T1, T2>(ctx, r_buf, s_buf, size, 0, 0, n_pes, pWrk, pSync);
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
template <typename T1, ROC_SHMEM_OP T2>
ReductionTester<T1, T2>::ReductionTester(
    TesterArguments args, std::function<void(T1 &, T1 &)> f1,
    std::function<std::pair<bool, std::string>(const T1 &, const T1 &)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2} {
  s_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));
  r_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));

  size_t p_wrk_size =
      std::max(args.max_msg_size / 2 + 1, ROC_SHMEM_REDUCE_MIN_WRKDATA_SIZE);
  pWrk = (T1 *)roc_shmem_malloc(p_wrk_size * sizeof(T1));

  size_t p_sync_size = ROC_SHMEM_REDUCE_SYNC_SIZE;
  pSync = (long *)roc_shmem_malloc(p_sync_size * sizeof(long));

  for (int i = 0; i < p_sync_size; i++) {
    pSync[i] = ROC_SHMEM_SYNC_VALUE;
  }
}

template <typename T1, ROC_SHMEM_OP T2>
ReductionTester<T1, T2>::~ReductionTester() {
  roc_shmem_free(s_buf);
  roc_shmem_free(r_buf);
  roc_shmem_free(pWrk);
  roc_shmem_free(pSync);
}

template <typename T1, ROC_SHMEM_OP T2>
void ReductionTester<T1, T2>::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(ReductionTest<T1, T2>), gridSize,
                     blockSize, shared_bytes, stream, loop, args.skip, timer,
                     s_buf, r_buf, pWrk, pSync, size, _type, _shmem_context);

  num_msgs = loop + args.skip;
  num_timed_msgs = loop;
}

template <typename T1, ROC_SHMEM_OP T2>
void ReductionTester<T1, T2>::resetBuffers(uint64_t size) {
  for (int i = 0; i < args.max_msg_size; i++) {
    init_buf(s_buf[i], r_buf[i]);
  }
}

template <typename T1, ROC_SHMEM_OP T2>
void ReductionTester<T1, T2>::verifyResults(uint64_t size) {
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
