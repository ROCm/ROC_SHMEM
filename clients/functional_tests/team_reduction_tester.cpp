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
__device__ void wg_team_to_all(roc_shmem_ctx_t ctx, roc_shmem_team_t, T *dest,
                               const T *source, int nreduce) {
  return;
}

/* Define templates to call ROC_SHMEM */
#define TEAM_REDUCTION_DEF_GEN(T, TNAME, Op_API, Op)                      \
  template <>                                                             \
  __device__ void wg_team_to_all<T, Op>(roc_shmem_ctx_t ctx,              \
                                        roc_shmem_team_t team, T * dest,  \
                                        const T *source, int nreduce) {   \
    roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(ctx, team, dest, source, \
                                                 nreduce);                \
  }

#define TEAM_ARITH_REDUCTION_DEF_GEN(T, TNAME)         \
  TEAM_REDUCTION_DEF_GEN(T, TNAME, sum, ROC_SHMEM_SUM) \
  TEAM_REDUCTION_DEF_GEN(T, TNAME, min, ROC_SHMEM_MIN) \
  TEAM_REDUCTION_DEF_GEN(T, TNAME, max, ROC_SHMEM_MAX) \
  TEAM_REDUCTION_DEF_GEN(T, TNAME, prod, ROC_SHMEM_PROD)

#define TEAM_BITWISE_REDUCTION_DEF_GEN(T, TNAME)       \
  TEAM_REDUCTION_DEF_GEN(T, TNAME, or, ROC_SHMEM_OR)   \
  TEAM_REDUCTION_DEF_GEN(T, TNAME, and, ROC_SHMEM_AND) \
  TEAM_REDUCTION_DEF_GEN(T, TNAME, xor, ROC_SHMEM_XOR)

#define TEAM_INT_REDUCTION_DEF_GEN(T, TNAME) \
  TEAM_ARITH_REDUCTION_DEF_GEN(T, TNAME)     \
  TEAM_BITWISE_REDUCTION_DEF_GEN(T, TNAME)

#define TEAM_FLOAT_REDUCTION_DEF_GEN(T, TNAME) \
  TEAM_ARITH_REDUCTION_DEF_GEN(T, TNAME)

TEAM_INT_REDUCTION_DEF_GEN(int, int)
TEAM_INT_REDUCTION_DEF_GEN(short, short)
TEAM_INT_REDUCTION_DEF_GEN(long, long)
TEAM_INT_REDUCTION_DEF_GEN(long long, longlong)
TEAM_FLOAT_REDUCTION_DEF_GEN(float, float)
TEAM_FLOAT_REDUCTION_DEF_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
// FLOAT_REDUCTION_DEF_GEN(long double, longdouble)

roc_shmem_team_t team_reduce_world_dup;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template <typename T1, ROC_SHMEM_OP T2>
__global__ void TeamReductionTest(int loop, int skip, uint64_t *timer,
                                  T1 *s_buf, T1 *r_buf, int size, TestType type,
                                  ShmemContextType ctx_type,
                                  roc_shmem_team_t team) {
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
    wg_team_to_all<T1, T2>(ctx, team, r_buf, s_buf, size);
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
TeamReductionTester<T1, T2>::TeamReductionTester(
    TesterArguments args, std::function<void(T1 &, T1 &)> f1,
    std::function<std::pair<bool, std::string>(const T1 &, const T1 &)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2} {
  s_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));
  r_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));
}

template <typename T1, ROC_SHMEM_OP T2>
TeamReductionTester<T1, T2>::~TeamReductionTester() {
  roc_shmem_free(s_buf);
  roc_shmem_free(r_buf);
}

template <typename T1, ROC_SHMEM_OP T2>
void TeamReductionTester<T1, T2>::preLaunchKernel() {
  int n_pes = roc_shmem_team_n_pes(ROC_SHMEM_TEAM_WORLD);

  team_reduce_world_dup = ROC_SHMEM_TEAM_INVALID;
  roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                               &team_reduce_world_dup);
}

template <typename T1, ROC_SHMEM_OP T2>
void TeamReductionTester<T1, T2>::launchKernel(dim3 gridSize, dim3 blockSize,
                                               int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(TeamReductionTest<T1, T2>), gridSize,
                     blockSize, shared_bytes, stream, loop, args.skip, timer,
                     s_buf, r_buf, size, _type, _shmem_context,
                     team_reduce_world_dup);

  num_msgs = loop + args.skip;
  num_timed_msgs = loop;
}

template <typename T1, ROC_SHMEM_OP T2>
void TeamReductionTester<T1, T2>::postLaunchKernel() {
  roc_shmem_team_destroy(team_reduce_world_dup);
}

template <typename T1, ROC_SHMEM_OP T2>
void TeamReductionTester<T1, T2>::resetBuffers(uint64_t size) {
  for (int i = 0; i < args.max_msg_size; i++) {
    init_buf(s_buf[i], r_buf[i]);
  }
}

template <typename T1, ROC_SHMEM_OP T2>
void TeamReductionTester<T1, T2>::verifyResults(uint64_t size) {
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
