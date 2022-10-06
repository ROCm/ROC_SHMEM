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

roc_shmem_team_t team_fcollect_world_dup;

/* Declare the template with a generic implementation */
template <typename T>
__device__ void
wg_fcollect(roc_shmem_ctx_t ctx,
                  roc_shmem_team_t team,
                  T *dest,
                  const T *source,
                  int nelems)
{
    return;
}

/* Define templates to call ROC_SHMEM */
#define FCOLLECT_DEF_GEN(T, TNAME) \
    template <> __device__ void \
    wg_fcollect<T>(roc_shmem_ctx_t ctx, \
                         roc_shmem_team_t team, \
                         T *dest, \
                         const T *source, \
                         int nelem) \
    { \
        roc_shmem_ctx_##TNAME##_wg_fcollect(ctx, team, dest, source, nelem); \
    }

FCOLLECT_DEF_GEN(float, float)
FCOLLECT_DEF_GEN(double, double)
FCOLLECT_DEF_GEN(char, char)
//FCOLLECT_DEF_GEN(long double, longdouble)
FCOLLECT_DEF_GEN(signed char, schar)
FCOLLECT_DEF_GEN(short, short)
FCOLLECT_DEF_GEN(int, int)
FCOLLECT_DEF_GEN(long, long)
FCOLLECT_DEF_GEN(long long, longlong)
FCOLLECT_DEF_GEN(unsigned char, uchar)
FCOLLECT_DEF_GEN(unsigned short, ushort)
FCOLLECT_DEF_GEN(unsigned int, uint)
FCOLLECT_DEF_GEN(unsigned long, ulong)
FCOLLECT_DEF_GEN(unsigned long long, ulonglong)

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
template<typename T1>
__global__
void FcollectTest(int loop,
                  int skip,
                  uint64_t *timer,
                  T1 *source_buf,
                  T1 *dest_buf,
                  int size,
                  ShmemContextType ctx_type,
                  roc_shmem_team_t team)
{
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
        wg_fcollect<T1>(ctx,
                        team,
                        dest_buf,    // T* dest
                        source_buf,  // const T* source
                        size);       // int nelement
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
FcollectTester<T1>::FcollectTester(TesterArguments args, std::function<void(T1&, T1&)> f1,
    std::function<std::pair<bool, std::string>(const T1&, T1)> f2)
    : Tester(args), init_buf{f1}, verify_buf{f2}
{
    int n_pes = roc_shmem_team_n_pes(ROC_SHMEM_TEAM_WORLD);
    source_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1));
    dest_buf = (T1 *)roc_shmem_malloc(args.max_msg_size * sizeof(T1) * n_pes);
}

template<typename T1>
FcollectTester<T1>::~FcollectTester()
{
    roc_shmem_free(source_buf);
    roc_shmem_free(dest_buf);
}

template<typename T1>
void
FcollectTester<T1>::preLaunchKernel()
{
    int n_pes = roc_shmem_team_n_pes(ROC_SHMEM_TEAM_WORLD);
    bw_factor = sizeof(T1) * n_pes;

    team_fcollect_world_dup = ROC_SHMEM_TEAM_INVALID;
    roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD,
                                 0,
                                 1,
                                 n_pes,
                                 nullptr,
                                 0,
                                 &team_fcollect_world_dup);
}

template<typename T1>
void
FcollectTester<T1>::launchKernel(dim3 gridSize,
                                      dim3 blockSize,
                                      int loop,
                                      uint64_t size)
{
    size_t shared_bytes;
    roc_shmem_dynamic_shared(&shared_bytes);

    hipLaunchKernelGGL(FcollectTest<T1>,
                       gridSize,
                       blockSize,
                       shared_bytes,
                       stream,
                       loop,
                       args.skip,
                       timer,
                       source_buf,
                       dest_buf,
                       size,
                       _shmem_context,
                       team_fcollect_world_dup);

    num_msgs = loop + args.skip;
    num_timed_msgs = loop;
}

template<typename T1>
void
FcollectTester<T1>::postLaunchKernel()
{
    roc_shmem_team_destroy(team_fcollect_world_dup);
}

template<typename T1>
void
FcollectTester<T1>::resetBuffers(uint64_t size)
{
    int n_pes = roc_shmem_team_n_pes(ROC_SHMEM_TEAM_WORLD);
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < size; j++) {
            // Note: This is redundant work, 
            // source is being reinitialized multiple times
            init_buf(source_buf[j], dest_buf[i * size + j]);
        }
    }
}

template<typename T1>
void
FcollectTester<T1>::verifyResults(uint64_t size)
{
    int n_pes = roc_shmem_team_n_pes(ROC_SHMEM_TEAM_WORLD);
    for (int i = 0; i < n_pes; i++) {
        for (int j = 0; j < size; j++) {
            auto r = verify_buf(dest_buf[i * size + j], i);
            if (r.first == false) {
                fprintf(stderr, "Data validation error at idx %d\n", j);
                fprintf(stderr, "%s.\n", r.second.c_str());
                //exit(-1);
                return;
            }
        }
    }
}
