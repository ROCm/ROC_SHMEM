/******************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef _TESTS_H_
#define _TESTS_H_

#include <algorithm>

#include <roc_shmem.hpp>

enum TestType
{
    GetTestType,
    GetNBITestType,
    PutTestType,
    PutNBITestType,
    GetSwarmTestType,
    ReductionTestType,
    AMO_FAddTestType,
    AMO_FIncTestType,
    AMO_FetchTestType,
    AMO_FCswapTestType,
    AMO_AddTestType,
    AMO_IncTestType,
    AMO_CswapTestType,
    InitTestType,
    PingPongTestType
};

__global__ void
PrimitiveTest(int loop, int skip, uint64_t *timer, char *s_buf, char *r_buf,
              int size, TestType type)
{
    uint64_t start;
    __shared__ roc_shmem_ctx_t ctx;
    roc_shmem_ctx_create(0, &ctx);

    if (hipThreadIdx_x == 0) {


        for(int i = 0; i < loop+skip; i++) {
            if(i == skip)
                start = roc_shmem_timer();
            switch (type) {
                case GetTestType:
                    roc_shmem_getmem(ctx, r_buf, s_buf, size, 1);
                    break;
                case GetNBITestType:
                    roc_shmem_getmem_nbi(ctx, r_buf, s_buf, size, 1);
                    break;
                case PutTestType:
                    roc_shmem_putmem(ctx, r_buf, s_buf, size, 1);
                    break;
                case PutNBITestType:
                    roc_shmem_putmem_nbi(ctx, r_buf, s_buf, size, 1);
                    break;
                default:
                    break;
            }
        }

        roc_shmem_quiet(ctx);

        timer[hipBlockIdx_x] =  roc_shmem_timer() - start;
    }
    roc_shmem_ctx_destroy(ctx);
}
__global__ void
PingPongTest(int loop, int skip, uint64_t *timer, int *r_buf)
{
    uint64_t start;
    __shared__ roc_shmem_ctx_t ctx;
    roc_shmem_ctx_create(0, &ctx);
    int pe = roc_shmem_my_pe(ctx);

    if (hipThreadIdx_x == 0) {
        for(int i=0; i<loop+skip; i++){
            if(i == skip)
                start = roc_shmem_timer();
            if(pe ==0){
                roc_shmem_p(ctx, r_buf, i+1, 1);
                roc_shmem_wait_until(ctx, r_buf, ROC_SHMEM_CMP_EQ, i+1);
            }else{
                roc_shmem_wait_until(ctx, r_buf, ROC_SHMEM_CMP_EQ, i+1);
                roc_shmem_p(ctx, r_buf, i+1, 0);
            }
        }
        timer[hipBlockIdx_x] =  roc_shmem_timer() - start;
    }
    roc_shmem_ctx_destroy(ctx);
}
__global__ void
PrimitiveAMOTest(int loop, int skip, uint64_t *timer, char *r_buf,
                 int64_t *ret_val, TestType type)
{

    uint64_t start;
    int64_t ret;
    int64_t cond =0;
    __shared__ roc_shmem_ctx_t ctx;
    roc_shmem_ctx_create(0, &ctx);

    if (hipThreadIdx_x == 0) {


        for(int i = 0; i < loop+skip; i++) {
            if(i == skip)
                start = roc_shmem_timer();
            switch (type) {
                case AMO_FAddTestType:
                    ret = roc_shmem_atomic_fetch_add<int64_t>(ctx,
                                                       (int64_t*)r_buf, 2, 1);
                    break;
                case AMO_FIncTestType:
                    ret = roc_shmem_atomic_fetch_inc<int64_t>(ctx,
                                                            (int64_t*)r_buf, 1);
                    break;
                case AMO_FetchTestType:
                    ret = roc_shmem_atomic_fetch<int64_t>(ctx,
                                                            (int64_t*)r_buf, 1);
                    break;
                case AMO_FCswapTestType:
                    ret = roc_shmem_atomic_fetch_cswap<int64_t>(ctx,
                                                            (int64_t*)r_buf,
                                                            cond, (int64_t)i, 1);
                    cond = i;
                    break;
                case AMO_AddTestType:
                    roc_shmem_atomic_add<int64_t>(ctx, (int64_t*)r_buf, 2, 1);
                    break;
                case AMO_IncTestType:
                    roc_shmem_atomic_inc<int64_t>(ctx, (int64_t*)r_buf, 1);
                    break;
                case AMO_CswapTestType:
                    roc_shmem_atomic_cswap<int64_t>(ctx, (int64_t*)r_buf,
                                                        cond, (int64_t)i, 1);
                    cond = i;
                    break;
                default:
                    break;
            }
        }
        roc_shmem_quiet(ctx);

        timer[hipBlockIdx_x] =  roc_shmem_timer() - start;
        *ret_val = ret;
        // do get to check the result for no-fetch ops
        roc_shmem_getmem(ctx, r_buf, r_buf, sizeof(int64_t), 1);
    }

    roc_shmem_ctx_destroy(ctx);
}

__global__ void
GetSwarmTest(int loop, int skip, uint64_t *timer, char *s_buf, char *r_buf,
             int size)
{
    uint64_t start = 0;
    __shared__ roc_shmem_ctx_t ctx;

    roc_shmem_ctx_create(0, &ctx);

    __syncthreads();

    int index = hipThreadIdx_x * size;

    for(int i = 0; i < loop+skip; i++)
        if(i == skip)
            start = roc_shmem_timer();
        roc_shmem_getmem(ctx, &r_buf[index], &s_buf[index], size, 1);

    __syncthreads();

    atomicAdd((unsigned long long *) &timer[hipBlockIdx_x],
              roc_shmem_timer() - start);

    roc_shmem_ctx_destroy(ctx);
}

__global__ void
ReductionTest(int loop, int skip, uint64_t *timer, float *s_buf, float *r_buf,
              float *pWrk, long *pSync, int size)
{
    uint64_t start;
    __shared__ roc_shmem_ctx_t ctx;

    roc_shmem_ctx_create(0, &ctx);

    if (hipThreadIdx_x == 0) {
        start = roc_shmem_timer();

    }
    __syncthreads();
    for (int i = 0; i < loop; i++) {
        roc_shmem_to_all<float, ROC_SHMEM_SUM>(ctx, r_buf, s_buf, size, 0, 0,
                                               2, pWrk, pSync);
    }
    __syncthreads();

    if (hipThreadIdx_x == 0) {
        timer[hipBlockIdx_x] = roc_shmem_timer() - start;
    }

    roc_shmem_ctx_destroy(ctx);
}

class Tester
{
  public:
    virtual void initBuffers(unsigned long size, int wg_size) = 0;

    virtual void resetBuffers() = 0;

    virtual void launchKernel(dim3 gridSize, dim3 blockSize,
                              hipStream_t stream, int loop, int skip,
                              uint64_t *timer, uint64_t size) = 0;

    virtual void verifyResults(int my_id, uint64_t size) = 0;

    virtual int numMsgs() = 0;

    virtual void freeBuffers() = 0;

    virtual ~Tester() {}

    static Tester* Create(TestType type, int rank);
};

class PrimitiveTester : public Tester
{
  public:

    explicit PrimitiveTester(TestType type)
        : Tester(), _type(type) {}

    virtual void
    initBuffers(unsigned long max_size, int wg_size)
    {
        _max_size = max_size;
        _wg_size = wg_size;
        s_buf = (char *)roc_shmem_malloc(max_size * wg_size);
        r_buf = (char *)roc_shmem_malloc(max_size * wg_size);
    }

    virtual void
    resetBuffers()
    {
        memset(s_buf, '0', _max_size * _wg_size);
        memset(r_buf, '1', _max_size * _wg_size);
    }

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 int skip, uint64_t *timer, uint64_t size)
    {
        size_t shared_bytes;
        roc_shmem_dynamic_shared(&shared_bytes);
        hipLaunchKernelGGL(PrimitiveTest, gridSize,
                           blockSize, shared_bytes, stream,
                           loop, skip, timer, s_buf, r_buf, size, _type);
        num_msgs = (loop+skip) * gridSize.x;
    }

    virtual void
    verifyResults(int my_id, uint64_t size)
    {
        int check_id = (_type == GetTestType || _type == GetNBITestType)
             ? 0 : 1;

        if (my_id == check_id) {
            for (int i = 0; i < size; i++) {
                if (r_buf[i] != '0') {
                    fprintf(stderr, "Data validation error at idx %d\n", i);
                    fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
                    exit(-1);
                }
            }
        }
    }

    virtual int numMsgs() { return num_msgs; }

    virtual void
    freeBuffers()
    {
        roc_shmem_free(s_buf);
        roc_shmem_free(r_buf);
    }

  protected:
    char *s_buf = nullptr;
    char *r_buf = nullptr;
    unsigned long long _max_size = 0;
    int _wg_size = 0;
    TestType _type = InitTestType;
    int num_msgs = 0;
};

class PrimitiveAMOTester : public Tester
{
  public:

    PrimitiveAMOTester(TestType type)
        : Tester(), _type(type){}

    virtual void
    initBuffers(unsigned long max_size, int wg_size)
    {
        _max_size = max_size;
        hipMalloc((void**)&_ret_val, max_size );
        r_buf = (char *)roc_shmem_malloc(max_size);
    }

    virtual void
    resetBuffers()
    {
        memset(r_buf, 0, _max_size );
        memset(_ret_val, 0, _max_size);
    }

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 int skip, uint64_t *timer, uint64_t size)
    {
        size_t shared_bytes;
        roc_shmem_dynamic_shared(&shared_bytes);
        hipLaunchKernelGGL(PrimitiveAMOTest, gridSize,
                           blockSize, shared_bytes, stream,
                           loop, skip, timer, r_buf, _ret_val, _type);
        num_msgs = (loop+skip) * gridSize.x;
    }

    virtual void
    verifyResults(int my_id, uint64_t size)
    {
        if(my_id ==0){
            int64_t expected_val = 0;
            int fetch_op = (_type == AMO_FAddTestType  ||
                        _type == AMO_FIncTestType  ||
                        _type == AMO_FetchTestType ||
                        _type == AMO_FCswapTestType) ? 1 : 0;
            switch (_type) {
                case AMO_FAddTestType:
                    expected_val = 2* (num_msgs-1);
                    break;
                case AMO_FIncTestType:
                    expected_val =(num_msgs-1);
                    break;
                case AMO_AddTestType:
                    expected_val =2*(num_msgs);
                    break;
                case AMO_IncTestType:
                    expected_val =(num_msgs);
                    break;
                case AMO_FCswapTestType:
                    expected_val = (num_msgs -2);
                    break;
                case AMO_CswapTestType:
                    expected_val = (num_msgs -1);
                    break;
                default:
                    break;
            }
            if (fetch_op ==1) {
                if (*_ret_val != expected_val) {
                    fprintf(stderr, "Data validation error\n");
                    fprintf(stderr, "Got %lld, Expected %lld\n",
                            *_ret_val, expected_val);
                    exit(-1);
                }
            }else{
                if (((int64_t*)r_buf)[0] != expected_val) {
                    fprintf(stderr, "Data validation error\n");
                    fprintf(stderr, "Got %lld, Expected %lld\n",
                        ((int64_t*)r_buf)[0], expected_val);
                    exit(-1);
                }
            }
        }
    }

    virtual int numMsgs() { return num_msgs; }

    virtual void
    freeBuffers()
    {
        roc_shmem_free(r_buf);
        hipFree(_ret_val);
    }

  protected:
    char *r_buf;
    TestType _type;
    int num_msgs;
    int _max_size;
    int64_t *_ret_val;
};

class PingPongTester : public Tester
{
  public:

    PingPongTester(TestType type)
        : Tester(){}

    virtual void
    initBuffers(unsigned long max_size, int wg_size)
    {
        r_buf = (int *)roc_shmem_malloc(sizeof(int));
    }

    virtual void
    resetBuffers()
    {
        memset(r_buf, 0, sizeof(int));
    }

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 int skip, uint64_t *timer, uint64_t size)
    {
        size_t shared_bytes;
        roc_shmem_dynamic_shared(&shared_bytes);
        hipLaunchKernelGGL(PingPongTest, gridSize,
                           blockSize, shared_bytes, stream,
                           loop, skip, timer, r_buf);
        num_msgs = (loop+skip) * gridSize.x;
    }
     virtual void
    verifyResults(int my_id, uint64_t size)
    {
    }

    virtual int numMsgs() { return num_msgs; }

    virtual void
    freeBuffers()
    {
        roc_shmem_free(r_buf);
    }

  protected:
    int *r_buf;
    int num_msgs;
};


class GetSwarmTester : public PrimitiveTester
{
  public:
    explicit GetSwarmTester(TestType type)
        : PrimitiveTester(type) {}

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 int skip, uint64_t *timer, uint64_t size)
    {
        size_t shared_bytes;
        roc_shmem_dynamic_shared(&shared_bytes);
        hipLaunchKernelGGL(GetSwarmTest, gridSize,
                           blockSize, shared_bytes, stream,
                           loop, skip, timer, s_buf, r_buf, size);
        num_msgs = (loop+skip) * gridSize.x * blockSize.x;
    }

    virtual void
    verifyResults(int my_id, uint64_t size)
    {
        if (my_id == 0) {
            for (int i = 0; i < size * _wg_size; i++) {
                if (r_buf[i] != '0') {
                    fprintf(stderr, "Data validation error at idx %d\n", i);
                    fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
                    exit(-1);
                }
            }
        }
    }
};

class ReductionTester : public Tester
{
  public:
    virtual void
    initBuffers(unsigned long max_size, int wg_size)
    {
        _max_size = max_size;
        s_buf = (float *)roc_shmem_malloc(max_size * sizeof(float));
        r_buf = (float *)roc_shmem_malloc(max_size * sizeof(float));

        size_t pWrkSize =
            std::max(max_size / 2 + 1, SHMEM_REDUCE_MIN_WRKDATA_SIZE);

        size_t pSyncSize = SHMEM_REDUCE_SYNC_SIZE;

        pWrk = (float *) roc_shmem_malloc(pWrkSize * sizeof(float));
        pSync = (long *) roc_shmem_malloc(pSyncSize * sizeof(long));

        for (int i = 0; i < pSyncSize; i++)
            pSync[i] = SHMEM_SYNC_VALUE;
    }

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 int skip, uint64_t *timer, uint64_t size)
    {
        size_t shared_bytes;
        roc_shmem_dynamic_shared(&shared_bytes);
        hipLaunchKernelGGL(ReductionTest, gridSize,
                           blockSize, shared_bytes, stream,
                           loop, skip, timer, s_buf, r_buf, pWrk, pSync, size);
    }

    virtual void
    freeBuffers()
    {
        roc_shmem_free(s_buf);
        roc_shmem_free(r_buf);
        roc_shmem_free(pWrk);
        roc_shmem_free(pSync);
    }

    virtual void
    resetBuffers()
    {
        for (int i = 0; i < _max_size; i++) {
            s_buf[i] = 1.0;
            r_buf[i] = 1.0;
        }
    }

    virtual void
    verifyResults(int my_id, uint64_t size)
    {
        for (int i = 0; i < size; i++) {
            if (r_buf[i] != 2.0) {
                fprintf(stderr, "Data validation error at idx %d\n", i);
                fprintf(stderr, "Got %f, Expected %f\n", r_buf[i], 2.0);
                exit(-1);
            }
        }
    }

    virtual int numMsgs() { return 0; }

  protected:
    uint64_t _max_size;
    float *s_buf;
    float *r_buf;
    float *pWrk;
    long *pSync;

};

Tester*
Tester::Create(TestType type, int rank)
{
    if (rank == 0)
        std::cout << "*** Creating Test: ";

    switch (type) {
        case InitTestType:
            if (rank == 0)
                std::cout << "Init ***" << std::endl;
            return new PrimitiveTester(type);
        case GetTestType:       // Get Test
            if (rank == 0)
                std::cout << "Blocking Gets***" << std::endl;
            return new PrimitiveTester(type);
        case GetNBITestType:    // Get NBI Test
            if (rank == 0)
                std::cout << "Non-Blocking Gets***" << std::endl;
            return new PrimitiveTester(type);
        case PutTestType:       // Put Test
            if (rank == 0)
                std::cout << "Blocking Puts***" << std::endl;
            return new PrimitiveTester(type);
        case PutNBITestType:    // Put NBI Test
            if (rank == 0)
                std::cout << "Non-Blocking Puts***" << std::endl;
            return new PrimitiveTester(type);
        case GetSwarmTestType:  // Get Swarm Test
            if (rank == 0)
                std::cout << "Get Swarm***" << std::endl;
            return new GetSwarmTester(type);
        case ReductionTestType: // Reduction Test
            if (rank == 0)
                std::cout << "All-to-All Reduction***" << std::endl;
            return new ReductionTester();
        case AMO_FAddTestType:       // AMO_FAdd Test
            if (rank == 0)
                std::cout << "AMO Fetch_Add***" << std::endl;
            return new PrimitiveAMOTester(type);
        case AMO_FIncTestType:       // AMO_FInc Test
            if (rank == 0)
                std::cout << "AMO Fetch_Inc***" << std::endl;
            return new PrimitiveAMOTester(type);
        case AMO_FetchTestType:       // AMO_Fetch Test
            if (rank == 0)
                std::cout << "AMO Fetch***" << std::endl;
            return new PrimitiveAMOTester(type);
        case AMO_FCswapTestType:       // AMO_FCswap Test
            if (rank == 0)
                std::cout << "AMO Fetch_CSWAP***" << std::endl;
            return new PrimitiveAMOTester(type);
        case AMO_AddTestType:       // AMO_Add Test
            if (rank == 0)
                std::cout << "AMO Add***" << std::endl;
            return new PrimitiveAMOTester(type);
        case AMO_IncTestType:       // AMO_Inc Test
            if (rank == 0)
                std::cout << "AMO Inc***" << std::endl;
            return new PrimitiveAMOTester(type);
        case AMO_CswapTestType:       // AMO_Cswap Test
            if (rank == 0)
                std::cout << "AMO Cswap***" << std::endl;
            return new PrimitiveAMOTester(type);
        case PingPongTestType:       // PingPong Test
            if (rank == 0)
                std::cout << "PingPong***" << std::endl;
            return new PingPongTester(type);

        default:
            if (rank == 0)
                std::cout << "Unknown***" << std::endl;
            return new PrimitiveTester(type);
    }
}

#endif /* _TESTS_H */
