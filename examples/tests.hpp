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

#include "ro_net.hpp"

enum TestType { GetTestType, GetNBITestType, PutTestType, PutNBITestType,
                GetSwarmTestType, ReductionTestType, InitTestType };

__global__ void
PrimitiveTest(ro_net_handle_t *handle, int loop, uint64_t *timer,
        char *s_buf, char *r_buf, int size, TestType type)
{
    int64_t start;
    __shared__ ro_net_wg_handle_t *wg_handle;
    __shared__ ro_net_ctx_t ctx;

    if (hipThreadIdx_x == 0) {
        ro_net_init(handle, &wg_handle);
        ro_net_ctx_create(0, &ctx, wg_handle);

        start = ro_net_timer(wg_handle);
        for(int i = 0; i < loop; i++) {
            switch (type) {
                case GetTestType:
                    ro_net_getmem(ctx, r_buf, s_buf, size, 1);
                    break;
                case GetNBITestType:
                    ro_net_getmem_nbi(ctx, r_buf, s_buf, size, 1);
                    break;
                case PutTestType:
                    ro_net_putmem(ctx, r_buf, s_buf, size, 1);
                    break;
                case PutNBITestType:
                    ro_net_putmem_nbi(ctx, r_buf, s_buf, size, 1);
                    break;
                default:
                    break;
            }
        }

        ro_net_finalize(handle, wg_handle);
        timer[hipBlockIdx_x] =  ro_net_timer(wg_handle) - start;
    }
}

__global__ void
GetSwarmTest(ro_net_handle_t *handle, int loop, uint64_t *timer,
                char *s_buf, char *r_buf, int size)
{
    int64_t start;
    __shared__ ro_net_wg_handle_t *wg_handle;
    __shared__ ro_net_ctx_t ctx;

    if (hipThreadIdx_x == 0) {
        ro_net_init(handle, &wg_handle);
        ro_net_ctx_create(0, &ctx, wg_handle);
    }

    __syncthreads();

    start = ro_net_timer(wg_handle);
    for(int i = 0; i < loop; i++)
        ro_net_getmem(ctx, r_buf, s_buf, size, 1);

    __syncthreads();

    atomicAdd((unsigned long long *) &timer[hipBlockIdx_x],
              ro_net_timer(wg_handle) - start);

    if (hipThreadIdx_x == 0)
        ro_net_finalize(handle, wg_handle);
}

__global__ void
ReductionTest(ro_net_handle_t *handle, int loop, uint64_t *timer,
              float *s_buf, float *r_buf, float *pWrk, long *pSync, int size)
{
    int64_t start;
    __shared__ ro_net_wg_handle_t *wg_handle;
    __shared__ ro_net_ctx_t ctx;

    if (hipThreadIdx_x == 0) {
        ro_net_init(handle, &wg_handle);
        ro_net_ctx_create(0, &ctx, wg_handle);

        start = ro_net_timer(wg_handle);
        for(int i = 0; i < loop; i++) {
            ro_net_float_sum_to_all(r_buf, s_buf, size, 0, 0, 2, pWrk,
                                      pSync, wg_handle);
        }

        ro_net_finalize(handle, wg_handle);
        timer[hipBlockIdx_x] =  ro_net_timer(wg_handle) - start;
    }
}

class Tester
{
  public:
    virtual void initBuffers(unsigned long long size) = 0;

    virtual void resetBuffers() = 0;

    virtual void launchKernel(dim3 gridSize, dim3 blockSize,
                              hipStream_t stream, int loop, uint64_t *timer,
                              uint64_t size, ro_net_handle_t *handle) = 0;

    virtual void verifyResults(int my_id, uint64_t size) = 0;

    virtual int numMsgs() = 0;

    virtual void freeBuffers() = 0;

    virtual ~Tester() {}

    static Tester* Create(TestType type);
};

class PrimitiveTester : public Tester
{
  public:

    PrimitiveTester(TestType type)
        : Tester(), _type(type) {}

    virtual void
    initBuffers(unsigned long long max_size)
    {
        _max_size = max_size;
	    s_buf = (char *)ro_net_malloc(max_size);
	    r_buf = (char *)ro_net_malloc(max_size);
    }

    virtual void
    resetBuffers()
    {
        memset(s_buf, '0', _max_size);
        memset(r_buf, '1', _max_size);
    }

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 uint64_t *timer, uint64_t size, ro_net_handle_t *handle)
    {
        hipLaunchKernelGGL(PrimitiveTest, dim3(gridSize),
                           dim3(blockSize), 0, stream, handle,
                           loop, timer, s_buf, r_buf, size, _type);
        num_msgs = loop * gridSize.x;
    }

    virtual void
    verifyResults(int my_id, uint64_t size)
    {
        int check_id = (_type == GetTestType || _type == GetNBITestType ||
                        _type == GetSwarmTestType) ? 0 : 1;
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
        ro_net_free(s_buf);
        ro_net_free(r_buf); 
    }

  protected:
    char *s_buf;
    char *r_buf;
    unsigned long long _max_size;
    TestType _type;
    int num_msgs;
};

class GetSwarmTester : public PrimitiveTester
{
  public:
    GetSwarmTester(TestType type)
        : PrimitiveTester(type) {}

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 uint64_t *timer, uint64_t size, ro_net_handle_t *handle)
    {
        hipLaunchKernelGGL(GetSwarmTest, dim3(gridSize),
                           dim3(blockSize), 0, stream, handle,
                           loop, timer, s_buf, r_buf, size);
        num_msgs = loop * gridSize.x * blockSize.x;
    }
};

class ReductionTester : public Tester
{
  public:
    virtual void
    initBuffers(unsigned long long max_size)
    {
        _max_size = max_size;
	    s_buf = (float *)ro_net_malloc(max_size * sizeof(float));
	    r_buf = (float *)ro_net_malloc(max_size * sizeof(float));
	    float *pWrk = (float *)ro_net_malloc(max_size * sizeof(float));
	    long *pSync = (long *)ro_net_malloc(max_size * sizeof(long));
    }

    virtual void
    launchKernel(dim3 gridSize, dim3 blockSize, hipStream_t stream, int loop,
                 uint64_t *timer, uint64_t size, ro_net_handle_t *handle)
    {
        hipLaunchKernelGGL(ReductionTest, dim3(gridSize),
                           dim3(blockSize), 0, stream, handle,
                           loop, timer, s_buf, r_buf, pWrk, pSync, size);
    }

    virtual void
    freeBuffers()
    {
        ro_net_free(s_buf);
        ro_net_free(r_buf);
        ro_net_free(pWrk);
        ro_net_free(pSync);
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
Tester::Create(TestType type)
{
    switch (type) {
        case InitTestType:
        case GetTestType:       // Get Test
        case GetNBITestType:    // Get NBI Test
        case PutTestType:       // Put Test
        case PutNBITestType:    // Put NBI Test
            return new PrimitiveTester(type);
        case GetSwarmTestType:  // Get Swarm Test
            return new GetSwarmTester(type);
        case ReductionTestType: // Reduction Test
            return new ReductionTester();
        default:
            return nullptr;
    }
}

#endif /* _TESTS_H */
