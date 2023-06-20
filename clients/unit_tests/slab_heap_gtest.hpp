/******************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCSHMEM_SLAB_HEAP_GTEST_HPP
#define ROCSHMEM_SLAB_HEAP_GTEST_HPP

#include "gtest/gtest.h"

#include "memory/slab_heap.hpp"
#include "util.hpp"

namespace rocshmem {

/**
 * @brief Datatype used by test
 */
using TYPE = uint32_t;

/**
 * @brief The bit pattern written to memory by each thread.
 */
static const TYPE THREAD_VALUE {0xAA};

inline __device__
void
write_to_memory(TYPE* raw_memory) {
    auto thread_idx {get_flat_block_id()};
    raw_memory[thread_idx] = THREAD_VALUE;
    __threadfence();
}

inline __device__
TYPE*
allocate_memory(SlabHeap* slab) {
    auto block_size {get_flat_block_size()};

    TYPE* dyn_arr {nullptr};
    size_t num_bytes {block_size * sizeof(TYPE)};

    slab->malloc(reinterpret_cast<void**>(&dyn_arr), num_bytes);
    return dyn_arr;
}

__global__
void
all_threads_once(SlabHeap* slab) {
    auto block_mem {allocate_memory(slab)};
    write_to_memory(block_mem);
}

class SlabHeapTestFixture : public ::testing::Test {
    using SLAB_PROXY_T = SlabHeapProxy<HIPAllocator>;

  public:
    void
    run_all_threads_once(uint32_t x_block_dim,
                         uint32_t x_grid_dim) {
        auto slab {slab_.get()};

        const dim3 hip_blocksize(x_block_dim, 1, 1);
        const dim3 hip_gridsize(x_grid_dim, 1, 1);

        hipLaunchKernelGGL(all_threads_once,
                           hip_gridsize,
                           hip_blocksize,
                           0,
                           nullptr,
                           slab);

        hipError_t return_code = hipStreamSynchronize(nullptr);
        if (return_code != hipSuccess) {
            printf("Failed in stream synchronize\n");
            assert(return_code == hipSuccess);
        }

        TYPE* ptr {reinterpret_cast<TYPE*>(slab->get_base_ptr())};

        size_t number_threads {x_block_dim * x_grid_dim};

        for (size_t i {0}; i < number_threads; i++) {
            ASSERT_EQ(ptr[i], THREAD_VALUE);
        }
    }

  protected:
    /**
     * @brief Slab heap object
     */
    SLAB_PROXY_T slab_ {};
};

} // namespace rocshmem

#endif // ROCSHMEM_SLAB_HEAP_GTEST_HPP
