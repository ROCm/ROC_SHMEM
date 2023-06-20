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

#ifndef ROCSHMEM_NOTIFIER_GTEST_HPP
#define ROCSHMEM_NOTIFIER_GTEST_HPP

#include "gtest/gtest.h"

#include "memory/hip_allocator.hpp"
#include "memory/notifier.hpp"
#include "util.hpp"

namespace rocshmem {

/**
 * @brief The bit pattern written to memory by each thread.
 */
static const uint8_t THREAD_VALUE {0xF9};

/**
 * @brief The bit pattern written to memory by each thread.
 */
static const uint64_t NOTIFIER_OFFSET {0x100B00};

inline __device__
void
write_to_memory(uint8_t* raw_memory) {
    auto thread_idx {get_flat_block_id()};
    raw_memory[thread_idx] = THREAD_VALUE;
    __threadfence();
}

__global__
void
all_threads_once(uint8_t* raw_memory,
                 Notifier* notifier) {
    notifier->write(NOTIFIER_OFFSET);
    uint64_t offset_u64 {notifier->read()};
    notifier->done();

    uint64_t raw_memory_u64 {reinterpret_cast<uint64_t>(raw_memory)};
    uint64_t address_u64 {raw_memory_u64 + offset_u64};
    uint8_t* address {reinterpret_cast<uint8_t*>(address_u64)};
    write_to_memory(address);
    __syncthreads();
}

class NotifierTestFixture : public ::testing::Test {
    using NotifierProxyT = NotifierProxy<HIPAllocator>;

  public:
    NotifierTestFixture() {
        assert(raw_memory_ == nullptr);
        hip_allocator_.allocate((void**)&raw_memory_, GIBIBYTE_);
        assert(raw_memory_);
    }

    ~NotifierTestFixture() {
        if (raw_memory_) {
            hip_allocator_.deallocate(raw_memory_);
        }
    }

    void
    run_all_threads_once(uint32_t x_block_dim,
                         uint32_t x_grid_dim) {
        const dim3 hip_blocksize(x_block_dim, 1, 1);
        const dim3 hip_gridsize(x_grid_dim, 1, 1);

        hipLaunchKernelGGL(all_threads_once,
                           hip_gridsize,
                           hip_blocksize,
                           0,
                           nullptr,
                           raw_memory_,
                           notifier_.get());

        hipError_t return_code = hipStreamSynchronize(nullptr);
        if (return_code != hipSuccess) {
            printf("Failed in stream synchronize\n");
            assert(return_code == hipSuccess);
        }

        size_t number_threads {x_block_dim * x_grid_dim};

        uint8_t* offset_addr {compute_offset_addr()};

        for (size_t i {0}; i < number_threads; i++) {
            ASSERT_EQ(offset_addr[i], THREAD_VALUE);
        }
    }

  protected:
    /**
     * @brief Helper function to reconstruct device calculation.
     */
    uint8_t*
    compute_offset_addr() {
        uint64_t raw_memory_u64 {reinterpret_cast<uint64_t>(raw_memory_)};
        uint64_t address_u64 {raw_memory_u64 + NOTIFIER_OFFSET};
        uint8_t* address {reinterpret_cast<uint8_t*>(address_u64)};
        return address;
    }

    /**
     * @brief An allocator to create objects in device memory.
     */
    HIPAllocator hip_allocator_ {};

    /**
     * @brief The size of the raw memory block below.
     */
    static const size_t GIBIBYTE_ {1 << 30};

    /**
     * @brief A block of memory used to hold individual writes from threads.
     */
    uint8_t *raw_memory_ {nullptr};

    /**
     * @brief Used to broadcast base offset for writing.
     */
    NotifierProxyT notifier_ {};
};


} // namespace rocshmem

#endif  // ROCSHMEM_NOTIFIER_GTEST_HPP
