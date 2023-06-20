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

#include "slab_heap_gtest.hpp"

using namespace rocshmem;

TEST_F(SlabHeapTestFixture, malloc_free) {
  void *ptr{nullptr};
  size_t request_bytes{48};

  auto slab{slab_.get()};
  slab->malloc(&ptr, request_bytes);

  ASSERT_NE(ptr, nullptr);
  ASSERT_NO_FATAL_FAILURE(slab->free(ptr));
}

TEST_F(SlabHeapTestFixture, overallocate_2GiB) {
  void *ptr{nullptr};
  size_t request_bytes{1UL << 31};

  auto slab{slab_.get()};
  slab->malloc(&ptr, request_bytes);

  ASSERT_EQ(ptr, nullptr);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_1) {
  run_all_threads_once(1, 1);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_2_1) {
  run_all_threads_once(2, 1);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_64_1) {
  run_all_threads_once(64, 1);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_128_1) {
  run_all_threads_once(128, 1);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_256_1) {
  run_all_threads_once(256, 1);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_512_1) {
  run_all_threads_once(512, 1);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1024_1) {
  run_all_threads_once(1024, 1);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_2) {
  run_all_threads_once(1, 2);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_8) {
  run_all_threads_once(1, 8);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_64) {
  run_all_threads_once(1, 64);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_128) {
  run_all_threads_once(1, 128);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_256) {
  run_all_threads_once(1, 256);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_1024) {
  run_all_threads_once(1, 1024);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_2048) {
  run_all_threads_once(1, 2048);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_4096) {
  run_all_threads_once(1, 4096);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_8192) {
  run_all_threads_once(1, 8192);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1_65536) {
  run_all_threads_once(1, 65536);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_2_2) {
  run_all_threads_once(2, 2);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_64_2) {
  run_all_threads_once(64, 2);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_256_50) {
  run_all_threads_once(256, 50);
}

TEST_F(SlabHeapTestFixture, run_all_threads_once_1024_512) {
  run_all_threads_once(1024, 512);
}
