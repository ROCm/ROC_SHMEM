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

#ifndef ROCSHMEM_FORWARD_LIST_GTEST_HPP
#define ROCSHMEM_FORWARD_LIST_GTEST_HPP

#include "gtest/gtest.h"

#include "containers/forward_list_impl.hpp"
#include "memory/hip_allocator.hpp"

namespace rocshmem {

class ForwardListTestFixture : public ::testing::Test {
  public:
    ForwardListTestFixture() {
    }

    void
    default_constructor_test() {
        ASSERT_EQ(list_.head_, nullptr);
        ASSERT_EQ(to_string(list_), std::string{});
    }

  private:
    ForwardList<std::string> list_ {};
};

} // namespace rocshmem

#endif  // ROCSHMEM_FORWARD_LIST_GTEST_HPP
