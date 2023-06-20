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

#include "forward_list_gtest.hpp"

#include <iostream>
#include <string>

using namespace rocshmem;

/*****************************************************************************
 ******************************* Fixture Tests *******************************
 *****************************************************************************/

TEST_F(ForwardListTestFixture, default_constructor) {
  default_constructor_test();
}

TEST(ForwardListTest, constructor_tests) {
  ForwardList<std::string> list_1{"rocshmem", "forward_list"};
  std::string str_1 = "rocshmem forward_list";
  ASSERT_EQ(to_string(list_1), str_1);

  // ForwardList<std::string> list_2(list_1.begin(), list_1.end());
  // std::cout << "list_2: " << list_2 << '\n';

  // ForwardList<std::string> list_3(list_1);
  // std::cout << "list_3: " << list_3 << '\n';

  // ForwardList<std::string> list_4(5, "rocm");
  // std::cout << "list_4: " << list_4 << '\n';
}
