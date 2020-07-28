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

#include <roc_shmem.hpp>

#include "tester.hpp"
#include "tester_arguments.hpp"

int main(int argc, char * argv[])
{
    /**
     * Setup the tester arguments.
     */
    TesterArguments args(argc, argv);

    /**
     * Must initialize rocshmem to access arguments needed by the tester.
     */
    roc_shmem_init(args.num_wgs);

    /**
     * Now grab the arguments from rocshmem.
     */
    args.get_rocshmem_arguments();

    /**
     * Using the arguments we just constructed, call the tester factory
     * method to get the tester (specified by the arguments).
     */
    Tester *test = Tester::create(args);

    /**
     * Run the tests
     */
    test->execute();

    /**
     * The tester factory method news the tester to create it so we clean
     * up the memory here.
     */
    delete test;

    /**
     * The rocshmem library needs to be cleaned up with this call. It pairs
     * with the init function above.
     */
    roc_shmem_finalize();

    return 0;
}
