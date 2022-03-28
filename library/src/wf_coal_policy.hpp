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

#ifndef ROCSHMEM_LIBRARY_SRC_WF_COAL_POLICY_HPP
#define ROCSHMEM_LIBRARY_SRC_WF_COAL_POLICY_HPP

#include "config.h"

#include <hip/hip_runtime.h>

namespace rocshmem {

class WfCoalOn
{
  public:
    /**
     * Coalesce contiguous messages from a single wavefront.
     *
     * With regards to calling threads, the command must already be the
     * same for active threads otherwise they must have diverged at the
     * function call level.
     */
    __device__ bool
    coalesce(int pe,
             const void *source,
             const void *dest,
             size_t &size);
};

class WfCoalOff
{
  public:
    __device__ bool
    coalesce(int pe,
             const void *source,
             const void *dest,
             size_t &size)
    {
        return true;
    }
};

/**
 * Compile time configuration options will enable or disable this feature.
 */
#ifdef USE_WF_COAL
typedef WfCoalOn WavefrontCoalescer;
#else
typedef WfCoalOff WavefrontCoalescer;
#endif

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_WF_COAL_POLICY_HPP
