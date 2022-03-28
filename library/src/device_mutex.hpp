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

#ifndef ROCSHMEM_LIBRARY_SRC_DEVICE_MUTEX_HPP
#define ROCSHMEM_LIBRARY_SRC_DEVICE_MUTEX_HPP

#include <hip/hip_runtime.h>

namespace rocshmem {

class DeviceMutex {
  public:
    /**
     * @brief Secondary constructor
     */
    DeviceMutex() = default;

    /**
     * @brief Primary constructor
     */
    DeviceMutex(bool shareable);

    /**
     * @brief locks the device mutex
     *
     * @return void
     */
    __device__ void
    lock();

    /**
     * @brief unlocks the device mutex
     *
     * @return void
     */
    __device__ void
    unlock();

  private:
    /**
     * @brief Context can be shared between different workgroups.
     */
    bool shareable_ {false};

    /**
     * @brief Shareable context lock.
     */
    int ctx_lock_ {0};

    /**
     * @brief Shareable context owner.
     */
    volatile int wg_owner_ {-1};

    /**
     * @brief Number of threads in the owning block inside of locked calls.
     */
    volatile int num_threads_in_lock_ {0};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_DEVICE_MUTEX_HPP
