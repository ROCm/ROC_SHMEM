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

#ifndef ROCSHMEM_LIBRARY_SRC_BACKEND_TYPE_HPP
#define ROCSHMEM_LIBRARY_SRC_BACKEND_TYPE_HPP

/**
 * @file backend_type.hpp
 * Defines the Backend derived class types and contains the DISPATCH macros.
 *
 * The type information is required to be known at compile time because
 * we use static dispatch to produce compile time polymorphism.
 *
 * The device cannot use runtime polymorphism because calls through virtual
 * functions are not supported at this time.
 */

#include "device_mutex.hpp"

namespace rocshmem {

/**
 * @brief Enumerates the Backend derived classes.
 *
 * @note Derived classes which use Backend as a base class must add
 * themselves to this enum class to support static polymorphism.
 */
enum class BackendType {
    RO_BACKEND,
    GPU_IB_BACKEND
};

/**
 * @brief Helper macro for some dispatch calls
 */
#define PAIR(A, B) A, B

/**
 * @brief Static dispatch method call.
 *
 * @note This variant does not require a lock to protect the method call.
 */
#define DISPATCH_NO_LOCK(Func) \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            static_cast<ROContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            static_cast<GPUIBContext*>(this)->Func; \
            break; \
        default: \
            break; \
    }

/**
 * @brief Device static dispatch method call.
 */
#define DISPATCH(Func) \
    dev_mtx_.lock(); \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            static_cast<ROContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            static_cast<GPUIBContext*>(this)->Func; \
            break; \
        default: \
            break; \
    } \
    dev_mtx_.unlock();

/**
 * @brief Device static dispatch method call with a return value.
 */
#define DISPATCH_RET(Func) \
    dev_mtx_.lock(); \
    auto ret_val {0}; \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            ret_val = static_cast<ROContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            ret_val = static_cast<GPUIBContext*>(this)->Func; \
            break; \
        default: \
            break; \
    } \
    dev_mtx_.unlock(); \
    return ret_val;

/**
 * @brief Device static dispatch method call with a return type of pointer.
 */
#define DISPATCH_RET_PTR(Func) \
    dev_mtx_.lock(); \
    void *ret_val {nullptr}; \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            ret_val = static_cast<ROContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            ret_val = static_cast<GPUIBContext*>(this)->Func; \
            break; \
        default: \
            break; \
    } \
    dev_mtx_.unlock(); \
    return ret_val;

/**
 * @brief Host static dispatch method call.
 *
 * @note There is no need to lock-unlock on host since we are using
 * MPI_THREAD_MULTIPLE (for RMA and AMO operations) and the ordering and
 * threading semantics of collectives in OpenSHMEM match those of MPI.
 */
#define HOST_DISPATCH(Func) \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            static_cast<ROHostContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            static_cast<GPUIBHostContext*>(this)->Func; \
            break; \
        default: \
            break; \
    }

/**
 * @brief Host static dispatch method call with return value.
 *
 * @note There is no need to lock-unlock on host since we are using
 * MPI_THREAD_MULTIPLE (for RMA and AMO operations) and the ordering and
 * threading semantics of collectives in OpenSHMEM match those of MPI.
 */
#define HOST_DISPATCH_RET(Func) \
    auto ret_val {0}; \
    switch (type) { \
        case BackendType::RO_BACKEND: \
            ret_val = static_cast<ROHostContext*>(this)->Func; \
            break; \
        case BackendType::GPU_IB_BACKEND: \
            ret_val = static_cast<GPUIBHostContext*>(this)->Func; \
            break; \
        default: \
            break; \
    } \
    return ret_val;

} // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_BACKEND_TYPE_HPP
