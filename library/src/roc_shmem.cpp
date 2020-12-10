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

/**
 * @file roc_shmem.cpp
 * @brief Public header for ROC_SHMEM device and host libraries.
 *
 * This is the implementation for the public roc_shmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 *
 * The device-side delegation is nasty because we can't use polymorphism with
 * our current shader compiler stack.  Maybe one day.....
 *
 * TODO: Could probably autogenerate many of these functions from macros.
 *
 * TODO: Support runtime backend detection.
 *
 */

#include <roc_shmem.hpp>

#include "backend.hpp"
#include "util.hpp"

#include <stdlib.h>

#define VERIFY_BACKEND() {                                                   \
        if (!backend) {                                                      \
            fprintf(stderr, "ROC_SHMEM_ERROR: %s in file '%s' in line %d\n", \
                            "Call 'roc_shmem_init'", __FILE__, __LINE__);\
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

Backend *backend = nullptr;

/**
 * Begin Host Code
 **/

[[maybe_unused]] Status
roc_shmem_init(unsigned num_wgs)
{
    assert(!backend);

    if (getenv("ROC_SHMEM_DEBUG") != nullptr)
        ROC_SHMEM_DEBUG = true;

    rocm_init();
    if (getenv("ROC_SHMEM_RO") != nullptr) {
        hipHostMalloc(&backend, sizeof(ROBackend));
        backend = new (backend) ROBackend(num_wgs);
    } else {
        hipHostMalloc(&backend, sizeof(GPUIBBackend));
        backend = new (backend) GPUIBBackend(num_wgs);
    }

    if (!backend)
        return Status::ROC_SHMEM_OOM_ERROR;

   return Status::ROC_SHMEM_SUCCESS;
}

[[maybe_unused]] int
roc_shmem_my_pe()
{
    VERIFY_BACKEND();
    return backend->getMyPE();
}

[[maybe_unused]] int
roc_shmem_n_pes()
{
    VERIFY_BACKEND();
    return backend->getNumPEs();
}

[[maybe_unused]] void *
roc_shmem_malloc(size_t size)
{
    VERIFY_BACKEND();

    void *ptr;
    backend->net_malloc(&ptr, size);
    return ptr;
}

[[maybe_unused]] Status
roc_shmem_free(void *ptr)
{
    VERIFY_BACKEND();
    return backend->net_free(ptr);
}

[[maybe_unused]] Status
roc_shmem_reset_stats()
{
    VERIFY_BACKEND();
    return backend->reset_stats();
}

[[maybe_unused]] Status
roc_shmem_dump_stats()
{
    /** TODO: Many stats are backend independent! **/
    VERIFY_BACKEND();
    return backend->dump_stats();
}

[[maybe_unused]] Status
roc_shmem_finalize()
{
    VERIFY_BACKEND();

    backend->~Backend();
    hipHostFree(backend);

    return Status::ROC_SHMEM_SUCCESS;
}

[[maybe_unused]] Status
roc_shmem_dynamic_shared(size_t *shared_bytes)
{
    VERIFY_BACKEND();
    backend->dynamic_shared(shared_bytes);
    return Status::ROC_SHMEM_SUCCESS;
}
