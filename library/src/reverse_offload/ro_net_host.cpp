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
#include "config.h"

#include <mpi.h>

#include "context_incl.hpp"
#include "backend_ro.hpp"
#include "backend_type.hpp"
#include "host.hpp"

namespace rocshmem {

__host__
ROHostContext::ROHostContext(const Backend &backend, long options)
    : Context(backend, true)
{
    backend_handle = (Backend *) (&backend);
    type = BackendType::RO_BACKEND;

    const ROBackend* b = static_cast<const ROBackend *>(&backend);

    host_interface = b->host_interface;
}

__host__
ROHostContext::~ROHostContext()
{
    /*
     * Nothing to do here since we haven't allocated any new memory on
     * this object. Just assert that the vector of windows is empty.
     */
}

__host__
WindowInfo*
ROHostContext::get_window_info() {
    return backend_handle->heap.get_window_info();
}

__host__ void
ROHostContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_putmem_nbi\n"));

    host_interface->putmem_nbi(dest, source, nelems, pe, get_window_info());
}

__host__ void
ROHostContext::getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_getmem_nbi\n"));

    host_interface->getmem_nbi(dest, source, nelems, pe, get_window_info());
}

__host__ void
ROHostContext::putmem(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_putmem\n"));

    host_interface->putmem(dest, source, nelems, pe, get_window_info());
}

__host__ void
ROHostContext::getmem(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_getmem\n"));

    host_interface->getmem(dest, source, nelems, pe, get_window_info());
}

__host__ void
ROHostContext::amo_add(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_add\n"));

    host_interface->amo_add(dst, value, cond, pe, get_window_info());
}

__host__ void
ROHostContext::amo_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_cas\n"));

    host_interface->amo_cas(dst, value, cond, pe, get_window_info());
}

__host__ int64_t
ROHostContext::amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_fetch_add\n"));

    return host_interface->amo_fetch_add(dst, value, cond, pe, get_window_info());
}

__host__ int64_t
ROHostContext::amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_fetch_cas\n"));

    return host_interface->amo_fetch_cas(dst, value, cond, pe, get_window_info());
}

__host__ void
ROHostContext::fence()
{
    DPRINTF(("Function: gpu_ib_host_fence\n"));

    host_interface->fence(get_window_info());
}

__host__ void
ROHostContext::quiet()
{
    DPRINTF(("Function: gpu_ib_host_quiet\n"));

    host_interface->quiet(get_window_info());
}

__host__ void
ROHostContext::sync_all()
{
    DPRINTF(("Function: gpu_ib_host_sync_all\n"));

    host_interface->sync_all(get_window_info());
}

__host__ void
ROHostContext::barrier_all()
{
    DPRINTF(("Function: gpu_ib_host_barrier_all\n"));

    host_interface->fence(get_window_info());

    host_interface->barrier_for_sync();
}

}  // namespace rocshmem
