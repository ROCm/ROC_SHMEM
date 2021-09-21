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
#include "config.h"

#include <mpi.h>

#include "context.hpp"
#include "backend.hpp"
#include "host.hpp"
#include "util.hpp"

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
    assert(list_of_windows.begin() == list_of_windows.end());
}

__host__ void
ROHostContext::register_memory(void *ptr, size_t size)
{
    MPI_Win host_if_window;
    WindowInfo *window_info;

    MPI_Win_create(ptr, size, 1, MPI_INFO_NULL, host_interface->get_comm_world(), &host_if_window);

    window_info = list_of_windows.add(host_if_window, ptr, size);

    MPI_Win_lock_all(MPI_MODE_NOCHECK, window_info->get_win());
}

__host__ void
ROHostContext::deregister_memory(void *ptr)
{
    int idx = list_of_windows.get_window_idx(ptr);

    WindowInfo *window_info = list_of_windows.get_window_at(idx);

    MPI_Win win = window_info->get_win();
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);

    list_of_windows.delete_window_at(idx);
}

__host__ void
ROHostContext::putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_putmem_nbi\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dest);

    host_interface->putmem_nbi(dest, source, nelems, pe, window_info);
}

__host__ void
ROHostContext::getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_getmem_nbi\n"));

    WindowInfo *window_info = list_of_windows.get_window_info((void*) source);

    host_interface->getmem_nbi(dest, source, nelems, pe, window_info);
}

__host__ void
ROHostContext::putmem(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_putmem\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dest);

    host_interface->putmem(dest, source, nelems, pe, window_info);
}

__host__ void
ROHostContext::getmem(void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF(("Function: gpu_ib_host_getmem\n"));

    WindowInfo *window_info = list_of_windows.get_window_info((void*) source);

    host_interface->getmem(dest, source, nelems, pe, window_info);
}

__host__ void
ROHostContext::amo_add(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_add\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dst);

    host_interface->amo_add(dst, value, cond, pe, window_info);
}

__host__ void
ROHostContext::amo_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_cas\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dst);

    host_interface->amo_cas(dst, value, cond, pe, window_info);
}

__host__ int64_t
ROHostContext::amo_fetch_add(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_fetch_add\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dst);

    return host_interface->amo_fetch_add(dst, value, cond, pe, window_info);
}

__host__ int64_t
ROHostContext::amo_fetch_cas(void *dst, int64_t value, int64_t cond, int pe)
{
    DPRINTF(("Function: gpu_ib_host_amo_fetch_cas\n"));

    WindowInfo *window_info = list_of_windows.get_window_info(dst);

    return host_interface->amo_fetch_cas(dst, value, cond, pe, window_info);
}

__host__ void
ROHostContext::fence()
{
    DPRINTF(("Function: gpu_ib_host_fence\n"));

    for (int i = 0; i < list_of_windows.size(); i++) {
        host_interface->fence(list_of_windows.get_window_at(i));
    }
}

__host__ void
ROHostContext::quiet()
{
    DPRINTF(("Function: gpu_ib_host_quiet\n"));

    for (int i = 0; i < list_of_windows.size(); i++) {
        host_interface->quiet(list_of_windows.get_window_at(i));
    }
}

__host__ void
ROHostContext::sync_all()
{
    DPRINTF(("Function: gpu_ib_host_sync_all\n"));

    for (int i = 0; i < list_of_windows.size(); i++) {
        host_interface->sync_all(list_of_windows.get_window_at(i));
    }
}

__host__ void
ROHostContext::barrier_all()
{
    DPRINTF(("Function: gpu_ib_host_barrier_all\n"));

    for (int i = 0; i < list_of_windows.size(); i++) {
        host_interface->quiet(list_of_windows.get_window_at(i));
    }

    host_interface->barrier_for_sync();
}
