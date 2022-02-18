/******************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef LIBRARY_SRC_HOST_HOST_HELPERS_HPP_
#define LIBRARY_SRC_HOST_HOST_HELPERS_HPP_

#include "host.hpp"

__host__ inline MPI_Aint
HostInterface::compute_offset(const void *dest,
                              void *win_start,
                              void *win_end) {
    MPI_Aint dest_disp;
    MPI_Aint start_disp;

    assert((reinterpret_cast<const char*>(dest) >=
            reinterpret_cast<char*>(win_start)) &&
           (reinterpret_cast<const char*>(dest) <
            reinterpret_cast<char*>(win_end)));

    MPI_Get_address(dest, &dest_disp);
    MPI_Get_address(win_start, &start_disp);

    return MPI_Aint_diff(dest_disp, start_disp);
}

__host__ inline void
HostInterface::complete_all(MPI_Win win) {
    MPI_Win_flush_all(win);   /* RMA operations */
    MPI_Win_sync(win);        /* memory stores */
}

__host__ inline void
HostInterface::initiate_put(void *dest,
                            const void *source,
                            size_t nelems,
                            int pe,
                            WindowInfo *window_info) {
    MPI_Win win = window_info->get_win();
    void *win_start = window_info->get_start();
    void *win_end = window_info->get_end();

    /* Calculate offset of remote dest from base address of window */
    MPI_Aint offset = compute_offset(dest, win_start, win_end);

    /*
     * Current semantics of our API restrict the buffers
     * passed in to be on the symmetric heap only. So,
     * flush the HDP since the GPU may have written the
     * latest value to the source buffer and we want the
     * NIC to DMA read the latest value instead of the
     * value that may have been cached in the HDP.
     */
    hdp_policy->hdp_flush();

    /* Offload remote write operation to MPI */
    MPI_Put(source, nelems, MPI_CHAR, pe, offset, nelems, MPI_CHAR, win);
}

__host__ inline void
HostInterface::initiate_get(void *dest,
                            const void *source,
                            size_t nelems,
                            int pe,
                            WindowInfo *window_info) {
    MPI_Win win = window_info->get_win();
    void *win_start = window_info->get_start();
    void *win_end = window_info->get_end();

    /* Calculate offset of remote source from base address of window */
    MPI_Aint offset = compute_offset(source, win_start, win_end);

    /* Offload remote fetch operation to MPI */
    MPI_Get(dest, nelems, MPI_CHAR, pe, offset, nelems, MPI_CHAR, win);
}

#endif  // LIBRARY_SRC_HOST_HOST_HELPERS_HPP_
