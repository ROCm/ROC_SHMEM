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

#ifndef ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_REGISTER_HPP
#define ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_REGISTER_HPP

#include "ro_net_internal.hpp"

namespace rocshmem {

struct BackendRegister {
    queue_element_t **queues {nullptr};
    queue_desc_t *queue_descs {nullptr};
    ROStats *profiler {nullptr};
    int num_threads {-1};
    bool done_flag {false};
    unsigned int *barrier_ptr {nullptr};
    bool *needs_quiet {nullptr};
    bool *needs_blocking {nullptr};
    uint64_t queue_size {0};
    char *g_ret {nullptr};
    HdpPolicy *hdp_policy {nullptr};
    WindowInfo **heap_window_info {nullptr};
    atomic_ret_t *atomic_ret {nullptr};
    bool gpu_queue {false};
    SymmetricHeap *heap_ptr {nullptr};
    int max_num_ctxs {-1};
    int *win_pool_alloc_bitmask {nullptr};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_REVERSE_OFFLOAD_BACKEND_REGISTER_HPP
