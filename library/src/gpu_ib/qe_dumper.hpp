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

#ifndef ROCSHMEM_LIBRARY_SRC_GPU_IB_QE_DUMPER_HPP
#define ROCSHMEM_LIBRARY_SRC_GPU_IB_QE_DUMPER_HPP

#include <hip/hip_runtime.h>
#include <infiniband/mlx5dv.h>

#include <string>

#include "backend_ib.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

class QeDumper {
 public:
    QeDumper(int dest_pe,
             int src_wg,
             int index);

    ~QeDumper();

    void
    dump_cq();

    void
    dump_sq();

 private:
    void
    dump_uint64_(size_t num_elems) const;

    int dest_pe_ {-1};

    int src_wg_ {-1};

    int index_ {-1};

    GPUIBBackend* gpu_backend_ {nullptr};

    std::string type_ {};

    QueuePair* qp_ {nullptr};

    uint64_t* raw_u64_ {nullptr};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_GPU_IB_QE_DUMPER_HPP
