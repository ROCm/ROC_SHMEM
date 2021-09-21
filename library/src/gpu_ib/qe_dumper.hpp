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

#ifndef LIBRARY_SRC_GPU_IB_QE_DUMPER_HPP__
#define LIBRARY_SRC_GPU_IB_QE_DUMPER_HPP__

#include <hip/hip_runtime.h>
#include <infiniband/mlx5dv.h>

#include <string>

#include "backend.hpp"
#include "queue_pair.hpp"

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
    _dump_uint64(size_t num_elems) const;

    int _dest_pe;

    int _src_wg;

    int _index;

    GPUIBBackend *_gpu_backend = nullptr;

    std::string _type {};

    QueuePair *_qp = nullptr;

    uint64_t *_raw_u64 = nullptr;
};

#endif  // LIBRARY_SRC_GPU_IB_QE_DUMPER_HPP__
