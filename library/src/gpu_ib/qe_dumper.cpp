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

#include "qe_dumper.hpp"

QeDumper::QeDumper(int dest_pe,
                   int src_wg,
                   int index)
    : _dest_pe(dest_pe),
      _src_wg(src_wg),
      _index(index) {
    void *temp = malloc(sizeof(GPUIBBackend*));
    _gpu_backend = static_cast<GPUIBBackend*>(temp);

    GPUIBBackend *gpu_handle_address;
    hipGetSymbolAddress(reinterpret_cast<void**>(&gpu_handle_address),
                        HIP_SYMBOL(gpu_handle));

    hipMemcpy(&_gpu_backend,
              gpu_handle_address,
              sizeof(GPUIBBackend*),
              hipMemcpyDeviceToHost);

    int qp_offset = _gpu_backend->num_wg * _dest_pe + _src_wg;

    _qp = &(_gpu_backend->networkImpl.gpu_qps[qp_offset]);
}

QeDumper::~QeDumper() {
    if (_gpu_backend) {
        free(_gpu_backend);
    }
}

void
QeDumper::dump_cq() {
    _type = "CQ";

    auto *raw_cqe = &(_qp->current_cq_q_H[_index]);
    _raw_u64 = reinterpret_cast<uint64_t*>(raw_cqe);

    _dump_uint64(8);
}

void
QeDumper::dump_sq() {
    _type = "SQ";

    auto *raw_sqe = &(_qp->current_sq_H[_index * 8]);
    _raw_u64 = reinterpret_cast<uint64_t*>(raw_sqe);

    _dump_uint64(8);
}

void
QeDumper::_dump_uint64(size_t num_elems) const {
    printf("%s(%d, %d, %d) *** = ",
           _type.c_str(),
           _dest_pe,
           _src_wg,
           _index);

    for (size_t i = 0; i < num_elems; i++) {
        printf(" %lx ", _raw_u64[i]);
    }

    printf("done %s\n", _type.c_str());
}
