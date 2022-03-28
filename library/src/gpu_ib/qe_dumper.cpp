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

#include "qe_dumper.hpp"

namespace rocshmem {

QeDumper::QeDumper(int dest_pe,
                   int src_wg,
                   int index)
    : dest_pe_(dest_pe),
      src_wg_(src_wg),
      index_(index) {
    void* temp = malloc(sizeof(GPUIBBackend*));
    gpu_backend_ = static_cast<GPUIBBackend*>(temp);

    GPUIBBackend* device_backend_proxy_address;
    CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&device_backend_proxy_address),
                        HIP_SYMBOL(device_backend_proxy)));

    CHECK_HIP(hipMemcpy(&gpu_backend_,
                        device_backend_proxy_address,
                        sizeof(GPUIBBackend*),
                        hipMemcpyDeviceToHost));

    int qp_offset = gpu_backend_->num_wg * dest_pe_ + src_wg_;

    qp_ = &(gpu_backend_->networkImpl.gpu_qps[qp_offset]);
}

QeDumper::~QeDumper() {
    if (gpu_backend_) {
        free(gpu_backend_);
    }
}

void
QeDumper::dump_cq() {
    type_ = "CQ";

    auto *raw_cqe = &(qp_->current_cq_q_H[index_]);
    raw_u64_ = reinterpret_cast<uint64_t*>(raw_cqe);

    dump_uint64_(8);
}

void
QeDumper::dump_sq() {
    type_ = "SQ";

    auto *raw_sqe = &(qp_->current_sq_H[index_ * 8]);
    raw_u64_ = reinterpret_cast<uint64_t*>(raw_sqe);

    dump_uint64_(8);
}

void
QeDumper::dump_uint64_(size_t num_elems) const {
    printf("%s(%d, %d, %d) *** = ",
           type_.c_str(),
           dest_pe_,
           src_wg_,
           index_);

    for (size_t i = 0; i < num_elems; i++) {
        printf(" %lx ", raw_u64_[i]);
    }

    printf("done %s\n", type_.c_str());
}

}  // namespace rocshmem
