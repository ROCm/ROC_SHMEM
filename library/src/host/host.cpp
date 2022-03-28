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

#include <mpi.h>

#include "config.h"  // NOLINT(build/include_subdir)
#include "host.hpp"
#include "host_helpers.hpp"
#include "util.hpp"
#include "window_info.hpp"

namespace rocshmem {

__host__
HostInterface::HostInterface(HdpPolicy* hdp_policy,
                             MPI_Comm roc_shmem_comm) {
    /*
     * Duplicate a communicator from roc_shem's comm
     * world for the host interface
     */
    MPI_Comm_dup(roc_shmem_comm, &host_comm_world_);
    MPI_Comm_rank(host_comm_world_, &my_pe_);
    MPI_Comm_rank(host_comm_world_, &num_pes_);

    /*
     * Create an MPI window on the HDP so that it can be flushed
     * by remote PEs for host-facing functions
     */
    hdp_policy_ = hdp_policy;

    // TODO(rozambre): enable in rocm 4.5
    // commenting this out until rocm 4.5
    // MPI_Win_create(hdp_policy->get_hdp_flush_addr(),
    //                sizeof(unsigned int),        /* size of window */
    //                sizeof(unsigned int),        /* displacement */
    //                MPI_INFO_NULL,
    //                host_comm_world_,
    //                &hdp_win);

    /*
     * Start a shared access epoch on windows of all ranks,
     * and let the library there is no need to check for
     * lock exclusivity during operations on this window
     * (MPI_MODE_NOCHECK).
     */
    // TODO(rozambre): enable in rocm 4.5
    // MPI_Win_lock_all(MPI_MODE_NOCHECK, hdp_win);
}

__host__
HostInterface::~HostInterface() {
    // TODO(rozambre): enable in rocm 4.5
    // MPI_Win_unlock_all(hdp_win);

    // MPI_Win_free(&hdp_win);

    MPI_Comm_free(&host_comm_world_);
}

__host__ void
HostInterface::putmem_nbi(void* dest,
                          const void* source,
                          size_t nelems,
                          int pe,
                          WindowInfo* window_info) {
    initiate_put(dest, source, nelems, pe, window_info);
}

__host__ void
HostInterface::getmem_nbi(void* dest,
                          const void* source,
                          size_t nelems,
                          int pe,
                          WindowInfo* window_info) {
    initiate_get(dest, source, nelems, pe, window_info);
}

__host__ void
HostInterface::putmem(void* dest,
                      const void* source,
                      size_t nelems,
                      int pe,
                      WindowInfo* window_info) {
    initiate_put(dest, source, nelems, pe, window_info);

    MPI_Win_flush_local(pe, window_info->get_win());
}

__host__ void
HostInterface::getmem(void* dest,
                      const void* source,
                      size_t nelems,
                      int pe,
                      WindowInfo* window_info) {
    initiate_get(dest, source, nelems, pe, window_info);

    MPI_Win_flush_local(pe, window_info->get_win());

    /*
     * Flush local HDP to ensure that the NIC's write
     * of the fetched data is visible in device memory
     */
    hdp_policy_->hdp_flush();
}

__host__ void
HostInterface::amo_add(void* dst,
                       int64_t value,
                       int64_t cond,
                       int pe,
                       WindowInfo* window_info) {
    /*
     * Most MPI implementations tend to use active messages to implement
     * MPI_Accumulate. So, to eliminate the potential involvement of the
     * target PE, we instead use fetch_add and disregard the return value.
     */
    int64_t ret {amo_fetch_add(dst, value, cond, pe, window_info)};
}

__host__ void
HostInterface::amo_cas(void* dst,
                       int64_t value,
                       int64_t cond,
                       int pe,
                       WindowInfo* window_info) {
    /* Perform the compare and swap and disregard the return value */
    int64_t ret {amo_fetch_cas(dst, value, cond, pe, window_info)};
}

__host__ int64_t
HostInterface::amo_fetch_add(void* dst,
                             int64_t value,
                             int64_t cond,
                             int pe,
                             WindowInfo* window_info) {
    /* Calculate offset of remote dest from base address of window */
    MPI_Aint offset {
        compute_offset(dst, window_info->get_start(), window_info->get_end())
    };

    /*
     * Flush the HDP of the remote PE so that the NIC does not
     * read stale values
     */
    flush_remote_hdp(pe);

    /* Offload remote fetch and op operation to MPI */
    int64_t ret {};
    MPI_Win win {window_info->get_win()};
    MPI_Fetch_and_op(&value, &ret, MPI_INT64_T, pe, offset, MPI_SUM, win);

    MPI_Win_flush_local(pe, win);

    return ret;
}

__host__ int64_t
HostInterface::amo_fetch_cas(void* dst,
                             int64_t value,
                             int64_t cond,
                             int pe,
                             WindowInfo* window_info) {
    /* Calculate offset of remote dest from base address of window */
    MPI_Aint offset {
        compute_offset(dst, window_info->get_start(), window_info->get_end())
    };

    /*
     * Flush the HDP of the remote PE so that the NIC does not
     * read stale values
     */
    flush_remote_hdp(pe);

    /* Offload remote compare and swap operation to MPI */
    int64_t ret {};
    MPI_Win win {window_info->get_win()};
    MPI_Compare_and_swap(&value,
                         &cond,
                         &ret,
                         MPI_INT64_T,
                         pe,
                         offset,
                         win);

    MPI_Win_flush_local(pe, win);

    return ret;
}

__host__ void inline
HostInterface::flush_remote_hdp(int pe) {
    unsigned flush_val {HdpBasePolicy::HDP_FLUSH_VAL};
    // TODO(rozambre): enable for rocm 4.5
    // MPI_Put(&flush_val, 1, MPI_UNSIGNED, pe, 0, 1, MPI_UNSIGNED, hdp_win);
    // MPI_Win_flush(pe, hdp_win);
}

__host__ void inline
HostInterface::flush_remote_hdps() {
    unsigned flush_val {HdpBasePolicy::HDP_FLUSH_VAL};
    for (size_t i {0}; i < num_pes_; i++) {
        if (i == my_pe_) {
            continue;
        }
        // TODO(rozambre): enable for rocm 4.5
        // MPI_Put(&flush_val,
        //         1,
        //         MPI_UNSIGNED,
        //         i,
        //         0,
        //         1,
        //         MPI_UNSIGNED,
        //         hdp_win);
    }
    // MPI_Win_flush_all(hdp_win);
}

__host__ void
HostInterface::fence(WindowInfo* window_info) {
    complete_all(window_info->get_win());

    /*
     * Flush my HDP and the HDPs of remote GPUs.
     * The HDP is a write-combining (WC) write-through
     * cache. But, even after the WC buffer is full and
     * the data is passed to the Data Fabric (DF), DF
     * can still reorder the writes. A flush ensures
     * that writes after the flush are written only
     * after those before the flush.
     */
    hdp_policy_->hdp_flush();
    flush_remote_hdps();

    return;
}

__host__ void
HostInterface::quiet(WindowInfo* window_info) {
    complete_all(window_info->get_win());

    /* Same explanation as in fence */
    hdp_policy_->hdp_flush();
    flush_remote_hdps();

    return;
}

__host__ void
HostInterface::sync_all(WindowInfo* window_info) {
    MPI_Win_sync(window_info->get_win());

    hdp_policy_->hdp_flush();
    /*
     * No need to flush remote
     * HDPs here since all PEs are
     * participating.
     */

    MPI_Barrier(host_comm_world_);

    return;
}

__host__ void
HostInterface::barrier_all(WindowInfo* window_info) {
    complete_all(window_info->get_win());

    /*
     * Flush my HDP cache so remote NICs will
     * see the latest values in device memory
     */
    hdp_policy_->hdp_flush();

    MPI_Barrier(host_comm_world_);
}

__host__ void
HostInterface::barrier_for_sync() {
    MPI_Barrier(host_comm_world_);
}

}  // namespace rocshmem
