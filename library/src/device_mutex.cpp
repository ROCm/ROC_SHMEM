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

#include "device_mutex.hpp"
#include "util.hpp"

namespace rocshmem {

DeviceMutex::DeviceMutex(bool shareable)
    : shareable_(shareable) {
}

__device__ void
DeviceMutex::lock() {
    if (!shareable_) {
        return;
    }
    /*
     * We need to check this context out to a work-group, and only let threads
     * that are a part of the owning work-group through. It's a bit like a
     * re-entrant lock, with the added twist that a thread checks out the lock
     * for his entire work-group.
     *
     * TODO: This is a very tough thing to get right for GPU and needs
     * to be tested! Also it does nothing to ensure that every work-group
     * is getting its fair share of the lock.
     */
    int num_threads_in_wv = wave_SZ();

    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        /*
         * All the metadata associated with this lock needs to be accessed
         * atomically or it will race.
         */
        while (atomicCAS(&ctx_lock_, 0, 1) == 1) {
        }

        /*
         * If somebody in my work-group already owns the default context, just
         * record how many threads are going to be here and go about our
         * business.
         *
         * If my work-group doesn't own the default context, then
         * we need to wait for it to become available. Relinquish
         * ctx_lock while waiting or it will never become available.
         *
         */
        int wg_id = get_flat_grid_id();

        while (wg_owner_ != wg_id) {
            if (wg_owner_ == -1) {
                wg_owner_ = wg_id;
                __threadfence();
            } else {
                ctx_lock_ = 0;
                __threadfence();
                /*
                 * TODO: Might want to consider some back-off here.
                 */
                while (atomicCAS(&ctx_lock_, 0, 1) == 1) {
                }
            }
        }

        num_threads_in_lock_ += num_threads_in_wv;
        __threadfence();

        ctx_lock_ = 0;
        __threadfence();
    }
}

__device__ void
DeviceMutex::unlock() {
    if (!shareable_) {
        return;
    }

    int num_threads_in_wv {wave_SZ()};

    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        while (atomicCAS(&ctx_lock_, 0, 1) == 1) {
        }

        num_threads_in_lock_ -= num_threads_in_wv;

        /*
         * Last thread out for this work-group opens the door for other
         * work-groups to take possession.
         */
        if (num_threads_in_lock_ == 0) {
            wg_owner_ = -1;
        }

        __threadfence();

        ctx_lock_ = 0;
        __threadfence();
    }
}

}  // namespace rocshmem
