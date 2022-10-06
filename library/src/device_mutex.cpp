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

__device__
uint64_t
number_active_lanes()
{
    /*
     * The __ballot(1) built-in instruction conducts an active lane roll
     * call storing the roll call result in a bit vector. Using its index
     * within the warp, each active lane contributes a '1' to the bit vector
     * Inactive lanes cannot contribute to the bit vector so their lane
     * values contain a '0'.
     *
     * For example, assume lanes 2, 5, and 62 active out of 64 lanes (0...63).
     * The result of the ballot would be the following:
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [0 0 1 0 0 1 0 ... 1 0]
     *
     * The __ballot(1) result is fed to __popcll.
     *
     * The __popcll instruction conducts a population count; it checks
     * the number of '1' values in a bit vector.
     *
     * Using the previous example, the result would be a 3 (since indices
     * 2, 5, and 62) contain a '1'.
     */
    return __popcll(__ballot(1));
}

__device__
uint64_t
lowest_active_lane()
{
    /*
     * The __ballot(1) built-in instruction conducts an active lane roll
     * call storing the roll call result in a bit vector. Using its index
     * within the warp, each active lane contributes a '1' to the bit vector
     * Inactive lanes cannot contribute to the bit vector so their lane
     * values contain a '0'.
     *
     * For example, assume lanes 2, 5, and 62 active out of 64 lanes (0...63).
     * The result of the ballot would be the following:
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [0 0 1 0 0 1 0 ... 1 0]
     *
     * The __ballot(1) result is fed to __ffsll.
     *
     * The __ffsll instruction finds the index of the least significant
     * bit set to '1' (in the input bit vector).
     *
     * In the previous example, the return result would be 2 (since index
     * 2 is the first bit with a '1').
     *
     * The '- 1' at the end is necessary because the return index is 1-based
     * instead of 0-based ([1...64] vs [0...63]).
     */
    return __ffsll(__ballot(1)) - 1;
}

__device__
uint64_t
active_logical_lane_id()
{
    /*
     * The __ballot(1) built-in instruction conducts an active lane roll
     * call storing the roll call result in a bit vector. Using its index
     * within the warp, each active lane contributes a '1' to the bit vector
     * Inactive lanes cannot contribute to the bit vector so their lane
     * values contain a '0'.
     *
     * For example, assume lanes 2, 5, and 62 active out of 64 lanes (0...63).
     * The result of the ballot would be the following:
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [0 0 1 0 0 1 0 ... 1 0]
     */
    uint64_t ballot = __ballot(1);

    /*
     * The physical_lane_id is the warp lane index of the thread executing
     * this code. The word 'physical' here denotes that the lane_id will
     * be the actual hardware lane_id.
     */
    uint64_t my_physical_lane_id = __lane_id();

    /*
     * Create a full bitset for subsequent operations.
     *
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [1 1 1 1 1 1 1 ... 1 1]
     */
    uint64_t all_ones_mask = -1;

    /*
     * Left-shift to zero-out the mask elements up to our lane_id.
     *
     * As an example, assume our lane_id is '5':
     *
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [0 0 0 0 0 1 1 ... 1 1]
     */
    uint64_t lane_mask = all_ones_mask << my_physical_lane_id;

    /*
     * Invert the lane_mask.
     *
     * Continue with lane_id '5' example:
     *
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [1 1 1 1 1 0 0 ... 0 0]
     */
    uint64_t inverted_mask = ~lane_mask;

    /*
     * Bit-wise And the inverted_mask and the ballot.
     *
     * The result contains a bitset with all active_lanes preceding this
     * thread in the ballot (all active threads with lower lane_ids).
     *
     * Continue with lane_id '5' example:
     *
     * ballot
     * ------------------------------
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [0 0 1 0 0 1 0 ... 1 0]
     *
     * inverted_mask
     * ------------------------------
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [1 1 1 1 1 0 0 ... 0 0]
     *
     * lower_active_lanes
     * ------------------------------
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [0 0 1 0 0 0 0 ... 0 0]
     */
    uint64_t lower_active_lanes = ballot & inverted_mask;

    /*
     * Conduct a population count on lower_active_lanes.
     *
     * The result gives an index into our logical_lane_id.
     *
     * Continue with lane_id '5' example:
     *
     * lower_active_lanes
     * ------------------------------
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     * value: [0 0 1 0 0 0 0 ... 0 0]
     *
     * my_logical_lane_id
     * ------------------------------
     *        [- - X - - - - ... - -] <- population_count = 1
     *
     * index:  0 0 0 0 0 0 0 ... 6 6
     *         0 1 2 3 4 5 6 ... 2 3
     *        [- - 0 - - 1 - ... 2 -] <- my_logical_lane_id = 1
     */
    uint64_t my_logical_lane_id = __popcll(lower_active_lanes);

    return my_logical_lane_id;
}


__device__ void
DeviceMutex::lock() {
    //auto total_active_lanes {number_active_lanes()};
    //auto logical_lane_id {active_logical_lane_id()};
    //bool is_lowest_active_lane {logical_lane_id == 0};
    //auto slot_in_ticket_array {get_flat_block_id() / warpSize};

    //if (is_lowest_active_lane) {
        //per_warp_ticket[slot_in_ticket_array] = 0;
    //}
    //__syncthreads_count(total_active_lanes);

    //while (per_warp_ticket != total_active_lanes) {
        //if (per_warp_ticket == logical_lane_id()) {
            //while (atomicCAS((int*)&lock_, 0, 1) == 1) {
            //}
            //__threadfence();
            //atomicAdd((int*)&per_warp_ticket, 1);
            //break;
        //}
    //}
    /** NOTE: this needs to be done this way dummy - leave the fence */
    while (atomicCAS((int*)&lock_, 0, 1) == 1) {
    }
    __threadfence();
}

//__device__ void
//DeviceMutex::try_lock(auto []fn() {}) {
    //do {
    //fn();
    //} while(atomicCAS()...)
//}

// 0000000000000000000000
// 0111100000000000000000 <- thread 1 - 4 active
// 1 <- takes lock - executes CS
// 1 <- release lock
// 2 <- ...
// ..
// 4 <- release
//
// every thread will have taken lock and gone through CS

__device__ void
DeviceMutex::unlock() {
    __threadfence();
    lock_ = 0;
}

OldDeviceMutex::OldDeviceMutex(bool shareable)
    : shareable_(shareable) {
}

__device__ void
OldDeviceMutex::lock() {
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
        while (atomicCAS((int*)&ctx_lock_, 0, 1) == 1)  {
            uint64_t time_now = clock64();
            int64_t wait_time = 100;
            while(time_now + wait_time > clock64()){
                __threadfence();
            }
            wait_time *= 2;
            wait_time = min(wait_time, 20000);
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
                // Performance is terrible. Backoff slightly helps.
                while (atomicCAS((int*)&ctx_lock_, 0, 1) == 1) {
                    uint64_t time_now = clock64();
                    int64_t wait_time = 100;
                    while(time_now + wait_time > clock64()){
                        __threadfence();
                    }
                    wait_time *= 2;
                    wait_time = min(wait_time, 20000);
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
OldDeviceMutex::unlock() {
    __threadfence();
    if (!shareable_) {
        return;
    }
    int num_threads_in_wv {wave_SZ()};

    if (get_flat_block_id() % WF_SIZE == lowerID()) {
        while (atomicCAS((int*)&ctx_lock_, 0, 1) == 1) {
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
