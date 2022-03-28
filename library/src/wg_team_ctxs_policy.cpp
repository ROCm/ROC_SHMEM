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

#include "wg_state.hpp"
#include "wg_team_ctxs_policy.hpp"

namespace rocshmem {

__device__
WGTeamInfo::WGTeamInfo(Team* team,
                       TeamInfo* team_info)
    : team_(team),
      ref_count_(0),
      team_info_(nullptr,
                 team_info->pe_start,
                 team_info->stride,
                 team_info->size) {
}

__device__ void
WGTeamCtxsPolicy::init(int max_num_teams) {
    max_num_teams_ = max_num_teams;

    size_t size {max_num_teams_ * sizeof(WGTeamInfo)};
    char* space {WGState::instance()->allocateDynamicShared(size)};
    wg_tinfo_pool_ = reinterpret_cast<WGTeamInfo*>(space);

    if (is_thread_zero_in_block()) {
        for (int i {0}; i < max_num_teams_; i++) {
            new (&(wg_tinfo_pool_[i])) WGTeamInfo();
        }
    }

    __syncthreads();
}

__device__ WGTeamInfo*
WGTeamCtxsPolicy::get_team_info(Team* team)
{
    __shared__ WGTeamInfo* wg_team_info_p;

    if (is_thread_zero_in_block()) {
        wg_team_info_p = find_team_in_pool(team);

        if (!wg_team_info_p) {
            auto index {find_available_pool_entry()};
            wg_team_info_p = &(wg_tinfo_pool_[index]);
            new (wg_team_info_p) WGTeamInfo(team,
                                            team->tinfo_wrt_world);

        }

        (wg_team_info_p->ref_count_)++;
    }

    __syncthreads();

    return wg_team_info_p;
}

__device__ size_t
WGTeamCtxsPolicy::find_available_pool_entry() {
    for (size_t i {0}; i < max_num_teams_; i++) {
        if (wg_tinfo_pool_[i].ref_count_ == 0) {
            return i;
        }
    }
    /* Entry should have been available; consider this as an error. */
    assert(false);
    return -1;
}

__device__ WGTeamInfo*
WGTeamCtxsPolicy::find_team_in_pool(Team* team) {
    for (int i {0}; i < max_num_teams_; i++) {
        if (wg_tinfo_pool_[i].ref_count_ == 0) {
            continue;
        }
        if (team == wg_tinfo_pool_[i].team_) {
            return &wg_tinfo_pool_[i];
        }
    }
    return nullptr;
}

__device__ void
WGTeamCtxsPolicy::remove_team_info(WGTeamInfo* wg_team_info)
{
    if (is_thread_zero_in_block()) {
        /* Object should've been created here first before removing it */
        assert(wg_team_info->ref_count_ > 0);

        (wg_team_info->ref_count_)--;

        if (wg_team_info->ref_count_ == 0) {
            new (wg_team_info) WGTeamInfo();
        }
    }

    __syncthreads();
}

}  // namespace rocshmem
