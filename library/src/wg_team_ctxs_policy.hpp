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

#ifndef ROCSHMEM_LIBRARY_SRC_WG_TEAM_CTXS_POLICY_HPP
#define ROCSHMEM_LIBRARY_SRC_WG_TEAM_CTXS_POLICY_HPP

#include "team.hpp"

namespace rocshmem {

class WGTeamInfo
{
  public:
    /**
     * @brief Constructor with default members
     */
    __device__
    WGTeamInfo() = default;

    /**
     * @brief Constructor with initialized members
     *
     * @param[in] team pointer used to track team info
     * @param[in] team_info information about participating PEs
     */
    __device__
    WGTeamInfo(Team* team,
               TeamInfo* team_info);

    /**
     * @brief Retrieve a copy of internal team information
     *
     * @return TeamInfo shallow copy
     */
    __device__ TeamInfo
    get() {
        return team_info_;
    }

  private:
    /*
     * Policy class manipulates a pool of this object type.
     */
    friend class WGTeamCtxsPolicy;

    /**
     * @brief Tracks how many teams are using a pool entry
     */
    size_t ref_count_ {0};

    /**
     * @brief Used as an identifier for this team info
     */
    Team* team_ {nullptr};

    /**
     * @brief Contains information about participating PEs in this team
     */
    TeamInfo team_info_ {};
};

class WGTeamCtxsPolicy
{
  public:
    /**
     * @brief Initialize the resources for this policy for max_num_teams.
     *
     * This method allocates space in LDS to store teams information
     * and initializes the team object in each entry.
     *
     * @param[in] max_num_teams protects library from over-allocating teams
     *
     * @return void
     */
    __device__ void
    init(int max_num_teams);

    /**
     * @brief Get the team info for a particular team.
     *
     * If the team is not found in any of the available entries, this
     * method will find the first available entry and build a new team
     * info object.
     *
     * @param[in] team_obj pointer needed to retrieve workgroup's team info
     *
     * @return Handle to internal team info object of workgroup
     */
    __device__ WGTeamInfo*
    get_team_info(Team* team_obj);

    /**
     * @brief Remove the team info for a particular team.
     *
     * @param[in] wg_team_info pointer to drop from tracking
     *
     * @return void
     */
    __device__ void
    remove_team_info(WGTeamInfo* wg_team_info);

  private:
    /**
     * @brief Finds an entry in the wg_tinfo_pool (c-array)
     *
     * @param[in] team pointer used to identify team
     *
     * @return either null OR team info pointer
     */
    __device__ WGTeamInfo*
    find_team_in_pool(Team* team);

    /**
     * @brief Finds an unused entry in the wg_tinfo_pool (c-array)
     *
     * @return -1 (an error) or the entry position
     *
     * @note this method may trip an assert in debug builds if an error
     * is encountered
     */
    __device__ size_t
    find_available_pool_entry();

    /**
     * @brief pointer to the pool of team infos
     */
    WGTeamInfo* wg_tinfo_pool_ {nullptr};

    /**
     * @brief bounds check to protect against too many teams created
     */
    int max_num_teams_ {-1};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_WG_TEAM_CTXS_POLICY_HPP
