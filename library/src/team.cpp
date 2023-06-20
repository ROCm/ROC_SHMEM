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

#include "src/team.hpp"

#include <cmath>

#include "include/roc_shmem.hpp"
#include "src/backend_bc.hpp"
#include "src/util.hpp"

namespace rocshmem {

roc_shmem_team_t ROC_SHMEM_TEAM_WORLD;

__host__ __device__ Team* get_internal_team(roc_shmem_team_t team) {
  return reinterpret_cast<Team*>(team);
}

GPUIBTeam* get_internal_gpu_ib_team(roc_shmem_team_t team) {
  return reinterpret_cast<GPUIBTeam*>(team);
}

ROTeam* get_internal_ro_team(roc_shmem_team_t team) {
  return reinterpret_cast<ROTeam*>(team);
}

__host__ __device__ int team_translate_pe(roc_shmem_team_t src_team, int src_pe,
                                          roc_shmem_team_t dst_team) {
  if (src_team == ROC_SHMEM_TEAM_INVALID ||
      dst_team == ROC_SHMEM_TEAM_INVALID) {
    return -1;
  }

  Team* src_team_obj{get_internal_team(src_team)};
  Team* dst_team_obj{get_internal_team(dst_team)};
  int src_pe_in_world{src_team_obj->get_pe_in_world(src_pe)};
  int dst_pe{dst_team_obj->get_pe_in_my_team(src_pe_in_world)};

  return dst_pe;
}

__host__ __device__ TeamInfo::TeamInfo(Team* _parent_team, int _pe_start,
                                       int _stride, int _size)
    : parent_team(_parent_team),
      pe_start(_pe_start),
      stride(_stride),
      size(_size) {
  log_stride = log2(stride);
}

__host__ Team::Team(Backend* handle, TeamInfo* team_info_wrt_parent,
                    TeamInfo* team_info_wrt_world, int _num_pes, int _my_pe,
                    MPI_Comm _mpi_comm)
    : world_size(handle->getNumPEs()),
      my_pe_in_world(handle->getMyPE()),
      tinfo_wrt_parent(team_info_wrt_parent),
      tinfo_wrt_world(team_info_wrt_world),
      num_pes(_num_pes),
      my_pe(_my_pe),
      mpi_comm(_mpi_comm) {}

__host__ __device__ int Team::get_pe_in_world(int pe) {
  int pe_start{tinfo_wrt_world->pe_start};
  int stride{tinfo_wrt_world->stride};

  return pe_start + stride * pe;
}

__host__ __device__ int Team::get_pe_in_my_team(int pe_in_world) {
  int pe_start{tinfo_wrt_world->pe_start};
  int stride{tinfo_wrt_world->stride};

  if (pe_in_world < pe_start) {
    return -1;  // Outside the start of the range
  }

  if ((pe_in_world - pe_start) % stride) {
    return -1;  // Not a multiple of stride
  }

  int pe_in_my_team{(pe_in_world - pe_start) / stride};
  if (pe_in_my_team >= num_pes) {
    return -1;  // Outside the end of the range
  }

  return pe_in_my_team;
}

__host__ Team::~Team() {}

}  // namespace rocshmem
