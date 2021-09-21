###############################################################################
# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

###############################################################################
# ADD ROCSHMEM TARGET FOR FILES IN CURRENT DIRECTORY
###############################################################################
target_sources(
  ${PROJECT_NAME}
  PRIVATE
    util.cpp
    roc_shmem_gpu.cpp
    context.cpp
    hdp_policy.cpp
    roc_shmem.cpp
    wf_coal_policy.cpp
    wg_state.cpp
    backend.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(
  ${PROJECT_NAME}
  PUBLIC
    -fgpu-rdc
)

#target_link_options(
  #${PROJECT_NAME}
  #PUBLIC
    #--hip-link
#)

###############################################################################
# ROCSHMEM TARGET FOR BACKENDS
###############################################################################
add_subdirectory(gpu_ib)
add_subdirectory(reverse_offload)
add_subdirectory(host)
