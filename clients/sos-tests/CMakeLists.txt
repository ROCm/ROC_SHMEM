##############################################################################
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

cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /opt/rocm/bin/hipcc)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

###############################################################################
# DEFAULT BUILD TYPE
###############################################################################
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "CMAKE_BUILD_TYPE unspecified: generating Release build")

  set(
    CMAKE_BUILD_TYPE
    "Release"
    CACHE
      STRING
        "build type: Release, Debug, RelWithDebInfo, MinSizeRel"
    FORCE
  )
endif()

###############################################################################
# PROJECT
###############################################################################
project(host_facing_sos_tests LANGUAGES CXX)

###############################################################################
# DEPENDENCIES
###############################################################################
find_package(hip REQUIRED)
find_package(rocshmem CONFIG REQUIRED)

###############################################################################
# SOURCES
###############################################################################
set(TEST_SOURCES
      hello.cpp
      barrier.cpp
      global_exit.cpp
      asym_alloc.cpp
      shmalloc.cpp
      bcast.cpp
      broadcast_active_set.cpp
      bcast_flood.cpp
      to_all.cpp
      reduce_in_place.cpp
      reduce_active_set.cpp
      max_reduction.cpp
      big_reduction.cpp
      cxx_test_shmem_p.cpp
      cxx_test_shmem_g.cpp
      put1.cpp
      get1.cpp
      put_nbi.cpp
      get_nbi.cpp
      bigput.cpp
      bigget.cpp
      waituntil.cpp
      cxx_test_shmem_wait_until.cpp
      shmem_test.cpp
      cxx_test_shmem_test.cpp
      atomic_inc.cpp
      cxx_test_shmem_atomic_add.cpp
      cxx_test_shmem_atomic_fetch.cpp
      cxx_test_shmem_atomic_cswap.cpp
      cxx_test_shmem_atomic_inc.cpp
      lfinc.cpp
      query_thread.cpp
      threading.cpp
      thread_wait.cpp
      mt_contention.cpp
      mt_a2a.cpp
      micro_unit_shmem.cpp
      circular_shift.cpp
      pi.cpp
      ping.cpp
      sping.cpp
)
# Automatic alternative (not reocommended 'cause cmake won't detect new files):
# file( GLOB TEST_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp )

foreach( testsourcefile ${TEST_SOURCES} )
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_include_directories(
        ${testname}
        PRIVATE rocshmem::rocshmem
    )
    target_link_libraries(
        ${testname}
        PRIVATE
          rocshmem::rocshmem
          hip::host
          -fgpu-rdc
    )
endforeach( testsourcefile ${APP_SOURCES} )
