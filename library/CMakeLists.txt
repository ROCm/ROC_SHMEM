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
  set(
    CMAKE_BUILD_TYPE
    "Debug"
    CACHE
      STRING
        "build type: Release, Debug, RelWithDebInfo, MinSizeRel"
    FORCE
  )
  message(STATUS "CMAKE_BUILD_TYPE unspecified: using ${CMAKE_BUILD_TYPE}")
endif()

###############################################################################
# MODULE SEARCH PATH
###############################################################################
set(
  CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_SOURCE_DIR}/cmake/Modules"
  "/opt/rocm/hip/cmake"
  "/opt/rocm/rocclr/lib/cmake/rocclr"
)

###############################################################################
# PROJECT
###############################################################################
project(rocshmem VERSION 1.1.0 LANGUAGES CXX)

###############################################################################
# CONFIGURATION OPTIONS
###############################################################################
option(DEBUG "Enable debug trace" OFF)
option(PROFILE "Enable statistics and timing support" OFF)
option(USE_DC "Enable IB dynamically connected transport (DC)" OFF)
option(USE_HDP_MAP "Enable hdp map through hdp_map module" OFF)
option(USE_IPC "Enable IPC support (using HIP)" OFF)
option(USE_THREADS "Enable workgroup threads to share network queues" OFF)
option(USE_WF_COAL "Enable wavefront message coalescing" OFF)

configure_file(config.h.in config.h)

###############################################################################
# CREATE ROCSHMEM LIBRARY
###############################################################################
add_library(
  ${PROJECT_NAME}
  STATIC
  ""
)

###############################################################################
# INCLUDE DIRECTORIES
###############################################################################
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>                 # CONFIG.H
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> # PUBLIC HEADERS
    $<INSTALL_INTERFACE:include>
)

###############################################################################
# SPECIFY PUBLIC HEADER FILES
###############################################################################
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES
    PUBLIC_HEADER
      "${CMAKE_BINARY_DIR}/config.h;${CMAKE_CURRENT_SOURCE_DIR}/include/roc_shmem.hpp;${CMAKE_CURRENT_SOURCE_DIR}/include/debug.hpp"
)

###############################################################################
# SUBDIRECTORY TARGETS
###############################################################################
add_subdirectory(src)
add_subdirectory(docs)

###############################################################################
# HIP
###############################################################################
find_package(hip REQUIRED)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    hip::device
)

###############################################################################
# HSA-RUNTIME64
###############################################################################
find_package(hsa-runtime64 REQUIRED)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    hsa-runtime64::hsa-runtime64
)

###############################################################################
# IBVERBS
###############################################################################
find_package(Ibverbs REQUIRED)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    ${IBVERBS_INCLUDE_DIRS}
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    ${IBVERBS_LIBRARIES}
)

###############################################################################
# MPI
###############################################################################
find_package(MPI REQUIRED)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
    ${MPI_CXX_HEADER_DIR}
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC
    ${MPI_mpi_LIBRARY}
    ${MPI_mpicxx_LIBRARY}
)

###############################################################################
# INSTALL
###############################################################################
include(GNUInstallDirs)

# Specify layout within the build directory
set(
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(
  CMAKE_LIBRARY_OUTPUT_DIRECTORY
  ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
)
set(
  CMAKE_RUNTIME_OUTPUT_DIRECTORY
  ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
)

# Offer alternatives for default installation subdirectories
set(
  INSTALL_LIBDIR
    ${CMAKE_INSTALL_LIBDIR} CACHE PATH
    "Installation directory for libraries"
)
set(
  INSTALL_BINDIR
    ${CMAKE_INSTALL_BINDIR} CACHE PATH
    "Installation directory for executables"
)
set(
  INSTALL_INCLUDEDIR
    ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH
    "Installation directory for header files"
)
set(
  DEF_INSTALL_CMAKEDIR
  share/cmake/${PROJECT_NAME}
)
set(
  INSTALL_CMAKEDIR
    ${DEF_INSTALL_CMAKEDIR} CACHE PATH
    "Installation directory for CMake files"
)

install(
  TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
    ARCHIVE
      DESTINATION ${INSTALL_LIBDIR}
      COMPONENT lib
    LIBRARY
      DESTINATION ${INSTALL_LIBDIR}
      COMPONENT lib
    PUBLIC_HEADER
      DESTINATION ${INSTALL_INCLUDEDIR}
      COMPONENT dev
    RUNTIME
      DESTINATION ${INSTALL_BINDIR}
      COMPONENT bin
)

install(
  EXPORT
    ${PROJECT_NAME}Targets
  FILE
    ${PROJECT_NAME}Targets.cmake
  NAMESPACE
    ${PROJECT_NAME}::
  DESTINATION
    ${INSTALL_CMAKEDIR}
  COMPONENT
    dev
)

###############################################################################
# PACKAGE
###############################################################################
include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  VERSION
    ${PROJECT_VERSION}
  COMPATIBILITY
    SameMajorVersion
)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION
    ${INSTALL_CMAKEDIR}
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION
    ${INSTALL_CMAKEDIR}
)

###############################################################################
# DEPENDENCIES
###############################################################################
include(cmake/rocmAnalyzers.cmake)

###############################################################################
# LINTER
###############################################################################
#rocm_enable_clang_tidy(
    #CHECKS
        #*
    #ALL
#)
#rocm_clang_tidy_check(${PROJECT_NAME})

###############################################################################
# ANALYZERS
###############################################################################
#rocm_enable_cppcheck(
    #CHECKS
        #warning
        #style
        #performance
        #portability
    #FORCE
    #INCONCLUSIVE
        #${CMAKE_CURRENT_SOURCE_DIR}/cppcheck.rules
    #SOURCES
        #src/
    #INCLUDE
        #${CMAKE_CURRENT_SOURCE_DIR}/include
    #DEFINE
        #CPPCHECK=1
        #__device__=
        #__host__=
#)

###############################################################################
# INCLUDE WHAT YOU USE
###############################################################################
# Requires installing Debian iwyu package.
# Misses hip headers and some standard clang headers since the stock
# Debian package search in the wrong location for header files.
# TODO: create a toggle for this option
#set_target_properties(
  #${PROJECT_NAME}
  #PROPERTIES
    #CXX_INCLUDE_WHAT_YOU_USE
      #"/usr/bin/iwyu"
#)

###############################################################################
# LINK WHAT YOU USE
###############################################################################
# The option only works on shared (not static) library build.
# TODO: create a toggle for this option
#set_target_properties(
  #${PROJECT_NAME}
  #PROPERTIES
    #LINK_WHAT_YOU_USE
      #TRUE
#)

###############################################################################
# SANITIZERS
###############################################################################
# The santizers do not work properly right now. Suspect that this occurs
# because the hip compiler is a derivative of llvm and not actually a
# normal compiler.
# TODO: try to get them working
#set(USE_SANITIZER Address)
#set(USE_SANITIZER Memory)
#set(USE_SANITIZER MemoryWithOrigins)
#set(USE_SANITIZER Undefined)
#set(USE_SANITIZER Thread)
#set(USE_SANITIZER Leak)
#set(USE_SANITIZER "Address;Undefined")

#include(cmake/cmake-scripts/sanitizers.cmake)

###############################################################################
# CODE COVERAGE
###############################################################################
#option(
  #CODE_COVERAGE
  #"Builds targets with code coverage instrumentation."
  #ON
#)

#include(cmake/cmake-scripts/code-coverage.cmake)

#target_code_coverage(${PROJECT_NAME})

###############################################################################
# DEPENDENCY GRAPH GENERATION
###############################################################################
# Create custom build target to show library dependencies
#include(cmake/cmake-scripts/dependency-graph.cmake)
#gen_dep_graph(pdf)

###############################################################################
# MEMCHECK
###############################################################################
# valgrind memcheck

###############################################################################
# ABI CHECKER
###############################################################################
# ABI compliance checker (ABICC)
# github.com/lvc/abi-compliance-checker/releases
