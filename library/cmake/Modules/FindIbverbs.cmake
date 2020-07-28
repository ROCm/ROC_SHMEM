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

find_package(PkgConfig REQUIRED QUIET)
pkg_check_modules(PC_IBVERBS QUIET libibverbs)

find_path(
    IBVERBS_INCLUDE_DIR infiniband/verbs.h
    HINTS ${PC_IBVERBS_INCLUDEDIR} ${PC_IBVERBS_INCLUDE_DIRS}
    PATH_SUFFIXES include
)

find_library(
    IBVERBS_LIBRARY
    NAMES ibverbs libibverbs
    HINTS ${PC_IBVERBS_LIBDIR} ${PC_IBVERBS_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
)

find_library(
    MLX5_LIBRARY
    NAMES mlx5 libmlx5
    HINTS ${PC_IBVERBS_LIBDIR} ${PC_IBVERBS_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib64
)

set(
    IBVERBS_LIBRARIES
    ${IBVERBS_LIBRARY} ${MLX5_LIBRARY}
    CACHE INTERNAL ""
)

set(
    IBVERBS_INCLUDE_DIRS
    ${IBVERBS_INCLUDE_DIR}
    CACHE INTERNAL ""
)

find_package_handle_standard_args(
    Ibverbs DEFAULT_MSG IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR
)

mark_as_advanced(IBVERBS_LIBRARY IBVERBS_INCLUDE_DIR)
