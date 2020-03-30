# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

AM_CPPFLAGS = -I$(top_srcdir)/include

lib_LIBRARIES = libroc_shmem.a

ACLOCAL_AMFLAGS = -Iconfig
AM_CXXFLAGS = -g -Wall -fno-stack-protector -Wno-format-security

include_HEADERS = $(top_srcdir)/include/roc_shmem.hpp

libroc_shmem_a_CXXFLAGS = $(AM_CXXFLAGS)
libroc_shmem_a_SOURCES = roc_shmem.cpp \
    util.cpp \
    roc_shmem_gpu.cpp \
    reverse_offload/ro_net.cpp \
    reverse_offload/ro_net_gpu.cpp \
    reverse_offload/ro_net_gpu_templates.hpp \
    reverse_offload/transport.cpp \
    reverse_offload/transport.hpp \
    reverse_offload/ro_net_internal.hpp \
    backend.hpp \
    backend.cpp \
    $(top_srcdir)/include/roc_shmem.hpp \
    hdp_helper.hpp \
    gpu_ib/gpu_ib.cpp \
    gpu_ib/gpu_ib_gpu.cpp \
    gpu_ib/gpu_ib_gpu_coll.cpp \
    gpu_ib/gpu_ib_gpu_templates.hpp \
    gpu_ib/gpu_ib_internal.hpp \
    gpu_ib/rtn.cpp \
    gpu_ib/rtn.hpp \
    gpu_ib/rtn_internal.hpp \
    gpu_ib/rtn_help.cpp \
    gpu_ib/rtn_rocm.cpp \
    gpu_ib/rtn_rocm.hpp \
    gpu_ib/thread_policy.hpp \
    gpu_ib/thread_policy.cpp \
    gpu_ib/rtn_gpu.cpp
