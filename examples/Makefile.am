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

TESTS = init get put get_nbi put_nbi get_swarm reduction
check_PROGRAMS = $(TESTS)

LOG_COMPILER = $(srcdir)/driver.sh "$(LAUNCH_CMD)"

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_LDFLAGS = -L$(top_builddir)/src -lro_net

init_SOURCES = test_driver.cpp tests.hpp util.cpp util.h tests.hpp
get_SOURCES = test_driver.cpp tests.hpp util.cpp util.h tests.hpp
put_SOURCES = test_driver.cpp tests.hpp util.cpp util.h tests.hpp
put_nbi_SOURCES = test_driver.cpp tests.hpp util.cpp util.h tests.hpp
get_nbi_SOURCES = test_driver.cpp tests.hpp util.cpp util.h tests.hpp
get_swarm_SOURCES = test_driver.cpp tests.hpp util.cpp util.h tests.hpp
reduction_SOURCES = test_driver.cpp tests.hpp util.cpp util.h tests.hpp
