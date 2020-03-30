#! /usr/bin/python
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

import os

suffixes = [ '.cpp', '.hpp', '.c', '.h' ]
directories = [ 'src', 'include' ]

def oksuffix(f):
    for s in suffixes:
        if f.endswith(s):
            return True
    return False

def try_index_dir(directory):
    for dirpath,subdirs,files in os.walk(os.path.join(cwd, directory)):
        okfiles = [f for f in files if oksuffix(f)]
        if okfiles:
            print >> file_list, \
                  '\n'.join([os.path.join(dirpath, f) for f in okfiles])


file_list = file('cscope.files', 'w')
cwd = os.getcwd()
for d in directories:
    try_index_dir(d)
file_list.close()

os.system("cscope -b")
