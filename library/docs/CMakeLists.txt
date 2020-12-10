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

find_package(Doxygen)
if (DOXYGEN_FOUND)
   set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
   set(DOXYGEN_SOURCE_BROWSER YES)
   set(DOXYGEN_EXTRACT_PRIVATE YES)
   set(DOXYGEN_MACRO_EXPANSION YES)
   set(DOXYGEN_GENERATE_LATEX YES)
   set(DOXYGEN_USE_PDFLATEX YES)

   configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
   doxygen_add_docs(
      docs
      ${PROJECT_SOURCE_DIR}
      COMMENT "Generate man pages"
      MACRO_EXPANSION YES
   )
endif()
