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

#include "memory_allocator.hpp"

#include <cassert>

namespace rocshmem {

MemoryAllocator::MemoryAllocator(hipError_t (*hip_alloc_fn)(void**, size_t, unsigned),
                                 hipError_t (*hip_free_fn)(void*))
    : _hip_alloc_finegrained(hip_alloc_fn), _hip_free(hip_free_fn)
{
}

MemoryAllocator::MemoryAllocator(hipError_t (*hip_alloc_fn)(void**, size_t),
                                 hipError_t (*hip_free_fn)(void*))
    : _hip_alloc(hip_alloc_fn), _hip_free(hip_free_fn)
{
}

MemoryAllocator::MemoryAllocator(std::function<void*(size_t)> alloc_fn,
                                 std::function<void(void*)> free_fn)
    : _alloc(alloc_fn), _free(free_fn)
{
}

void
MemoryAllocator::allocate(void** void_ptr, size_t size)
{
    assert(void_ptr);
    assert(size >= 0);

    if (_alloc) {
        *(reinterpret_cast<char**>(void_ptr)) =
            reinterpret_cast<char*>(_alloc(size));
        return;
    }
    if (_hip_alloc) {
        _hip_return_value = _hip_alloc(void_ptr, size);
        assert(_hip_return_value == hipSuccess);
        return;
    }
    if (_hip_alloc_finegrained) {
        unsigned flags = hipDeviceMallocFinegrained;
        _hip_return_value = _hip_alloc_finegrained(void_ptr,
                                                   size,
                                                   flags);
        assert(_hip_return_value == hipSuccess);
        return;
    }
}

void
MemoryAllocator::deallocate(void* ptr)
{
    if (!ptr) {
        return;
    }
    if (_free) {
        _free(ptr);
        return;
    }
    if (_hip_free) {
        _hip_return_value = _hip_free(ptr);
        assert(_hip_return_value == hipSuccess);
        return;
    }
}

}  // namespace rocshmem
