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

#ifndef ROCSHMEM_LIBRARY_SRC_WGSTATE_HPP
#define ROCSHMEM_LIBRARY_SRC_WGSTATE_HPP

#include "util.hpp"
#include "wg_team_ctxs_policy.hpp"

namespace rocshmem {

class Context;

class WGState
{
    /*
     * Ptr to current base of dynamic shared region for this work-group.
     */
    char *dynamicPtr = nullptr;

    /*
     * Global buffer ID that I have checked out from Backend.  Used to index
     * into per-work-group global memory resources.
     */
    uint32_t buffer_id = 0;

    /*
     * Single private context associated with this work-group.
     */
    Context * wg_ctx = nullptr;

    /*
     * Reserve an index for per-wg global buffers from the bufferToken pool.
     */
    __device__ unsigned int reserve_wg_buffers(int num_buffers);

    __device__ WGState();

  public:
    /*
     * Create WGState for calling work-group at the start of the dynamic
     * segment.
     */
    static __device__ void create();

    /*
     * Get WGState associated with the calling work-group
     */
    static __device__ WGState * instance();

    /*
     * Return any global buffers reserved by this work-group.
     */
    __device__ void return_buffers();

    /*
     * Allocate memory from dynamic shared segment.  Must be called as a
     * work-group collective.
     */
    __device__ char *allocateDynamicShared(size_t size);

    /*
     * Get the ID assocated with global buffers reserved by this work-group.
     */
    __device__ int get_global_buffer_id() const;

    /*
     * Set the private context for this work-group.
     */
    __device__ void set_private_ctx(Context *ctx) { wg_ctx = ctx; }

    /*
     * Get the private context for this work-group.
     */
    __device__ Context * get_private_ctx() const { return wg_ctx; }

    /**
     * The policy to support team contexts
     */
    WGTeamCtxsPolicy team_ctxs_policy {};
};

/*
 * TODO: There are a number of backend-specific buffers that are per-wg rather
 * than per Context.  We should move them into derived classes of WGState,
 * because WGState owns the index to these buffers (buffer_id) and because
 * we don't need to duplicate the pointers across all contexts. In the case
 * of shared contexts, this is pretty much a requirement, since each WG
 * using the shared Context needs to supply his own buffers rather than getting
 * them from Context.
 */

}  // namespace rocshmem

#endif  // ROCSHMEM_LIBRARY_SRC_WGSTATE_HPP
