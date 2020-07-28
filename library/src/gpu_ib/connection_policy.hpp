/******************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef CONNECTIONPOLICY_H
#define CONNECTIONPOLICY_H

#include "config.h"

struct mlx5_base_av;
/*
 * CRTP base class for connection type
 */
template <typename Derived>
class ConnectionBase
{
  public:
   /*
    * Control segment WQE offset imposed by this connection type.
    */
    __device__ int wqeCntrlOffset()
        { return static_cast<Derived *>(this)->wqeCntrlOffsetImpl(); }

   /*
    * Whether or not we need to force PE-level divergence when posting for
    * this connection type.
    */
    __device__ bool forcePostDivergence()
        { return static_cast<Derived *>(this)->forcePostDivergenceImpl(); }

   /*
    * Number of WQEs produced by this connection type for the given opcode.
    */
    __device__ uint32_t getNumWqes(uint8_t opcode)
        { return static_cast<Derived *>(this)->getNumWqesImpl(opcode); }

   /*
    * Updates the connection-specific segment in the SQ.
    */
    __device__ bool updateConnectionSegment(mlx5_base_av *wqe, int pe)
        { return static_cast<Derived *>(this)->
            updateConnectionSegmentImpl(wqe, pe); }

   /*
    * Set the rkey based on this connection type.
    */
    __device__ void setRkey(uint32_t &rkey, int pe)
        { return static_cast<Derived *>(this)->setRkeyImpl(rkey, pe); }
};

class Connection;

/*
 * Connection policy corresponding to an RC connection type.
 */
class RCConnectionImpl : public ConnectionBase<RCConnectionImpl>
{
  public:
    RCConnectionImpl(Connection *conn, uint32_t *_vec_rkey);

    __device__ int wqeCntrlOffsetImpl() { return 0; }

    __device__ bool forcePostDivergenceImpl() { return true; }

    __device__ uint32_t getNumWqesImpl(uint8_t opcode);

    __device__ bool updateConnectionSegmentImpl(mlx5_base_av *wqe, int pe);

    __device__ void setRkeyImpl(uint32_t &rkey, int pe);
};

/*
 * Connection policy corresponding to a DC connection type.
 */
class DCConnectionImpl : public ConnectionBase<DCConnectionImpl>
{
    uint32_t *vec_dct_num = nullptr;

    uint32_t *vec_rkey = nullptr;

    uint16_t *vec_lids = nullptr;

  public:
    DCConnectionImpl(Connection *conn, uint32_t *_vec_rkey);

    __device__ int wqeCntrlOffsetImpl() { return 1; }

    __device__ bool forcePostDivergenceImpl() { return false; }

    __device__ uint32_t getNumWqesImpl(uint8_t opcode);

    __device__ bool updateConnectionSegmentImpl(mlx5_base_av *wqe, int pe);

    __device__ void setRkeyImpl(uint32_t &rkey, int pe);
};

/*
 * Select which one of our connection policies to use at compile time.
 */
#ifdef USE_DC
typedef DCConnectionImpl ConnectionImpl;
#else
typedef RCConnectionImpl ConnectionImpl;
#endif

#endif //CONNECTIONPOLICY_H
