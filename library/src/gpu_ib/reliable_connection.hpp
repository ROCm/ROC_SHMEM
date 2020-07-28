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

#ifndef __RELIABLE_CONNECTION_HPP__
#define __RELIABLE_CONNECTION_HPP__

#include "connection.hpp"

class ReliableConnection : public Connection
{
  public:
    explicit ReliableConnection(GPUIBBackend *backend);
    virtual ~ReliableConnection() override;

    virtual roc_shmem_status_t get_remote_conn(int &remote_conn) override;

    virtual void post_wqes() override;

    virtual roc_shmem_status_t
    initialize_rkey_handle(uint32_t **heap_rkey_handle, ibv_mr *mr) override;

  private:
    virtual InitQPState initqp(uint8_t port) override;

    virtual RtrState rtr(dest_info_t *dest, uint8_t port) override;

    virtual RtsState rts(dest_info_t *dest) override;

    virtual QPInitAttr qpattr(ibv_qp_cap cap) override;

    virtual roc_shmem_status_t create_qps_1() override;

    virtual roc_shmem_status_t
    create_qps_2(int port, int my_rank,
                 ibv_port_attr *ib_port_att) override;

    virtual roc_shmem_status_t
    create_qps_3(int port, ibv_qp *qp, int offset,
                 ibv_port_attr *ib_port_att) override;

    virtual roc_shmem_status_t allocate_dynamic_members(int num_wg) override;

    virtual roc_shmem_status_t free_dynamic_members() override;

    virtual roc_shmem_status_t
    initialize_1(int port,
                 int num_wg) override;

    virtual void
    initialize_wr_fields(ibv_exp_send_wr &wr,
                         ibv_ah *ah, int dc_key) override;

    virtual int
    get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx) override;

    std::vector<dest_info_t> all_qp;
};

#endif // __RELIABLE_CONNECTION_HPP__
