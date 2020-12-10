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

#ifndef __DYNAMIC_CONNECTION_HPP__
#define __DYNAMIC_CONNECTION_HPP__

#include "connection.hpp"

class DynamicConnection : public Connection
{
  public:
    explicit DynamicConnection(GPUIBBackend *backend);
    virtual ~DynamicConnection() override;

    virtual Status get_remote_conn(int &remote_conn) override;

    virtual void post_wqes() override;

    virtual Status
    initialize_rkey_handle(uint32_t **heap_rkey_handle, ibv_mr *mr) override;
    void free_rkey_handle(uint32_t *heap_rkey_handle) override;


    uint32_t * get_vec_dct_num() const { return vec_dct_num; }

    uint16_t * get_vec_lids() const { return vec_lids; }

  private:
    virtual InitQPState initqp(uint8_t port) override;

    virtual RtrState rtr(dest_info_t *dest, uint8_t port) override;

    virtual RtsState rts(dest_info_t *dest) override;

    virtual QPInitAttr qpattr(ibv_qp_cap cap) override;

    Status connect_dci(ibv_qp *qp, uint8_t port);

    Status create_dct(int32_t &dct_num,
                                  ibv_cq *cq,
                                  ibv_srq *srq,
                                  uint8_t port);

    bool transport_enabled();

    bool status_good(ibv_exp_dct *dct);

    ibv_exp_dct_init_attr dct_init_attr(ibv_cq *cq, ibv_srq *srq,
                                        uint8_t port) const;

    virtual Status
    create_qps_1() override;

    virtual Status
    create_qps_2(int port, int my_rank,
                 ibv_port_attr *ib_port_att) override;

    virtual Status
    create_qps_3(int port, ibv_qp *qp, int offset,
                 ibv_port_attr *ib_port_att) override;

    virtual Status allocate_dynamic_members(int num_wg) override;

    virtual Status free_dynamic_members() override;

    virtual Status
    initialize_1(int port, int num_wg) override;

    virtual void
    initialize_wr_fields(ibv_exp_send_wr &wr,
                         ibv_ah *ah, int dc_key) override;

    virtual int
    get_sq_dv_offset(int pe_idx, int num_qps, int wg_idx) override;

    int num_dcis = 1;
    int num_dct = 1;

    static constexpr int DC_IB_KEY = 0x1ee7a330;

    uint32_t *dcts_num = nullptr;
    uint16_t *lids = nullptr;

    ibv_ah *ah = nullptr;
    ibv_srq *srq = nullptr;
    ibv_cq *dct_cq = nullptr;
    uint32_t *vec_dct_num = nullptr;
    uint16_t *vec_lids = nullptr;
};

#endif // __DYNAMIC_CONNECTION_HPP__
