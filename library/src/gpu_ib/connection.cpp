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

#include "src/gpu_ib/connection.hpp"

#include <mpi.h>

#include <mutex>  // NOLINT(build/c++11)
#include <vector>

#include "src/gpu_ib/backend_ib.hpp"
#include "src/gpu_ib/queue_pair.hpp"
#include "src/util.hpp"

namespace rocshmem {

int Connection::use_gpu_mem = 0;
int Connection::coherent_cq = 0;

Connection::Connection(GPUIBBackend* b, int k) : backend(b), key_offset(k) {
  char* value = nullptr;

  if ((value = getenv("ROC_SHMEM_USE_IB_HCA"))) {
    requested_dev = value;
  }

  if ((value = getenv("ROC_SHMEM_SQ_SIZE"))) {
    sq_size = atoi(value);
  }

  if ((value = getenv("ROC_SHMEM_USE_CQ_GPU_MEM")) != nullptr) {
    cq_use_gpu_mem = atoi(value);
  }

  if ((value = getenv("ROC_SHMEM_USE_SQ_GPU_MEM")) != nullptr) {
    sq_use_gpu_mem = atoi(value);
  }
}

Connection::~Connection() { delete ib_state; }

void Connection::reg_mr(void* ptr, size_t size, ibv_mr** mr, bool managed) {
  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
               IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  if (managed) {
    access |= IBV_ACCESS_ON_DEMAND;
  }

  *mr = ibv_reg_mr(ib_state->pd, ptr, size, access);

  if (*mr == nullptr) {
    abort();
  }
}

unsigned Connection::total_number_connections() {
  int connections;
  get_remote_conn(&connections);
  return backend->num_blocks_ * connections;
}

void Connection::initialize(int num_block) {
  allocate_dynamic_members(num_block);

  int ib_devices{0};
  dev_list = ibv_get_device_list(&ib_devices);
  if (dev_list == nullptr) {
    abort();
  }

  struct ibv_device* ib_dev = dev_list[0];
  if (requested_dev != nullptr) {
    for (int i = 0; i < ib_devices; i++) {
      const char* select_dev = ibv_get_device_name(dev_list[i]);
      if (strstr(select_dev, requested_dev) != nullptr) {
        ib_dev = dev_list[i];
        break;
      }
    }
  }

  uint8_t port = 1;
  ib_init(ib_dev, port);

  int hip_dev_id = 0;
  CHECK_HIP(hipGetDevice(&hip_dev_id));

  int ib_fork_err = ibv_fork_init();
  if (ib_fork_err != 0) printf("error: ibv_fork_init failed \n");

  sq_post_dv = static_cast<sq_post_dv_t*>(
      malloc(sizeof(sq_post_dv_t) * total_number_connections()));

  if (sq_post_dv == nullptr) {
    abort();
  }

  create_qps(port, backend->my_pe, &ib_state->portinfo);
  initialize_1(port, num_block);

  MPI_Barrier(backend->thread_comm);
  free_dynamic_members();
}

void Connection::finalize() {
  ibv_free_device_list(dev_list);

  int ret = ibv_dereg_mr(backend->networkImpl.heap_mr);
  if (ret) {
    abort();
  }
  // comment until rocm 4.5
  // ibv_dereg_mr(backend->networkImpl.hdp_mr);
  ibv_dereg_mr(backend->networkImpl.mr);
}

void Connection::ib_init(struct ibv_device* ib_dev, uint8_t port) {
  ib_state = new ib_state_t;
  if (!ib_state) {
    abort();
  }

  ib_state->context = ibv_open_device(ib_dev);
  if (!ib_state->context) {
    delete ib_state;
    abort();
  }

  ib_state->pd = ibv_alloc_pd(ib_state->context);
  if (!ib_state->pd) {
    delete ib_state;
    abort();
  }

  ibv_parent_domain_init_attr pattr;
  init_parent_domain_attr(&pattr);
  ib_state->pd = ibv_alloc_parent_domain(ib_state->context, &pattr);

  ibv_query_port(ib_state->context, port, &ib_state->portinfo);
}

template <typename StateType>
void Connection::try_to_modify_qp(ibv_qp* qp, StateType state) {
  ibv_modify_qp(qp, &state.exp_qp_attr, state.exp_attr_mask);
}

void Connection::init_qp_status(ibv_qp* qp, uint8_t port) {
  try_to_modify_qp<InitQPState>(qp, initqp(port));
}

/**
 * rtr stands for 'ready to receive'
 */
void Connection::change_status_rtr(ibv_qp* qp, dest_info_t* dest,
                                     uint8_t port) {
  try_to_modify_qp<RtrState>(qp, rtr(dest, port));
}

/**
 * rts stands for 'ready to send'
 */
void Connection::change_status_rts(ibv_qp* qp, dest_info_t* dest) {
  try_to_modify_qp<RtsState>(qp, rts(dest));
}

void Connection::create_qps(uint8_t port, int my_rank,
                              ibv_port_attr* ib_port_att) {
  create_qps_1();

  ibv_qp_cap cap{};
  cap.max_send_wr = sq_size;
  cap.max_send_sge = 1;
  cap.max_inline_data = 4;

  QPInitAttr qp_init_attr = qpattr(cap);

  size_t qp_size = total_number_connections();
  cqs.resize(qp_size);
  qps.resize(qp_size);

  int cqe = qp_init_attr.attr.cap.max_send_wr;
  for (auto& entry : cqs) {
    entry = create_cq(ib_state->context, ib_state->pd, cqe);
    if (!entry) {
      abort();
    }
  }

  create_qps_2(port, my_rank, ib_port_att);

  for (int i = 0; i < qps.size(); i++) {
    qps[i] =
        create_qp(ib_state->pd, ib_state->context, &qp_init_attr.attr, cqs[i]);
    if (!qps[i]) {
      abort();
    }

    create_qps_3(port, qps[i], i, ib_port_att);
  }
}

void Connection::initialize_gpu_policy(ConnectionImpl** conn,
                                         uint32_t* heap_rkey) {
  CHECK_HIP(hipMalloc(reinterpret_cast<void**>(conn), sizeof(ConnectionImpl)));
  new (*conn) ConnectionImpl(this, heap_rkey);
}

/*
 * Create and write the rdma segment to the SQ
 */
void Connection::set_rdma_seg(mlx5_wqe_raddr_seg* rdma, uint64_t address,
                              uint32_t rkey) {
  rdma->raddr = htobe64(address);
  rdma->rkey = htobe32(rkey);
}

/*
 * Retrieve the address of a SQ.
 * We used this address to write the WQE directly to the SQ.
 */
uint64_t* Connection::get_address_sq(int i) {
  mlx5dv_obj mlx_obj;
  mlx5dv_qp qp_out;

  mlx_obj.qp.in = qps[i];
  mlx_obj.qp.out = &qp_out;

  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);

  return reinterpret_cast<uint64_t*>(qp_out.sq.buf);
}

void* Connection::buf_alloc([[maybe_unused]] struct ibv_pd* pd,
                            [[maybe_unused]] void* pd_context, size_t size,
                            [[maybe_unused]] size_t alignment,
                            [[maybe_unused]] uint64_t resource_type) {
  if (use_gpu_mem) {
    void* dev_ptr;
    if (coherent_cq == 1) {
#if defined USE_COHERENT_HEAP || defined USE_CACHED_HEAP
      CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&dev_ptr), size));
#else
      CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size,
                                      hipDeviceMallocFinegrained));
#endif
    } else {
      CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void**>(&dev_ptr), size,
                                      hipDeviceMallocFinegrained));
    }
    memset(dev_ptr, 0, size);
    return dev_ptr;
  }
  return IBV_ALLOCATOR_USE_DEFAULT;
}

void Connection::buf_release([[maybe_unused]] struct ibv_pd* pd,
                             [[maybe_unused]] void* pd_context, void* ptr,
                             [[maybe_unused]] uint64_t resource_type) {
  if (use_gpu_mem) {
    CHECK_HIP(hipFree(ptr));
  } else {
    free(ptr);
  }
}

void Connection::init_parent_domain_attr(ibv_parent_domain_init_attr* attr1) {
  attr1->pd = ib_state->pd;
  attr1->td = nullptr;
  attr1->comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  attr1->alloc = Connection::buf_alloc;
  attr1->free = Connection::buf_release;
  attr1->pd_context = nullptr;
}

ibv_cq* Connection::create_cq(ibv_context* context, ibv_pd* pd, int cqe) {
  use_gpu_mem = cq_use_gpu_mem;

  ibv_cq_init_attr_ex cq_attr;
  memset(&cq_attr, 0, sizeof(ibv_cq_init_attr_ex));
  cq_attr.cqe = cqe;
  cq_attr.cq_context = nullptr;
  cq_attr.channel = nullptr;
  cq_attr.comp_vector = 0;
  cq_attr.flags = 0;  // see ibv_exp_cq_create_flags
  cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD;
  cq_attr.parent_domain = pd;

  coherent_cq = 1;
  ibv_cq_ex* cq = ibv_create_cq_ex(context, &cq_attr);
  coherent_cq = 0;
  if (!cq) {
    printf("error in ibv_create_cq_ex: %d %s\n", errno, strerror(errno));
    return nullptr;
  }
  return ibv_cq_ex_to_cq(cq);
}

void Connection::init_gpu_qp_from_connection(QueuePair* gpu_qp,
                                               int conn_num) {
  int hip_dev_id = 0;
  CHECK_HIP(hipGetDevice(&hip_dev_id));
  use_gpu_mem = cq_use_gpu_mem;

  mlx5dv_cq cq_out;
  mlx5dv_obj mlx_obj;
  mlx_obj.cq.in = cqs[conn_num];
  mlx_obj.cq.out = &cq_out;

  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_CQ);
  gpu_qp->cq_log_size = log2(cq_out.cqe_cnt);
  gpu_qp->cq_size = cq_out.cqe_cnt;

  void* gpu_ptr = nullptr;
  if (use_gpu_mem) {
    gpu_qp->current_cq_q = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);
  } else {
    rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(cq_out.buf),
                                   cq_out.cqe_cnt * 64, &gpu_ptr, hip_dev_id);
    gpu_qp->current_cq_q = reinterpret_cast<mlx5_cqe64*>(gpu_ptr);
  }
  gpu_qp->current_cq_q_H = reinterpret_cast<mlx5_cqe64*>(cq_out.buf);

  rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(cq_out.dbrec), 64,
                                 &gpu_ptr, hip_dev_id);

  gpu_qp->dbrec_cq = reinterpret_cast<volatile uint32_t*>(gpu_ptr);

  use_gpu_mem = sq_use_gpu_mem;

  mlx5dv_qp qp_out;
  mlx_obj.qp.in = qps[conn_num];
  mlx_obj.qp.out = &qp_out;

  mlx5dv_init_obj(&mlx_obj, MLX5DV_OBJ_QP);

  gpu_qp->max_nwqe = (qp_out.sq.wqe_cnt);

  volatile uint32_t* dbrec_send = qp_out.dbrec + 1;

  if (use_gpu_mem) {
    gpu_qp->current_sq = reinterpret_cast<uint64_t*>(qp_out.sq.buf);
    gpu_qp->dbrec_send = reinterpret_cast<volatile uint32_t*>(dbrec_send);
  } else {
    gpu_ptr = nullptr;
    rocm_memory_lock_to_fine_grain(reinterpret_cast<void*>(qp_out.sq.buf),
                                   qp_out.sq.wqe_cnt * 64, &gpu_ptr,
                                   hip_dev_id);

    gpu_qp->current_sq = reinterpret_cast<uint64_t*>(gpu_ptr);

    rocm_memory_lock_to_fine_grain(
        reinterpret_cast<void*>(const_cast<uint32_t*>(dbrec_send)), 32,
        &gpu_ptr, hip_dev_id);

    gpu_qp->dbrec_send = reinterpret_cast<volatile uint32_t*>(gpu_ptr);
  }

  gpu_qp->current_sq_H = reinterpret_cast<uint64_t*>(qp_out.sq.buf);

  gpu_qp->setDBval(*(reinterpret_cast<uint64_t*>(qp_out.sq.buf)));

  rocm_memory_lock_to_fine_grain(qp_out.bf.reg, qp_out.bf.size * 2, &gpu_ptr,
                                 hip_dev_id);

  gpu_qp->db.ptr = reinterpret_cast<uint64_t*>(gpu_ptr);

  uint32_t* sq = reinterpret_cast<uint32_t*>(qp_out.sq.buf);
  uint32_t ctrl_qp_sq = (reinterpret_cast<uint32_t*>(sq))[1];
  gpu_qp->ctrl_qp_sq = ctrl_qp_sq & 0xFFFFFF;
  gpu_qp->ctrl_sig = (reinterpret_cast<uint64_t*>(sq))[1];
  gpu_qp->rkey = (reinterpret_cast<uint32_t*>(sq))[6 + key_offset];
  gpu_qp->lkey = (reinterpret_cast<uint32_t*>(sq))[9 + key_offset];
}

ibv_qp* Connection::create_qp(ibv_pd* pd, ibv_context* context,
                              ibv_qp_init_attr_ex* qp_attr, ibv_cq* cq) {
  use_gpu_mem = sq_use_gpu_mem;

  ibv_qp* qp = nullptr;

  assert(pd);
  assert(context);
  assert(qp_attr);

  qp_attr->send_cq = cq;
  qp_attr->recv_cq = cq;
  qp_attr->pd = pd;

  qp_attr->comp_mask = IBV_QP_INIT_ATTR_PD;

  qp = create_qp_0(context, qp_attr);

  if (!qp) {
    printf("***** error ibv_create_qp failed %d m %m \n", errno, errno);
    ibv_destroy_cq(cq);
  }

  return qp;
}

}  // namespace rocshmem
