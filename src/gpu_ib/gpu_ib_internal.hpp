/******************************************************************************
 * Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef GPU_IB_INTERNAL_HPP
#define GPU_IB_INTERNAL_HPP

#include "config.h"

#include <roc_shmem.hpp>
#include <pthread.h>

#include <infiniband/verbs.h>
#include <infiniband/verbs_exp.h>
#include <infiniband/peer_ops.h>
extern "C"{
#include <infiniband/mlx5dv.h>
}

#include "rtn.hpp"
#include "backend.hpp"

#define CHECK_HIP(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

constexpr int  MAX_WG_SIZE = 1024;

int heap_size = 1024*1024*1024;
int SQ_size =1024;
int sleep_thread = 5; //time in sec
int pre_init_done = 0;
uint32_t *last_nb_post = nullptr;

#ifdef _USE_DC_
int num_dcis = 1;
int num_dct  = 1;
constexpr int DC_IB_KEY = 0x1ee7a330;

uint32_t *dcts_num;
uint16_t *lids;

struct ibv_ah           *ah;
struct ibv_srq *srq    = NULL;
struct ibv_cq  *dct_cq = NULL;

#endif

#ifdef _USE_IPC_
constexpr int MAX_NUM_GPUs = 8;
#endif

enum atomic_op {
    ATOMIC_FCAS    = 0x11,// Atomic_Compare_and_Swap
    ATOMIC_FADD    = 0x12,// Atomic_Fetch_and_Add
    ATOMIC_MFCAS   = 0x14,// Atomic_Masked_Compare_and_Swap -
                         //    (ExtendedAtomic operation)
    ATOMIC_MFADD   = 0x15// Atomic_Masked_Fetch_and_Add -
                         //    (Extended Atomic operation)
};

typedef struct context_ib {
    struct ibv_context      *context;
    struct ibv_pd           *pd;
    struct ibv_mr           *mr;
    struct ibv_port_attr     portinfo;
} context_ib_t;

typedef struct dest_info {
    int     lid;
    int     qpn;
    int     psn;
}dest_info_t;

typedef struct heap_info{
    void        *base_heap;
    uint32_t    rkey;
} heap_info_t;

struct roc_shmem {
    int             num_wg;
    RTNGlobalHandle *rtn_handle;
    char            **heap_bases;
    uint32_t        *heap_rkey;
    ibv_mr          *heap_mr;
    uint32_t        lkey;
    size_t          current_heap_offset;
    pthread_t       thread;
    int64_t         *barrier_sync;
    char            *g_ret;
#ifdef _USE_DC_
    uint32_t        *vec_dct_num;
    uint16_t        *vec_lids;
#endif
#ifdef _USE_IPC_
    char            **ipc_bases;
    uint8_t         shm_size;
#endif
    bool            thread_done;
};

#endif
