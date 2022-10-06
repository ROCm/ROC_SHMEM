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

/**
 * @file roc_shmem.cpp
 * @brief Public header for ROC_SHMEM device and host libraries.
 *
 * This is the implementation for the public roc_shmem.hpp header file.  This
 * guy just extracts the transport from the opaque public handles and delegates
 * to the appropriate backend.
 *
 * The device-side delegation is nasty because we can't use polymorphism with
 * our current shader compiler stack.  Maybe one day.....
 *
 * TODO: Could probably autogenerate many of these functions from macros.
 *
 * TODO: Support runtime backend detection.
 *
 */
#include "config.h"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include <hip/hip_runtime.h>

#include <roc_shmem.hpp>
#include <cstdlib>


#include "context_incl.hpp"
#include "backend_bc.hpp"
#include "team.hpp"
#include "util.hpp"
#include "templates.hpp"

#include "reverse_offload/ro_net_gpu_templates.hpp"
#include "gpu_ib/gpu_ib_gpu_templates.hpp"

/******************************************************************************
 **************************** Device Vars And Init ****************************
 *****************************************************************************/

namespace rocshmem {

__constant__ roc_shmem_ctx_t ROC_SHMEM_CTX_DEFAULT;

__constant__ Backend *device_backend_proxy;

__device__ void
roc_shmem_wg_init()
{
    int provided;

    /*
     * Non-threaded init is allowed to select any thread mode, so don't worry
     * if provided is different.
     */
    roc_shmem_wg_init_thread(ROC_SHMEM_THREAD_WG_FUNNELED, &provided);
}

__device__ void
roc_shmem_wg_init_thread(int requested, int *provided)
{
    device_backend_proxy->wait_wg_init_done();
    device_backend_proxy->create_wg_state();
    roc_shmem_query_thread(provided);
}

__device__ void
roc_shmem_query_thread(int *provided)
{
#ifdef USE_THREADS
    *provided = ROC_SHMEM_THREAD_MULTIPLE;
#else
    *provided = ROC_SHMEM_THREAD_WG_FUNNELED;
#endif
}

__device__ void
roc_shmem_wg_finalize()
{
    device_backend_proxy->finalize_wg_state();
}

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

__device__ void
roc_shmem_putmem(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_putmem(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_put(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_put(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_p(T *dest, T value, int pe)
{
    roc_shmem_p(ROC_SHMEM_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__device__ T
roc_shmem_g(const T *source, int pe)
{
    return roc_shmem_g(ROC_SHMEM_CTX_DEFAULT, source, pe);
}

__device__ void
roc_shmem_getmem(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_getmem(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_get(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_get(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void
roc_shmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_putmem_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_put_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_put_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void
roc_shmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_getmem_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmem_get_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_get_nbi(ROC_SHMEM_CTX_DEFAULT, dest, source, nelems, pe);
}

__device__ void
roc_shmem_fence()
{
    roc_shmem_ctx_fence(ROC_SHMEM_CTX_DEFAULT);
}
__device__ void
roc_shmem_fence(int pe)
{
    roc_shmem_ctx_fence(ROC_SHMEM_CTX_DEFAULT, pe);
}


__device__ void
roc_shmem_quiet()
{
    roc_shmem_ctx_quiet(ROC_SHMEM_CTX_DEFAULT);
}

template <typename T>
__device__ T
roc_shmem_atomic_fetch_add(T *dest, T val, int pe)
{
    return roc_shmem_atomic_fetch_add(ROC_SHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ T
roc_shmem_atomic_compare_swap(T *dest, T cond, T val, int pe)
{
    return roc_shmem_atomic_compare_swap(ROC_SHMEM_CTX_DEFAULT,
                                        dest, cond, val, pe);
}

template <typename T>
__device__ T
roc_shmem_atomic_fetch_inc(T *dest, int pe)
{
    return roc_shmem_atomic_fetch_inc(ROC_SHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T>
__device__ T
roc_shmem_atomic_fetch(T *dest, int pe)
{
    return roc_shmem_atomic_fetch(ROC_SHMEM_CTX_DEFAULT, dest, pe);
}

template <typename T>
__device__ void
roc_shmem_atomic_add(T *dest, T val, int pe)
{
    roc_shmem_atomic_add(ROC_SHMEM_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__device__ void
roc_shmem_atomic_inc(T *dest, int pe)
{
    roc_shmem_atomic_inc(ROC_SHMEM_CTX_DEFAULT, dest, pe);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

__device__ int
translate_pe(roc_shmem_ctx_t ctx, int pe)
{
    if (ctx.team_opaque) {
        WGTeamInfo* wg_tinfo = reinterpret_cast<WGTeamInfo*>(ctx.team_opaque);
        TeamInfo lds_tinfo = wg_tinfo->get();
        return (lds_tinfo.pe_start + lds_tinfo.stride * pe);
    } else {
        return pe;
    }
}

__device__ Context *
get_internal_ctx(roc_shmem_ctx_t ctx)
{
    return reinterpret_cast<Context *>(ctx.ctx_opaque);
}

__device__ void
wg_ctx_create(long option, void **ctx)
{
    /*
     * TODO: We don't really create any new contexts here.  Init creates a
     * single wg_private context per WG and a single DEFAULT_CTX for the whole
     * GPU.  We will return the user one of these two to satisfy their request.
     */

    Context* phys_ctx {nullptr};

    if (option & (ROC_SHMEM_CTX_WG_PRIVATE | ROC_SHMEM_CTX_PRIVATE)) {
        /*
         * Locking policy for WG-private context is set during
         * context creation based on threading mode for the runtime.
         *
         * If the runtime is set for ROC_SHMEM_THREAD_MULTIPLE, then it would be
         * possible to decide at a per CTX level more optimized semantics
         * (e.g., ROC_SHMEM_CTX_SERIALIZED would disable intra-wg locking).
         *
         * Unfortunately, since we use the same CTX for multiple ctx_create,
         * we are stuck with the most restrictive performance mode for the
         * given thread policy.
         */
        phys_ctx = WGState::instance()->get_private_ctx();
    } else {
        /*
         * All SHARED contexts satisfied with the DEFAULT_CTX because it is
         * the only context with the required visibility (global).
         *
         * SERIALIZED is a missed performance oppurtunity like mentioned for
         * ROC_SHMEM_CTX_WG_PRIVATE, but ignoring it is allowable for correctness.
         */
        phys_ctx = get_internal_ctx(ROC_SHMEM_CTX_DEFAULT);
    }

    phys_ctx->setFence(option);

    *ctx = phys_ctx;

    __syncthreads();

}

__device__ void
roc_shmem_wg_ctx_create(long option, roc_shmem_ctx_t *ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_create\n");

    void *phy_ctx;

    wg_ctx_create(option, &phy_ctx);

    /* Set the pointer to the physical ctx */
    ctx->ctx_opaque = phy_ctx;
    /* Since this ctx is on TEAM_WORLD, we don't need any PE translation */
    ctx->team_opaque = nullptr;

    /* Perform backend-specific allocation */
    get_internal_ctx((*ctx))->ctx_create();
}

__device__ int
roc_shmem_wg_team_create_ctx(roc_shmem_team_t team,
                             long options,
                             roc_shmem_ctx_t *ctx)
{
    GPU_DPRINTF("Function: roc_shmem_team_create_ctx\n");

    if (team == ROC_SHMEM_TEAM_INVALID) {
        return -1;
    }

    /* Get a physical ctx */
    void* phy_ctx;
    wg_ctx_create(options, &phy_ctx);
    ctx->ctx_opaque = phy_ctx;

    /**
     * As of ROCm 4.5, support for __managed__ is under development.
     * Hence, users cannot access ROC_SHMEM_TEAM_WORLD in device code
     * and so we do not check for it here.
     * TODO: When __managed__ is supported, we need to declare
     * ROC_SHMEM_TEAM_WORLD with __managed__ and then we need to
     * check for team equaling ROC_SHMEM_TEAM_WORLD, in which case we
     * need to set ctx->team_opaque to nullptr.
     *
     * For now, directly get the info stored about this team on LDS.
     */
    Team* team_obj {get_internal_team(team)};
    WGState* wg_state_p {WGState::instance()};
    WGTeamCtxsPolicy& team_ctxs_pol_ref {wg_state_p->team_ctxs_policy};
    WGTeamInfo* info_wrt_world {team_ctxs_pol_ref.get_team_info(team_obj)};
    if (is_thread_zero_in_block()) {
        ctx->team_opaque = info_wrt_world;
        __syncthreads();
    }

    return 0;
}

__device__ void
roc_shmem_wg_ctx_destroy(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_destroy\n");

    /**
     * If this was allocated on a team then return the team info
     * slot on the LDS pool.
     */
    if (ctx.team_opaque) {
        WGState::instance()->team_ctxs_policy.remove_team_info(reinterpret_cast<WGTeamInfo *>(ctx.team_opaque));
    }

    /*
     * Delay destroying contexts until the work-group indicates that it is
     * finished in roc_shmem_wg_finalize().  Need to do this for now since
     * we are recycling contexts and don't want to destroy one prematurely.
     */
    // get_internal_ctx(ctx)->ctx_destroy();
}

__device__ void
roc_shmem_ctx_threadfence_system(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_threadfence_system\n");

    get_internal_ctx(ctx)->threadfence_system();
}

__device__ void
roc_shmem_ctx_putmem(roc_shmem_ctx_t ctx, void *dest, const void *source,
                 size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_putmem\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->putmem(dest, source, nelems, pe_in_world);
}

template <typename T> __device__ void
roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_put\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->put(dest, source, nelems, pe_in_world);
}

template <typename T> __device__ void
roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_p\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->p(dest, value, pe_in_world);
}

template <typename T> __device__ T
roc_shmem_g(roc_shmem_ctx_t ctx, const T *source, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_g\n");

    int pe_in_world = translate_pe(ctx, pe);

    return get_internal_ctx(ctx)->g(source, pe_in_world);
}

__device__ void
roc_shmem_ctx_getmem(roc_shmem_ctx_t ctx, void *dest, const void *source,
                 size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_getmem\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->getmem(dest, source, nelems, pe_in_world);
}

template <typename T> __device__ void
roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems,
              int pe)
{
    GPU_DPRINTF("Function: roc_shmem_get\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->get(dest, source, nelems, pe_in_world);
}

__device__ void
roc_shmem_ctx_putmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source,
                     size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_putmem_nbi\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe_in_world);
}

template <typename T> __device__ void
roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_put_nbi\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe_in_world);
}

__device__ void
roc_shmem_ctx_getmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source,
                     size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_getmem_nbi\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->getmem_nbi(dest, source, nelems, pe_in_world);
}

template <typename T>
__device__ void
roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_get_nbi\n");

    int pe_in_world = translate_pe(ctx, pe);

    get_internal_ctx(ctx)->get_nbi(dest, source, nelems, pe_in_world);
}

__device__ void
roc_shmem_ctx_fence(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_fence\n");

    get_internal_ctx(ctx)->fence();
}

__device__ void
roc_shmem_ctx_fence(roc_shmem_ctx_t ctx, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_fence\n");

    get_internal_ctx(ctx)->fence(pe);
}


__device__ void
roc_shmem_ctx_quiet(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_quiet\n");

    get_internal_ctx(ctx)->quiet();
}

__device__ void*
roc_shmem_ptr(const void * dest, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_ptr\n");

    return get_internal_ctx(ROC_SHMEM_CTX_DEFAULT)->shmem_ptr(dest, pe);
}

template <typename T, ROC_SHMEM_OP Op> __device__ void
roc_shmem_wg_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source,
                    int nreduce, int PE_start, int logPE_stride,
                    int PE_size, T *pWrk, long *pSync)
{
    GPU_DPRINTF("Function: roc_shmem_to_all\n");

    get_internal_ctx(ctx)->to_all<T, Op>(dest, source, nreduce, PE_start,
                                         logPE_stride, PE_size, pWrk, pSync);
}

template <typename T, ROC_SHMEM_OP Op> __device__ void
roc_shmem_wg_to_all(roc_shmem_ctx_t ctx, roc_shmem_team_t team,
                    T *dest, const T *source, int nreduce)
{
    GPU_DPRINTF("Function: roc_shmem_to_all\n");

    get_internal_ctx(ctx)->to_all<T, Op>(team, dest, source, nreduce);
}

template <typename T>
__device__ void
roc_shmem_wg_broadcast(roc_shmem_ctx_t ctx,
                       T *dest,
                       const T *source,
                       int nelem,
                       int pe_root,
                       int pe_start,
                       int log_pe_stride,
                       int pe_size,
                       long *p_sync)
{
    GPU_DPRINTF("Function: roc_shmem_broadcast\n");

    get_internal_ctx(ctx)->broadcast<T>(dest,
                                        source,
                                        nelem,
                                        pe_root,
                                        pe_start,
                                        log_pe_stride,
                                        pe_size,
                                        p_sync);
}

template <typename T>
__device__ void
roc_shmem_wg_broadcast(roc_shmem_ctx_t ctx,
                       roc_shmem_team_t team,
                       T *dest,
                       const T *source,
                       int nelem,
                       int pe_root)
{
    GPU_DPRINTF("Function: Team-based roc_shmem_broadcast\n");

    get_internal_ctx(ctx)->broadcast<T>(team,
                                        dest,
                                        source,
                                        nelem,
                                        pe_root);
}

template <typename T>
__device__ void
roc_shmem_wg_alltoall(roc_shmem_ctx_t ctx,
                      roc_shmem_team_t team,
                      T *dest,
                      const T *source,
                      int nelem)
{
    GPU_DPRINTF("Function: roc_shmem_alltoall\n");

    get_internal_ctx(ctx)->alltoall<T>(team,
                                       dest,
                                       source,
                                       nelem);
}

template <typename T>
__device__ void
roc_shmem_wg_fcollect(roc_shmem_ctx_t ctx,
                      roc_shmem_team_t team,
                      T *dest,
                      const T *source,
                      int nelem)
{
    GPU_DPRINTF("Function: roc_shmem_fcollect\n");

    get_internal_ctx(ctx)->fcollect<T>(team,
                                       dest,
                                       source,
                                       nelem);
}

template <typename T>
__device__ void
roc_shmem_wait_until(T *ptr, roc_shmem_cmps cmp, T val)
{
    GPU_DPRINTF("Function: roc_shmem_wait_until\n");

    Context *ctx_internal = get_internal_ctx(ROC_SHMEM_CTX_DEFAULT);
    ctx_internal->ctxStats.incStat(NUM_WAIT_UNTIL);

    ctx_internal->wait_until(ptr, cmp, val);
}

template <typename T>
__device__ int
roc_shmem_test(T *ptr, roc_shmem_cmps cmp, T val)
{
    GPU_DPRINTF("Function: roc_shmem_testl\n");

    Context *ctx_internal = get_internal_ctx(ROC_SHMEM_CTX_DEFAULT);
    ctx_internal->ctxStats.incStat(NUM_TEST);

    return ctx_internal->test(ptr, cmp, val);
}

__device__ void
roc_shmem_ctx_wg_barrier_all(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_barrier_all\n");

    get_internal_ctx(ctx)->barrier_all();
}

__device__ void
roc_shmem_wg_barrier_all()
{
    roc_shmem_ctx_wg_barrier_all(ROC_SHMEM_CTX_DEFAULT);
}

__device__ void
roc_shmem_ctx_wg_sync_all(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_sync_all\n");

    get_internal_ctx(ctx)->sync_all();
}

__device__ void
roc_shmem_wg_sync_all()
{
    roc_shmem_ctx_wg_sync_all(ROC_SHMEM_CTX_DEFAULT);
}

__device__ void
roc_shmem_ctx_wg_team_sync(roc_shmem_ctx_t ctx, roc_shmem_team_t team)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_sync_all\n");

    get_internal_ctx(ctx)->sync(team);
}

__device__ void
roc_shmem_wg_team_sync(roc_shmem_team_t team)
{
    roc_shmem_ctx_wg_team_sync(ROC_SHMEM_CTX_DEFAULT, team);
}

__device__ int
roc_shmem_ctx_n_pes(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_n_pes\n");

    return get_internal_ctx(ctx)->num_pes;
}

__device__ int
roc_shmem_n_pes()
{
    return get_internal_ctx(ROC_SHMEM_CTX_DEFAULT)->num_pes;
}

__device__ int
roc_shmem_ctx_my_pe(roc_shmem_ctx_t ctx)
{
    GPU_DPRINTF("Function: roc_shmem_ctx_my_pe\n");

    return get_internal_ctx(ctx)->my_pe;
}

__device__ int
roc_shmem_my_pe()
{
    return get_internal_ctx(ROC_SHMEM_CTX_DEFAULT)->my_pe;
}

__device__ uint64_t
roc_shmem_timer()
{
    GPU_DPRINTF("Function: roc_shmem_timer\n");

    return __read_clock();
}

template <typename T> __device__ T
roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_atomic_fetch_add\n");

    return get_internal_ctx(ctx)->amo_fetch_add(dest, val, 0, pe);
}

template <typename T> __device__ T
roc_shmem_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest, T cond, T val,
                             int pe)
{
    GPU_DPRINTF("Function: roc_shmem_atomic_compare_swap\n");

    return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T> __device__ T
roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_atomic_fetch_inc\n");

    return get_internal_ctx(ctx)->amo_fetch_add(dest, 1, 0, pe);
}

template <typename T> __device__ T
roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_atomic_fetch\n");

    return get_internal_ctx(ctx)->amo_fetch_add(dest, 0, 0, pe);
}

template <typename T> __device__ void
roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_atomic_add\n");

    get_internal_ctx(ctx)->amo_add((void*)dest, val, 0, pe);
}

template <typename T>
__device__ void
roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    GPU_DPRINTF("Function: roc_shmem_atomic_inc\n");

    get_internal_ctx(ctx)->amo_add(dest, 1, 0, pe);
}

/**
 *      SHMEM X RMA API for WG and Wave level
 */
__device__ void
roc_shmemx_ctx_putmem_wave(roc_shmem_ctx_t ctx, void *dest, const void *source,
                           size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_wave\n");

    get_internal_ctx(ctx)->putmem_wave(dest, source, nelems, pe);
}

__device__ void
roc_shmemx_ctx_putmem_wg(roc_shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_wg\n");

    get_internal_ctx(ctx)->putmem_wg(dest, source, nelems, pe);
}

__device__ void
roc_shmemx_ctx_putmem_nbi_wave(roc_shmem_ctx_t ctx, void *dest, const void *source,
                               size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_nbi_wave\n");

    get_internal_ctx(ctx)->putmem_nbi_wave(dest, source, nelems, pe);
}

__device__ void
roc_shmemx_ctx_putmem_nbi_wg(roc_shmem_ctx_t ctx, void *dest, const void *source,
                             size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_putmem_nbi_wg\n");

    get_internal_ctx(ctx)->putmem_nbi_wg(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmemx_put_wave(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_put_wave\n");

    get_internal_ctx(ctx)->put_wave(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmemx_put_wg(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_put_wg\n");

    get_internal_ctx(ctx)->put_wg(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmemx_put_nbi_wave(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_put_nbi_wave\n");

    get_internal_ctx(ctx)->put_nbi_wave(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmemx_put_nbi_wg(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_put_nbi_wg\n");

    get_internal_ctx(ctx)->put_nbi_wg(dest, source, nelems, pe);
}

__device__ void
roc_shmemx_ctx_getmem_wg(roc_shmem_ctx_t ctx, void *dest, const void *source,
                         size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_wg\n");

    get_internal_ctx(ctx)->getmem_wg(dest, source, nelems, pe);
}

__device__ void
roc_shmemx_ctx_getmem_wave(roc_shmem_ctx_t ctx, void *dest, const void *source,
                           size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_wave\n");

    get_internal_ctx(ctx)->getmem_wave(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmemx_get_wg(roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems,
              int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_get_wg\n");

    get_internal_ctx(ctx)->get_wg(dest, source, nelems, pe);
}

template <typename T> __device__ void
roc_shmemx_get_wave(roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems,
              int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_get_wave\n");

    get_internal_ctx(ctx)->get_wave(dest, source, nelems, pe);
}

__device__ void
roc_shmemx_ctx_getmem_nbi_wg(roc_shmem_ctx_t ctx, void *dest, const void *source,
                             size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_nbi_wg\n");

    get_internal_ctx(ctx)->getmem_nbi_wg(dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmemx_get_nbi_wg(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_get_nbi_wg\n");

    get_internal_ctx(ctx)->get_nbi_wg(dest, source, nelems, pe);
}

__device__ void
roc_shmemx_ctx_getmem_nbi_wave(roc_shmem_ctx_t ctx, void *dest, const void *source,
                               size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_ctx_getmem_nbi_wave\n");

    get_internal_ctx(ctx)->getmem_nbi_wave(dest, source, nelems, pe);
}

template <typename T>
__device__ void
roc_shmemx_get_nbi_wave(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    GPU_DPRINTF("Function: roc_shmemx_get_nbi_wave\n");

    get_internal_ctx(ctx)->get_nbi_wave(dest, source, nelems, pe);
}

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__device__ int
roc_shmem_team_translate_pe(roc_shmem_team_t src_team,
                            int src_pe,
                            roc_shmem_team_t dst_team)
{
    return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************* Template Generation Macros *************************
 *****************************************************************************/

/**
 * Template generator for reductions
 **/
#define REDUCTION_GEN(T, Op) \
    template __device__ void \
    roc_shmem_wg_to_all<T, Op>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                               int nreduce, int PE_start, int logPE_stride, \
                               int PE_size, T *pWrk, long *pSync); \
    template __device__ void \
    roc_shmem_wg_to_all<T, Op>(roc_shmem_ctx_t ctx, roc_shmem_team_t team, \
                               T *dest, const T *source, int nreduce);

/**
 * Declare templates for the required datatypes (for the compiler)
 **/

#define RMA_GEN(T) \
    template __device__ void \
    roc_shmem_put<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                     size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_put_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_p<T>(roc_shmem_ctx_t ctx, T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_get<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                     size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_get_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe); \
    template __device__ T \
    roc_shmem_g<T>(roc_shmem_ctx_t ctx, const T *source, int pe); \
    template __device__ void \
    roc_shmem_put<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_put_nbi<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_p<T>(T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_get<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmem_get_nbi<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ T \
    roc_shmem_g<T>(const T *source, int pe); \
    template __device__ void \
    roc_shmem_wg_broadcast<T>(roc_shmem_ctx_t ctx, \
                              T *dest, \
                              const T *source, \
                              int nelem, \
                              int pe_root, \
                              int pe_start, \
                              int log_pe_stride, \
                              int pe_size, \
                              long *p_sync);\
    template __device__ void \
    roc_shmem_wg_broadcast<T>(roc_shmem_ctx_t ctx, \
                              roc_shmem_team_t team, \
                              T *dest, \
                              const T *source, \
                              int nelem, \
                              int pe_root); \
    template __device__ void \
    roc_shmem_wg_alltoall<T>(roc_shmem_ctx_t ctx, \
                             roc_shmem_team_t team, \
                             T *dest, \
                             const T *source, \
                             int nelem); \
    template __device__ void \
    roc_shmem_wg_fcollect<T>(roc_shmem_ctx_t ctx, \
                             roc_shmem_team_t team, \
                             T *dest, \
                             const T *source, \
                             int nelem); \
    template __device__ void \
    roc_shmemx_put_wave<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                           size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_put_wg<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe);\
    template __device__ void \
    roc_shmemx_put_wave<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_put_wg<T>(T *dest, const T *source, size_t nelems, int pe);\
    template __device__ void \
    roc_shmemx_put_nbi_wave<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                               size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_put_nbi_wg<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                             size_t nelems, int pe);\
    template __device__ void \
    roc_shmemx_put_nbi_wave<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_put_nbi_wg<T>(T *dest, const T *source, size_t nelems, int pe);\
    template __device__ void \
    roc_shmemx_get_wave<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                           size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_get_wg<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe);\
    template __device__ void \
    roc_shmemx_get_wave<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_get_wg<T>(T *dest, const T *source, size_t nelems, int pe);\
    template __device__ void \
    roc_shmemx_get_nbi_wave<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                               size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_get_nbi_wg<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                             size_t nelems, int pe);\
    template __device__ void \
    roc_shmemx_get_nbi_wave<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __device__ void \
    roc_shmemx_get_nbi_wg<T>(T *dest, const T *source, size_t nelems, int pe);

#define AMO_GEN(T) \
    template __device__ T \
    roc_shmem_atomic_fetch_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, \
                                  int pe); \
    template __device__ T \
    roc_shmem_atomic_compare_swap<T>(roc_shmem_ctx_t ctx,  T *dest, T cond, \
                                    T value, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __device__ void \
    roc_shmem_atomic_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_atomic_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_add<T>(T *dest, T value, int pe); \
    template __device__ T \
    roc_shmem_atomic_compare_swap<T>(T *dest, T cond, T value, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch_inc<T>(T *dest, int pe); \
    template __device__ T \
    roc_shmem_atomic_fetch<T>(T *dest, int pe); \
    template __device__ void \
    roc_shmem_atomic_add<T>(T *dest, T value, int pe); \
    template __device__ void \
    roc_shmem_atomic_inc<T>(T *dest, int pe);

#define WAIT_GEN(T) \
     template __device__ void \
    roc_shmem_wait_until<T>(T *ptr, roc_shmem_cmps cmp, T val);\
    template __device__ int \
    roc_shmem_test<T>(T *ptr, roc_shmem_cmps cmp, T val);\
    template __device__ void \
    Context::wait_until<T>(T *ptr, roc_shmem_cmps cmp, T val); \
    template __device__ int \
    Context::test<T>(T *ptr, roc_shmem_cmps cmp, T val);

#define ARITH_REDUCTION_GEN(T) \
    REDUCTION_GEN(T, ROC_SHMEM_SUM) \
    REDUCTION_GEN(T, ROC_SHMEM_MIN) \
    REDUCTION_GEN(T, ROC_SHMEM_MAX) \
    REDUCTION_GEN(T, ROC_SHMEM_PROD)

#define BITWISE_REDUCTION_GEN(T) \
    REDUCTION_GEN(T, ROC_SHMEM_OR) \
    REDUCTION_GEN(T, ROC_SHMEM_AND) \
    REDUCTION_GEN(T, ROC_SHMEM_XOR)

#define INT_REDUCTION_GEN(T) \
    ARITH_REDUCTION_GEN(T) \
    BITWISE_REDUCTION_GEN(T)

#define FLOAT_REDUCTION_GEN(T) \
    ARITH_REDUCTION_GEN(T)

/**
 * Define APIs to call the template functions
 **/

#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                                 int nreduce, int PE_start, int logPE_stride, \
                                                 int PE_size, T *pWrk, long *pSync) \
    { \
        roc_shmem_wg_to_all<T, Op>(ctx, dest, source, nreduce, PE_start, \
                                   logPE_stride, PE_size, pWrk, pSync); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_wg_to_all(roc_shmem_ctx_t ctx, roc_shmem_team_t team, \
                                                 T *dest, const T *source, int nreduce) \
    { \
        roc_shmem_wg_to_all<T, Op>(ctx, team, dest, source, nreduce); \
    }

#define ARITH_REDUCTION_DEF_GEN(T, TNAME) \
    REDUCTION_DEF_GEN(T, TNAME, sum, ROC_SHMEM_SUM) \
    REDUCTION_DEF_GEN(T, TNAME, min, ROC_SHMEM_MIN) \
    REDUCTION_DEF_GEN(T, TNAME, max, ROC_SHMEM_MAX) \
    REDUCTION_DEF_GEN(T, TNAME, prod, ROC_SHMEM_PROD)

#define BITWISE_REDUCTION_DEF_GEN(T, TNAME) \
    REDUCTION_DEF_GEN(T, TNAME, or, ROC_SHMEM_OR) \
    REDUCTION_DEF_GEN(T, TNAME, and, ROC_SHMEM_AND) \
    REDUCTION_DEF_GEN(T, TNAME, xor, ROC_SHMEM_XOR)

#define INT_REDUCTION_DEF_GEN(T, TNAME) \
    ARITH_REDUCTION_DEF_GEN(T, TNAME) \
    BITWISE_REDUCTION_DEF_GEN(T, TNAME)

#define FLOAT_REDUCTION_DEF_GEN(T, TNAME) \
    ARITH_REDUCTION_DEF_GEN(T, TNAME)

#define RMA_DEF_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_ctx_##TNAME##_put(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                size_t nelems, int pe) \
    { \
        roc_shmem_put<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                    size_t nelems, int pe) \
    { \
        roc_shmem_put_nbi<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe) \
    { \
        roc_shmem_p<T>(ctx, dest, value, pe); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_get(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                size_t nelems, int pe) \
    { \
        roc_shmem_get<T>(ctx, dest, source, nelems, pe); \
    } \
     __device__ T \
    roc_shmem_ctx_##TNAME##_g(roc_shmem_ctx_t ctx, const T *source, int pe) \
    { \
        return roc_shmem_g<T>(ctx, source, pe); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                    size_t nelems, int pe) \
    { \
        roc_shmem_get_nbi<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmem_##TNAME##_put(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_put<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmem_##TNAME##_put_nbi(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_put_nbi<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmem_##TNAME##_p(T *dest, T value, int pe) \
    { \
        roc_shmem_p<T>(dest, value, pe); \
    } \
    __device__ void \
    roc_shmem_##TNAME##_get(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_get<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmem_##TNAME##_get_nbi(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_get_nbi<T>(dest, source, nelems, pe); \
    } \
     __device__ T \
    roc_shmem_##TNAME##_g(const T *source, int pe) \
    { \
        return roc_shmem_g<T>(source, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_put_wave(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                      size_t nelems, int pe) \
    { \
        roc_shmemx_put_wave<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_put_wg(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                             size_t nelems, int pe) \
    { \
        roc_shmemx_put_wg<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_put_wave(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_put_wave<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_put_wg(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_put_wg<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_put_nbi_wave(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                          size_t nelems, int pe) \
    { \
        roc_shmemx_put_nbi_wave<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_put_nbi_wg(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                        size_t nelems, int pe) \
    { \
        roc_shmemx_put_nbi_wg<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_put_nbi_wave(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_put_nbi_wave<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_put_nbi_wg(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_put_nbi_wg<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_get_wave(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                               size_t nelems, int pe) \
    { \
        roc_shmemx_get_wave<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_get_wg(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                    size_t nelems, int pe) \
    { \
        roc_shmemx_get_wg<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_get_wave(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_get_wave<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_get_wg(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_get_wg<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_get_nbi_wave(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                          size_t nelems, int pe) \
    { \
        roc_shmemx_get_nbi_wave<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_ctx_##TNAME##_get_nbi_wg(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                        size_t nelems, int pe) \
    { \
        roc_shmemx_get_nbi_wg<T>(ctx, dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_get_nbi_wave(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_get_nbi_wave<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmemx_##TNAME##_get_nbi_wg(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmemx_get_nbi_wg<T>(dest, source, nelems, pe); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_broadcast(roc_shmem_ctx_t ctx, \
                                         T *dest, \
                                         const T *source, \
                                         int nelem, \
                                         int pe_root, \
                                         int pe_start, \
                                         int log_pe_stride, \
                                         int pe_size, \
                                         long *p_sync) \
    { \
        roc_shmem_wg_broadcast<T>(ctx, dest, source, nelem, pe_root, pe_start, \
                                  log_pe_stride, pe_size, p_sync); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_broadcast(roc_shmem_ctx_t ctx, \
                                         roc_shmem_team_t team, \
                                         T *dest, \
                                         const T *source, \
                                         int nelem, \
                                         int pe_root) \
    { \
        roc_shmem_wg_broadcast<T>(ctx, team, dest, source, nelem, pe_root); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_alltoall(roc_shmem_ctx_t ctx, \
                                         roc_shmem_team_t team, \
                                         T *dest, \
                                         const T *source, \
                                         int nelem) \
    { \
        roc_shmem_wg_alltoall<T>(ctx, team, dest, source, nelem); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_wg_fcollect(roc_shmem_ctx_t ctx, \
                                         roc_shmem_team_t team, \
                                         T *dest, \
                                         const T *source, \
                                         int nelem) \
    { \
        roc_shmem_wg_fcollect<T>(ctx, team, dest, source, nelem); \
    }

#define AMO_DEF_GEN(T, TNAME) \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T value, \
                                             int pe) \
    { \
        return roc_shmem_atomic_fetch_add<T>(ctx, dest, value, pe); \
    } \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest, T cond, \
                                               T value, int pe) \
    { \
        return roc_shmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe); \
    } \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch_inc<T>(ctx, dest, pe); \
    } \
    __device__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch<T>(ctx, dest, pe); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_atomic_add(roc_shmem_ctx_t ctx, T *dest, T value, int pe) \
    { \
        roc_shmem_atomic_add<T>(ctx, dest, value, pe); \
    } \
    __device__ void \
    roc_shmem_ctx_##TNAME##_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        roc_shmem_atomic_inc<T>(ctx, dest, pe); \
    } \
    __device__ T \
    roc_shmem_##TNAME##_atomic_fetch_add(T *dest, T value, int pe) \
    { \
        return roc_shmem_atomic_fetch_add<T>(dest, value, pe); \
    } \
    __device__ T \
    roc_shmem_##TNAME##_atomic_compare_swap(T *dest, T cond, T value, int pe) \
    { \
        return roc_shmem_atomic_compare_swap<T>(dest, cond, value, pe); \
    } \
    __device__ T \
    roc_shmem_##TNAME##_atomic_fetch_inc(T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch_inc<T>(dest, pe); \
    } \
    __device__ T \
    roc_shmem_##TNAME##_atomic_fetch(T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch<T>(dest, pe); \
    } \
    __device__ void \
    roc_shmem_##TNAME##_atomic_add(T *dest, T value, int pe) \
    { \
        roc_shmem_atomic_add<T>(dest, value, pe); \
    } \
    __device__ void \
    roc_shmem_##TNAME##_atomic_inc(T *dest, int pe) \
    { \
        roc_shmem_atomic_inc<T>(dest, pe); \
    }

#define WAIT_DEF_GEN(T, TNAME) \
    __device__ void \
    roc_shmem_##TNAME##_wait_until(T *ptr, roc_shmem_cmps cmp, T val) \
    { \
        roc_shmem_wait_until<T>(ptr, cmp, val); \
    } \
    __device__ int \
    roc_shmem_##TNAME##_test(T *ptr, roc_shmem_cmps cmp, T val) \
    { \
        return roc_shmem_test<T>(ptr, cmp, val); \
    }

/******************************************************************************
 ************************* Macro Invocation Per Type **************************
 *****************************************************************************/

INT_REDUCTION_GEN(int)
INT_REDUCTION_GEN(short)
INT_REDUCTION_GEN(long)
INT_REDUCTION_GEN(long long)
FLOAT_REDUCTION_GEN(float)
FLOAT_REDUCTION_GEN(double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
//FLOAT_REDUCTION_GEN(long double)

/* Supported RMA types */
RMA_GEN(float) RMA_GEN(double) RMA_GEN(char) // RMA_GEN(long double)
RMA_GEN(signed char) RMA_GEN(short) RMA_GEN(int) RMA_GEN(long)
RMA_GEN(long long) RMA_GEN(unsigned char) RMA_GEN(unsigned short)
RMA_GEN(unsigned int) RMA_GEN(unsigned long) RMA_GEN(unsigned long long)

/* Supported AMO types for now (Only 64-bits)  */
AMO_GEN(int64_t)
AMO_GEN(uint64_t)
//AMO_GEN(long long)
//AMO_GEN(unsigned long long)
//AMO_GEN(size_t)
//AMO_GEN(ptrdiff_t)

/* Supported synchronization types */
//WAIT_GEN(float) WAIT_GEN(double) WAIT_GEN(char) //WAIT_GEN(long double)
//WAIT_GEN(unsigned char) WAIT_GEN(unsigned short)
//WAIT_GEN(signed char) WAIT_GEN(short)
WAIT_GEN(int) WAIT_GEN(long) WAIT_GEN(long long)
WAIT_GEN(unsigned int) WAIT_GEN(unsigned long) WAIT_GEN(unsigned long long)

INT_REDUCTION_DEF_GEN(int, int)
INT_REDUCTION_DEF_GEN(short, short)
INT_REDUCTION_DEF_GEN(long, long)
INT_REDUCTION_DEF_GEN(long long, longlong)
FLOAT_REDUCTION_DEF_GEN(float, float)
FLOAT_REDUCTION_DEF_GEN(double, double)
// long double reduction fails. hipcc/device may not support long double.
// so disable it for now.
//FLOAT_REDUCTION_DEF_GEN(long double, longdouble)

RMA_DEF_GEN(float, float)
RMA_DEF_GEN(double, double)
RMA_DEF_GEN(char, char)
//RMA_DEF_GEN(long double, longdouble)
RMA_DEF_GEN(signed char, schar)
RMA_DEF_GEN(short, short)
RMA_DEF_GEN(int, int)
RMA_DEF_GEN(long, long)
RMA_DEF_GEN(long long, longlong)
RMA_DEF_GEN(unsigned char, uchar)
RMA_DEF_GEN(unsigned short, ushort)
RMA_DEF_GEN(unsigned int, uint)
RMA_DEF_GEN(unsigned long, ulong)
RMA_DEF_GEN(unsigned long long, ulonglong)

AMO_DEF_GEN(int64_t, int64)
AMO_DEF_GEN(uint64_t, uint64)
//AMO_DEF_GEN(long long, longlong)
//AMO_DEF_GEN(unsigned long long, ulonglong)
//AMO_DEF_GEN(size_t, size)
//AMO_DEF_GEN(ptrdiff_t, ptrdiff)

//WAIT_DEF_GEN(float, float)
//WAIT_DEF_GEN(double, double)
//WAIT_DEF_GEN(char, char)
//WAIT_DEF_GEN(long double, longdouble)
//WAIT_DEF_GEN(signed char, schar)
//WAIT_DEF_GEN(short, short)
WAIT_DEF_GEN(int, int)
WAIT_DEF_GEN(long, long)
WAIT_DEF_GEN(long long, longlong)
//WAIT_DEF_GEN(unsigned char, uchar)
//WAIT_DEF_GEN(unsigned short, ushort)
WAIT_DEF_GEN(unsigned int, uint)
WAIT_DEF_GEN(unsigned long, ulong)
WAIT_DEF_GEN(unsigned long long, ulonglong)

}  // namespace rocshmem
