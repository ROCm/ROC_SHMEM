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

#include <roc_shmem.hpp>

#include <cstdlib>
#include <functional>

#include "context_incl.hpp"
#include "backend_bc.hpp"
#include "team.hpp"
#include "util.hpp"
#include "templates_host.hpp"
#include "mpi_init_singleton.hpp"

#include "gpu_ib/backend_ib.hpp"
#include "gpu_ib/gpu_ib_host_templates.hpp"
#include "reverse_offload/backend_ro.hpp"
#include "reverse_offload/ro_net_host_templates.hpp"

namespace rocshmem {

#define VERIFY_BACKEND() {                                                   \
        if (!backend) {                                                      \
            fprintf(stderr, "ROC_SHMEM_ERROR: %s in file '%s' in line %d\n", \
                            "Call 'roc_shmem_init'", __FILE__, __LINE__);\
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

Backend *backend = nullptr;

roc_shmem_ctx_t ROC_SHMEM_HOST_CTX_DEFAULT;

/**
 * Begin Host Code
 **/


[[maybe_unused]]
__host__ Status inline
library_init(unsigned num_wgs)
{
    assert(!backend);
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess)
        exit (-static_cast<int>(Status::ROC_SHMEM_UNKNOWN_ERROR));

    if (count == 0) {
        printf("No GPU found! \n");
        exit(-1);
    }

    rocm_init();
    if (getenv("ROC_SHMEM_RO") != nullptr) {
        CHECK_HIP(hipHostMalloc(&backend, sizeof(ROBackend)));
        backend = new (backend) ROBackend(num_wgs);
    } else {
        CHECK_HIP(hipHostMalloc(&backend, sizeof(GPUIBBackend)));
        backend = new (backend) GPUIBBackend(num_wgs);
    }

    if (!backend)
        return Status::ROC_SHMEM_OOM_ERROR;

    return Status::ROC_SHMEM_SUCCESS;
}

[[maybe_unused]]
__host__ Status
roc_shmem_init(unsigned num_wgs)
{
    return library_init(num_wgs);
}

[[maybe_unused]]
__host__ int
roc_shmem_init_thread(int required, int *provided, unsigned num_wgs)
{
    int ret;
    Status status;

    status = library_init(num_wgs);

    if (status == Status::ROC_SHMEM_SUCCESS) {
        roc_shmem_query_thread(provided);
        ret = 0;        /* successful */
    } else {
        ret = -1;       /* erroneous non-zero value */
    }

    return ret;
}

[[maybe_unused]]
__host__ int
roc_shmem_my_pe()
{
    MPIInitSingleton *s = s->GetInstance();
    return s->get_rank();
}

[[maybe_unused]]
__host__ int
roc_shmem_n_pes()
{
    MPIInitSingleton *s = s->GetInstance();
    return s->get_nprocs();
}

[[maybe_unused]]
__host__ void *
roc_shmem_malloc(size_t size)
{
    VERIFY_BACKEND();

    void *ptr;
    backend->heap.malloc(&ptr, size);

    roc_shmem_barrier_all();

    return ptr;
}

[[maybe_unused]]
__host__ void
roc_shmem_free(void *ptr)
{
    VERIFY_BACKEND();

    roc_shmem_barrier_all();

    backend->heap.free(ptr);
}

[[maybe_unused]]
__host__ Status
roc_shmem_reset_stats()
{
    VERIFY_BACKEND();
    return backend->reset_stats();
}

[[maybe_unused]]
__host__ Status
roc_shmem_dump_stats()
{
    /** TODO: Many stats are backend independent! **/
    VERIFY_BACKEND();
    return backend->dump_stats();
}

[[maybe_unused]]
__host__ Status
roc_shmem_finalize()
{
    VERIFY_BACKEND();

    /*
     * Destroy all the ctxs that the user
     * created but did not manually destroy
     */
    backend->destroy_remaining_ctxs();

    /*
     * Destroy all the teams that the user
     * created but did not manually destroy
     */
    auto team_destroy {std::bind(&Backend::team_destroy,
                                 backend,
                                 std::placeholders::_1)};
    backend->team_tracker.destroy_all(team_destroy);

    backend->~Backend();
    CHECK_HIP(hipHostFree(backend));

    delete MPIInitSingleton::GetInstance();

    return Status::ROC_SHMEM_SUCCESS;
}

[[maybe_unused]]
__host__ Status
roc_shmem_dynamic_shared(size_t *shared_bytes)
{
    if (getenv("ROC_SHMEM_RO") != nullptr) {
        ro_net_get_dynamic_shared(shared_bytes);
    }else{
        int num_pes = roc_shmem_n_pes();
        gpu_ib_get_dynamic_shared(shared_bytes, num_pes);
    }
    return Status::ROC_SHMEM_SUCCESS;
}

__host__ void
roc_shmem_query_thread(int *provided)
{
    /*
     * Host-facing functions always support full
     * thread flexibility i.e. THREAD_MULTIPLE.
     */
    *provided = ROC_SHMEM_THREAD_MULTIPLE;
}

__host__ void
roc_shmem_global_exit(int status)
{
    VERIFY_BACKEND();
    backend->global_exit(status);
}

/******************************************************************************
 ****************************** Teams Interface *******************************
 *****************************************************************************/

__host__ int
roc_shmem_team_n_pes(roc_shmem_team_t team)
{
    if (team == ROC_SHMEM_TEAM_INVALID) {
        return -1;
    } else {
        return get_internal_team(team)->num_pes;
    }
}

__host__ int
roc_shmem_team_my_pe(roc_shmem_team_t team)
{
    if (team == ROC_SHMEM_TEAM_INVALID) {
        return -1;
    } else {
        return get_internal_team(team)->my_pe;
    }
}

__host__ inline int
pe_in_active_set(int start, int stride, int size, int pe)
{
    /* Active set triplet is described with respect to team world */

    int translated_pe = (pe - start) / stride;

    if ((pe < start) ||
        ((pe - start) % stride) ||
        (translated_pe >= size)) {
        translated_pe = -1;
    }

    return translated_pe;
}

__host__ int
roc_shmem_team_split_strided(roc_shmem_team_t parent_team,
                             int start,
                             int stride,
                             int size,
                             const roc_shmem_team_config_t *config,
                             long config_mask,
                             roc_shmem_team_t *new_team)
{
    VERIFY_BACKEND();

    *new_team = ROC_SHMEM_TEAM_INVALID;

    auto num_user_teams {backend->team_tracker.get_num_user_teams()};
    auto max_num_teams {backend->team_tracker.get_max_num_teams()};
    if (num_user_teams >= max_num_teams - 1) {
        return static_cast<int>(Status::ROC_SHMEM_TOO_MANY_TEAMS_ERROR);
    }

    if (parent_team == ROC_SHMEM_TEAM_INVALID) {
        return 0; //TODO: is this the right return value?
    }

    Team *parent_team_obj = get_internal_team(parent_team);

    /* Santity check inputs */
    if (start < 0 || start >= parent_team_obj->num_pes ||
        size < 1 || size > parent_team_obj->num_pes || stride < 1) {
        return -1;
    }

    /* Calculate pe_start, stride, and pe_end wrt team world */
    int pe_start_in_world = parent_team_obj->get_pe_in_world(start);
    int stride_in_world   = stride * parent_team_obj->tinfo_wrt_world->stride;
    int pe_end_in_world   = pe_start_in_world + stride_in_world * (size - 1);

    /* Check if size is out of bounds */
    if (pe_end_in_world > backend->num_pes) {
        return -1;
    }

    /* Calculate my PE in the new team */
    int my_pe_in_world    = backend->my_pe;
    int my_pe_in_new_team = pe_in_active_set(pe_start_in_world, stride_in_world, size, my_pe_in_world);

    /* Create team infos */
    TeamInfo *team_info_wrt_parent, *team_info_wrt_world;

    CHECK_HIP(hipMalloc(&team_info_wrt_parent, sizeof(TeamInfo)));
    new (team_info_wrt_parent) TeamInfo(parent_team_obj,
                                        start,
                                        stride,
                                        size);

    auto* team_world {backend->team_tracker.get_team_world()};
    CHECK_HIP(hipMalloc(&team_info_wrt_world, sizeof(TeamInfo)));
    new (team_info_wrt_world) TeamInfo(team_world,
                                       pe_start_in_world,
                                       stride_in_world,
                                       size);

    /* Create a new MPI communicator for this team */
    int color;
    if (my_pe_in_new_team < 0) {
        color = MPI_UNDEFINED;
    } else {
        color = 1;
    }

    MPI_Comm team_comm;
    MPI_Comm_split(parent_team_obj->mpi_comm, color, my_pe_in_world, &team_comm);

    /**
     * Allocate new team for GPU-inittiated communication with backend-specific objects
     * TODO: are there any backend specific objects?
     */
    if (my_pe_in_new_team < 0) {
        *new_team = ROC_SHMEM_TEAM_INVALID;
    } else {
        backend->create_new_team(parent_team_obj,
                                 team_info_wrt_parent,
                                 team_info_wrt_world,
                                 size,
                                 my_pe_in_new_team,
                                 team_comm,
                                 new_team);

        /* Track the newly created team to destroy it in finalize if the user does not */
        backend->team_tracker.track(*new_team);
    }

    return 0;
}

__host__ void
roc_shmem_team_destroy(roc_shmem_team_t team)
{
    if (team == ROC_SHMEM_TEAM_INVALID ||
        team == ROC_SHMEM_TEAM_WORLD) {
        /* Do nothing */
        return;
    }

    backend->team_tracker.untrack(team);

    backend->team_destroy(team);
}

__host__ int
roc_shmem_team_translate_pe(roc_shmem_team_t src_team,
                            int src_pe,
                            roc_shmem_team_t dst_team)
{
    return team_translate_pe(src_team, src_pe, dst_team);
}

/******************************************************************************
 ************************** Default Context Wrappers **************************
 *****************************************************************************/

template <typename T>
__host__ void
roc_shmem_put(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_put(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_putmem(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_putmem(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ void
roc_shmem_p(T *dest, T value, int pe)
{
    roc_shmem_p(ROC_SHMEM_HOST_CTX_DEFAULT, dest, value, pe);
}

template <typename T>
__host__ void
roc_shmem_get(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_get(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_getmem(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_getmem(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ T
roc_shmem_g(const T *source, int pe)
{
    return roc_shmem_g(ROC_SHMEM_HOST_CTX_DEFAULT, source, pe);
}

template <typename T>
__host__ void
roc_shmem_put_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_put_nbi(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_putmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_putmem_nbi(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ void
roc_shmem_get_nbi(T *dest, const T *source, size_t nelems, int pe)
{
    roc_shmem_get_nbi(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

__host__ void
roc_shmem_getmem_nbi(void *dest, const void *source, size_t nelems, int pe)
{
    roc_shmem_ctx_getmem_nbi(ROC_SHMEM_HOST_CTX_DEFAULT, dest, source, nelems, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_fetch_add(T *dest, T val, int pe)
{
    return roc_shmem_atomic_fetch_add(ROC_SHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_compare_swap(T *dest, T cond, T val, int pe)
{
    return roc_shmem_atomic_compare_swap(ROC_SHMEM_HOST_CTX_DEFAULT,
                                        dest, cond, val, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_fetch_inc(T *dest, int pe)
{
    return roc_shmem_atomic_fetch_inc(ROC_SHMEM_HOST_CTX_DEFAULT, dest, pe);
}

template <typename T>
__host__ T
roc_shmem_atomic_fetch(T *dest, int pe)
{
    return roc_shmem_atomic_fetch(ROC_SHMEM_HOST_CTX_DEFAULT, dest, pe);
}

template <typename T>
__host__ void
roc_shmem_atomic_add(T *dest, T val, int pe)
{
    roc_shmem_atomic_add(ROC_SHMEM_HOST_CTX_DEFAULT, dest, val, pe);
}

template <typename T>
__host__ void
roc_shmem_atomic_inc(T *dest, int pe)
{
    roc_shmem_atomic_inc(ROC_SHMEM_HOST_CTX_DEFAULT, dest, pe);
}

__host__ void
roc_shmem_fence()
{
    roc_shmem_ctx_fence(ROC_SHMEM_HOST_CTX_DEFAULT);
}

__host__ void
roc_shmem_quiet()
{
    roc_shmem_ctx_quiet(ROC_SHMEM_HOST_CTX_DEFAULT);
}

/******************************************************************************
 ************************* Private Context Interfaces *************************
 *****************************************************************************/

__host__ Context *
get_internal_ctx(roc_shmem_ctx_t ctx)
{
    return reinterpret_cast<Context *>(ctx.ctx_opaque);
}

__host__ int
roc_shmem_ctx_create(int64_t options,
                     roc_shmem_ctx_t *ctx)
{
    DPRINTF("Host function: roc_shmem_ctx_create\n");

    void *phys_ctx;
    backend->ctx_create(options, &phys_ctx);

    ctx->ctx_opaque = phys_ctx;
    /* This team in on TEAM_WORLD, no need for team info */
    ctx->team_opaque = nullptr;

    /* Track this context, if needed. */
    backend->track_ctx(reinterpret_cast<Context *>(phys_ctx), options);

    return 0;
}

__host__ void
roc_shmem_ctx_destroy(roc_shmem_ctx_t ctx)
{
    DPRINTF("Host function: roc_shmem_ctx_destroy\n");

    /* TODO: Implicit quiet on this context */

    Context *phys_ctx = get_internal_ctx(ctx);

    backend->untrack_ctx(phys_ctx);

    backend->ctx_destroy(phys_ctx);
}

template <typename T> __host__ void
roc_shmem_put(roc_shmem_ctx_t ctx, T *dest, const T *source,
              size_t nelems, int pe)
{
    DPRINTF("Host function: roc_shmem_put\n");

    get_internal_ctx(ctx)->put(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_putmem(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF("Host function: roc_shmem_ctx_putmem\n");

    get_internal_ctx(ctx)->putmem(dest, source, nelems, pe);
}

template <typename T> __host__ void
roc_shmem_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe)
{
    DPRINTF("Host function: roc_shmem_p\n");

    get_internal_ctx(ctx)->p(dest, value, pe);
}

template <typename T> __host__ void
roc_shmem_get(roc_shmem_ctx_t ctx, T *dest, const T *source, size_t nelems,
              int pe)
{
    DPRINTF("Host function: roc_shmem_get\n");

    get_internal_ctx(ctx)->get(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_getmem(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF("Host function: roc_shmem_ctx_getmem\n");

    get_internal_ctx(ctx)->getmem(dest, source, nelems, pe);
}

template <typename T> __host__ T
roc_shmem_g(roc_shmem_ctx_t ctx, const T *source, int pe)
{
    DPRINTF("Host function: roc_shmem_g\n");

    return get_internal_ctx(ctx)->g(source, pe);
}

template <typename T> __host__ void
roc_shmem_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    DPRINTF("Host function: roc_shmem_put_nbi\n");

    get_internal_ctx(ctx)->put_nbi(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_putmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF("Host function: roc_shmem_ctx_putmem_nbi\n");

    get_internal_ctx(ctx)->putmem_nbi(dest, source, nelems, pe);
}

template <typename T>
__host__ void
roc_shmem_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source,
                  size_t nelems, int pe)
{
    DPRINTF("Host function: roc_shmem_get_nbi\n");

    get_internal_ctx(ctx)->get_nbi(dest, source, nelems, pe);
}

__host__ void
roc_shmem_ctx_getmem_nbi(roc_shmem_ctx_t ctx, void *dest, const void *source, size_t nelems, int pe)
{
    DPRINTF("Host function: roc_shmem_ctx_getmem_nbi\n");

    get_internal_ctx(ctx)->getmem_nbi(dest, source, nelems, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    DPRINTF("Host function: roc_shmem_atomic_fetch_add\n");

    return get_internal_ctx(ctx)->amo_fetch_add(dest, val, 0, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest, T cond, T val,
                             int pe)
{
    DPRINTF("Host function: roc_shmem_atomic_compare_swap\n");

    return get_internal_ctx(ctx)->amo_fetch_cas(dest, val, cond, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    DPRINTF("Host function: roc_shmem_atomic_fetch_inc\n");

    return get_internal_ctx(ctx)->amo_fetch_add(dest, 1, 0, pe);
}

template <typename T> __host__ T
roc_shmem_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    DPRINTF("Host function: roc_shmem_atomic_fetch\n");

    return get_internal_ctx(ctx)->amo_fetch_add(dest, 0, 0, pe);
}

template <typename T> __host__ void
roc_shmem_atomic_add(roc_shmem_ctx_t ctx, T *dest, T val, int pe)
{
    DPRINTF("Host function: roc_shmem_atomic_add\n");

    get_internal_ctx(ctx)->amo_add((void*)dest, val, 0, pe);
}

template <typename T>
__host__ void
roc_shmem_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe)
{
    DPRINTF("Host function: roc_shmem_atomic_inc\n");

    get_internal_ctx(ctx)->amo_add(dest, 1, 0, pe);
}

__host__ void
roc_shmem_ctx_fence(roc_shmem_ctx_t ctx)
{
    DPRINTF("Host function: roc_shmem_ctx_fence\n");

    get_internal_ctx(ctx)->fence();
}

__host__ void
roc_shmem_ctx_quiet(roc_shmem_ctx_t ctx)
{
    DPRINTF("Host function: roc_shmem_ctx_quiet\n");

    get_internal_ctx(ctx)->quiet();
}

__host__ void
roc_shmem_barrier_all()
{
    DPRINTF("Host function: roc_shmem_barrier_all\n");

    get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->barrier_all();
}

__host__ void
roc_shmem_sync_all()
{
    DPRINTF("Host function: roc_shmem_sync_all\n");

    get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->sync_all();
}

template <typename T>
__host__ void
roc_shmem_broadcast(roc_shmem_ctx_t ctx,
                    T *dest,
                    const T *source,
                    int nelem,
                    int pe_root,
                    int pe_start,
                    int log_pe_stride,
                    int pe_size,
                    long *p_sync)
{
    DPRINTF("Host function: roc_shmem_broadcast\n");

    get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->broadcast<T>(dest,
                                             source,
                                             nelem,
                                             pe_root,
                                             pe_start,
                                             log_pe_stride,
                                             pe_size,
                                             p_sync);
}

template <typename T>
__host__ void
roc_shmem_broadcast(roc_shmem_ctx_t ctx,
                    roc_shmem_team_t team,
                    T *dest,
                    const T *source,
                    int nelem,
                    int pe_root)
{
    DPRINTF("Host function: Team-based roc_shmem_broadcast\n");

    get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->broadcast<T>(team,
                                                               dest,
                                                               source,
                                                               nelem,
                                                               pe_root);
}

template <typename T, ROC_SHMEM_OP Op> __host__ void
roc_shmem_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source,
                 int nreduce, int PE_start, int logPE_stride,
                 int PE_size, T *pWrk, long *pSync)
{
    DPRINTF("Host function: roc_shmem_to_all\n");

    get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->to_all<T, Op>(dest, source, nreduce, PE_start,
                                         logPE_stride, PE_size, pWrk, pSync);
}

template <typename T, ROC_SHMEM_OP Op> __host__ void
roc_shmem_to_all(roc_shmem_ctx_t ctx, roc_shmem_team_t team,
                 T *dest, const T *source, int nreduce)
{
    DPRINTF("Host function: Team-based roc_shmem_to_all\n");

    get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->to_all<T, Op>(team, dest, source, nreduce);
}

template <typename T>
__host__ void
roc_shmem_wait_until(T *ptr, roc_shmem_cmps cmp, T val)
{
    DPRINTF("Host function: roc_shmem_wait_until\n");

    get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->wait_until(ptr, cmp, val);
}

template <typename T>
__host__ int
roc_shmem_test(T *ptr, roc_shmem_cmps cmp, T val)
{
    DPRINTF("Host function: roc_shmem_testl\n");

    return get_internal_ctx(ROC_SHMEM_HOST_CTX_DEFAULT)->test(ptr, cmp, val);
}

/**
 * Template generator for reductions
 **/
#define REDUCTION_GEN(T, Op) \
    template __host__ void \
    roc_shmem_to_all<T, Op>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                            int nreduce, int PE_start, int logPE_stride, \
                            int PE_size, T *pWrk, long *pSync); \
    template __host__ void \
    roc_shmem_to_all<T, Op>(roc_shmem_ctx_t ctx, roc_shmem_team_t team, \
                            T *dest, const T *source, int nreduce);

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
 * Declare templates for the required datatypes (for the compiler)
 **/
#define RMA_GEN(T) \
    template __host__ void \
    roc_shmem_put<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                     size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_put_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_p<T>(roc_shmem_ctx_t ctx, T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_get<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                     size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_get_nbi<T>(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                         size_t nelems, int pe); \
    template __host__ T \
    roc_shmem_g<T>(roc_shmem_ctx_t ctx, const T *source, int pe); \
    template __host__ void \
    roc_shmem_put<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_put_nbi<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_p<T>(T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_get<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ void \
    roc_shmem_get_nbi<T>(T *dest, const T *source, size_t nelems, int pe); \
    template __host__ T \
    roc_shmem_g<T>(const T *source, int pe); \
    template __host__ void \
    roc_shmem_broadcast<T>(roc_shmem_ctx_t ctx, \
                           T *dest, \
                           const T *source, \
                           int nelem, \
                           int pe_root, \
                           int pe_start, \
                           int log_pe_stride, \
                           int pe_size, \
                           long *p_sync); \
    template __host__ void \
    roc_shmem_broadcast<T>(roc_shmem_ctx_t ctx, \
                           roc_shmem_team_t team, \
                           T *dest, \
                           const T *source, \
                           int nelem, \
                           int pe_root);

#define AMO_GEN(T) \
    template __host__ T \
    roc_shmem_atomic_fetch_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, \
                                  int pe); \
    template __host__ T \
    roc_shmem_atomic_compare_swap<T>(roc_shmem_ctx_t ctx,  T *dest, T cond, \
                                    T value, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __host__ void \
    roc_shmem_atomic_add<T>(roc_shmem_ctx_t ctx,  T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_atomic_inc<T>(roc_shmem_ctx_t ctx,  T *dest, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch_add<T>(T *dest, T value, int pe); \
    template __host__ T \
    roc_shmem_atomic_compare_swap<T>(T *dest, T cond, T value, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch_inc<T>(T *dest, int pe); \
    template __host__ T \
    roc_shmem_atomic_fetch<T>(T *dest, int pe); \
    template __host__ void \
    roc_shmem_atomic_add<T>(T *dest, T value, int pe); \
    template __host__ void \
    roc_shmem_atomic_inc<T>(T *dest, int pe);

#define WAIT_GEN(T) \
    template __host__ void \
    roc_shmem_wait_until<T>(T *ptr, roc_shmem_cmps cmp, T val); \
    template __host__ int \
    roc_shmem_test<T>(T *ptr, roc_shmem_cmps cmp, T val);


/**
* Define APIs to call the template functions
**/

#define REDUCTION_DEF_GEN(T, TNAME, Op_API, Op) \
    __host__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_to_all(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                              int nreduce, int PE_start, int logPE_stride, \
                                              int PE_size, T *pWrk, long *pSync) \
    { \
        roc_shmem_to_all<T, Op>(ctx, dest, source, nreduce, PE_start, \
                                logPE_stride, PE_size, pWrk, pSync); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_##Op_API##_to_all(roc_shmem_ctx_t ctx, roc_shmem_team_t team, \
                                              T *dest, const T *source, int nreduce) \
    { \
        roc_shmem_to_all<T, Op>(ctx, team, dest, source, nreduce); \
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
    __host__ void \
    roc_shmem_ctx_##TNAME##_put(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                size_t nelems, int pe) \
    { \
        roc_shmem_put<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_put_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                    size_t nelems, int pe) \
    { \
        roc_shmem_put_nbi<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_p(roc_shmem_ctx_t ctx, T *dest, T value, int pe) \
    { \
        roc_shmem_p<T>(ctx, dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_get(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                size_t nelems, int pe) \
    { \
        roc_shmem_get<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_get_nbi(roc_shmem_ctx_t ctx, T *dest, const T *source, \
                                    size_t nelems, int pe) \
    { \
        roc_shmem_get_nbi<T>(ctx, dest, source, nelems, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_g(roc_shmem_ctx_t ctx, const T *source, int pe) \
    { \
        return roc_shmem_g<T>(ctx, source, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_put(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_put<T>(dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_put_nbi(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_put_nbi<T>(dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_p(T *dest, T value, int pe) \
    { \
        roc_shmem_p<T>(dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_get(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_get<T>(dest, source, nelems, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_get_nbi(T *dest, const T *source, size_t nelems, int pe) \
    { \
        roc_shmem_get_nbi<T>(dest, source, nelems, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_g(const T *source, int pe) \
    { \
        return roc_shmem_g<T>(source, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_broadcast(roc_shmem_ctx_t ctx, \
                                      T *dest, \
                                      const T *source, \
                                      int nelem, \
                                      int pe_root, \
                                      int pe_start, \
                                      int log_pe_stride, \
                                      int pe_size, \
                                      long *p_sync) \
    { \
        roc_shmem_broadcast<T>(ctx, dest, source, nelem, pe_root, pe_start, \
                               log_pe_stride, pe_size, p_sync); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_broadcast(roc_shmem_ctx_t ctx, \
                                      roc_shmem_team_t team, \
                                      T *dest, \
                                      const T *source, \
                                      int nelem, \
                                      int pe_root) \
    { \
        roc_shmem_broadcast<T>(ctx, team, dest, source, nelem, pe_root); \
    }

#define AMO_DEF_GEN(T, TNAME) \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_add(roc_shmem_ctx_t ctx, T *dest, T value, \
                                             int pe) \
    { \
        return roc_shmem_atomic_fetch_add<T>(ctx, dest, value, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_compare_swap(roc_shmem_ctx_t ctx, T *dest, T cond, \
                                               T value, int pe) \
    { \
        return roc_shmem_atomic_compare_swap<T>(ctx, dest, cond, value, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch_inc(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch_inc<T>(ctx, dest, pe); \
    } \
    __host__ T \
    roc_shmem_ctx_##TNAME##_atomic_fetch(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch<T>(ctx, dest, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_atomic_add(roc_shmem_ctx_t ctx, T *dest, T value, int pe) \
    { \
        roc_shmem_atomic_add<T>(ctx, dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_ctx_##TNAME##_atomic_inc(roc_shmem_ctx_t ctx, T *dest, int pe) \
    { \
        roc_shmem_atomic_inc<T>(ctx, dest, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch_add(T *dest, T value, int pe) \
    { \
        return roc_shmem_atomic_fetch_add<T>(dest, value, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_compare_swap(T *dest, T cond, T value, int pe) \
    { \
        return roc_shmem_atomic_compare_swap<T>(dest, cond, value, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch_inc(T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch_inc<T>(dest, pe); \
    } \
    __host__ T \
    roc_shmem_##TNAME##_atomic_fetch(T *dest, int pe) \
    { \
        return roc_shmem_atomic_fetch<T>(dest, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_atomic_add(T *dest, T value, int pe) \
    { \
        roc_shmem_atomic_add<T>(dest, value, pe); \
    } \
    __host__ void \
    roc_shmem_##TNAME##_atomic_inc(T *dest, int pe) \
    { \
        roc_shmem_atomic_inc<T>(dest, pe); \
    }

#define WAIT_DEF_GEN(T, TNAME) \
    __host__ void \
    roc_shmem_##TNAME##_wait_until(T *ptr, roc_shmem_cmps cmp, T val) \
    { \
        roc_shmem_wait_until<T>(ptr, cmp, val); \
    } \
    __host__ int \
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
WAIT_GEN(float) WAIT_GEN(double) WAIT_GEN(char) //WAIT_GEN(long double)
WAIT_GEN(signed char) WAIT_GEN(short) WAIT_GEN(int) WAIT_GEN(long)
WAIT_GEN(long long) WAIT_GEN(unsigned char) WAIT_GEN(unsigned short)
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

WAIT_DEF_GEN(float, float)
WAIT_DEF_GEN(double, double)
WAIT_DEF_GEN(char, char)
//WAIT_DEF_GEN(long double, longdouble)
WAIT_DEF_GEN(signed char, schar)
WAIT_DEF_GEN(short, short)
WAIT_DEF_GEN(int, int)
WAIT_DEF_GEN(long, long)
WAIT_DEF_GEN(long long, longlong)
WAIT_DEF_GEN(unsigned char, uchar)
WAIT_DEF_GEN(unsigned short, ushort)
WAIT_DEF_GEN(unsigned int, uint)
WAIT_DEF_GEN(unsigned long, ulong)
WAIT_DEF_GEN(unsigned long long, ulonglong)

}  // namespace rocshmem
