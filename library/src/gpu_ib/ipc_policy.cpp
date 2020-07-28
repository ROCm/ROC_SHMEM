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

#include "config.h"

#ifdef DEBUG
#define HIP_ENABLE_PRINTF 1
#endif

#include "context.hpp"
#include "backend.hpp"
#include "wg_state.hpp"
#include "util.hpp"
#include <mpi.h>



__host__ uint32_t
IpcOnImpl::ipcDynamicShared()
{
    return (shm_size * sizeof(uintptr_t));
}
__host__ void
IpcOnImpl::ipcHostInit(int my_pe, char** heap_bases)
{
    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        0,
                        MPI_INFO_NULL,
                        &shmcomm);

    int Shm_size;
    MPI_Comm_size(shmcomm, &Shm_size);

    int shm_rank;
    MPI_Comm_rank(shmcomm, &shm_rank);

    hipIpcMemHandle_t *vec_ipc_handle =
        (hipIpcMemHandle_t*) malloc(sizeof(hipIpcMemHandle_t) * Shm_size);

    char * base_heap = heap_bases[my_pe];
    CHECK_HIP(hipIpcGetMemHandle(&vec_ipc_handle[shm_rank], base_heap));

    MPI_Allgather(MPI_IN_PLACE,
                  sizeof(hipIpcMemHandle_t),
                  MPI_CHAR,
                  vec_ipc_handle,
                  sizeof(hipIpcMemHandle_t),
                  MPI_CHAR,
                  shmcomm);

    char ** ipc_base;
    CHECK_HIP(hipMalloc((void**)&ipc_base, sizeof(char**) * Shm_size));

    for (int i = 0; i < Shm_size; i++) {
        if (i != shm_rank) {
            CHECK_HIP(hipIpcOpenMemHandle((void**)&ipc_base[i],
                                          vec_ipc_handle[i],
                                          hipIpcMemLazyEnablePeerAccess));
        } else {
            ipc_base[i] = base_heap;
        }
    }
    shm_size = Shm_size;
    ipc_bases = ipc_base;

    free(vec_ipc_handle);

}

__device__ void
IpcOnImpl::ipcGpuInit(GPUIBBackend* gpu_backend, GPUIBContext* ctx,
                      int thread_id)
{
    GPU_DPRINTF("Function: ipcGpuInit \n");
    shm_size  = gpu_backend->ipcImpl.shm_size;
    char ** ipc_base_lds = reinterpret_cast<char **>(
        WGState::instance()->allocateDynamicShared(shm_size *
                                                   sizeof(uintptr_t)));

    for (int i = thread_id; i < shm_size; i++)
        ipc_base_lds[i]= gpu_backend->ipcImpl.ipc_bases[i];

    ipc_bases =  (ipc_base_lds);

}
__device__ void
IpcOnImpl::ipcCopy(void * dst, void* src, size_t size)
{
    memcpy(dst, src, size);
}

