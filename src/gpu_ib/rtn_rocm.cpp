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

#include <stdio.h>
#include <stdlib.h>

#include "rtn_rocm.hpp"
#include "hdp_helper.hpp"
#include "util.hpp"

typedef struct device_agent {
    hsa_agent_t agent;
    hsa_amd_memory_pool_t pool;
} device_agent_t;

std::vector<device_agent_t> gpu_agents;
std::vector<device_agent_t> cpu_agents;

hsa_status_t
rtn_rocm_hsa_amd_memory_pool_callback(hsa_amd_memory_pool_t memory_pool,
                                      void* data)
{
    hsa_status_t status;
    hsa_amd_memory_pool_global_flag_t pool_flag;

    status =
        hsa_amd_memory_pool_get_info(memory_pool,
                                     HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                     &pool_flag);

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failure to get pool info: 0x%x", status);
        return status;
    }

    if (pool_flag == (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
        HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED )  ) {
        *(hsa_amd_memory_pool_t *)data = memory_pool;
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
rtn_rocm_hsa_agent_callback(hsa_agent_t agent, void* data)
{
    hsa_device_type_t device_type;
    hsa_status_t status;

    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failure to get device type: 0x%x", status);
        return status;
    }

    if (device_type == HSA_DEVICE_TYPE_GPU) {
        gpu_agents.emplace_back();
        gpu_agents.back().agent = agent;
        status = hsa_amd_agent_iterate_memory_pools(
                                        agent,
                                        rtn_rocm_hsa_amd_memory_pool_callback,
                                        &(gpu_agents.back().pool));
    }

    if (device_type == HSA_DEVICE_TYPE_CPU) {
        cpu_agents.emplace_back();
        cpu_agents.back().agent = agent;
        status = hsa_amd_agent_iterate_memory_pools(
                                        agent,
                                        rtn_rocm_hsa_amd_memory_pool_callback,
                                        &(cpu_agents.back().pool));
    }

    return status;
}

hsa_amd_hdp_flush_t *
rtn_rocm_hdp(void)
{
   hsa_amd_hdp_flush_t * hdp;
   hipHostMalloc((void**) &hdp,
                 sizeof(hsa_amd_hdp_flush_t) * gpu_agents.size());

   for (int i = 0; i < gpu_agents.size(); i++) {
        hdp[i].HDP_REG_FLUSH_CNTL = 0;
        hdp[i].HDP_MEM_FLUSH_CNTL = 0;
        hsa_agent_get_info(gpu_agents[i].agent,
            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HDP_FLUSH),
            &hdp[i]);
   }

   return hdp;
}

#ifndef _USE_HDP_MAP_
hdp_reg_t *
rtn_hdp_flush_map(int rtn_id)
{
    hdp_reg_t * rtn_hdp;
    hsa_amd_hdp_flush_t * hdp;

    hipMalloc((void**) &rtn_hdp, sizeof(hdp_reg_t));
    hdp = rtn_rocm_hdp();
    rtn_hdp->cpu_hdp_flush = hdp[rtn_id].HDP_MEM_FLUSH_CNTL;
    rtn_hdp->gpu_hdp_flush = rtn_hdp->cpu_hdp_flush;

    DPRINTF(("hdp regs cpu %p  gpu %p\n", rtn_hdp->cpu_hdp_flush,
           rtn_hdp->gpu_hdp_flush));

   return rtn_hdp;
}
#endif

int
rtn_rocm_init()
{
    hsa_status_t status;
    status = hsa_init();

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failure to open HSA connection: 0x%x", status);
        goto end;
    }

    /* Collect information about GPU agents */
    status = hsa_iterate_agents(rtn_rocm_hsa_agent_callback, NULL);

    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
        printf("Failure to iterate HSA agents: 0x%x", status);
        goto end;
    }
    return 0;

end:
    return 1;
}

void
rtn_rocm_memory_lock(void *ptr, size_t size, void **gpu_ptr, int gpu_id)
{
    hsa_status_t status = hsa_amd_memory_lock(ptr, size,
                                              &(gpu_agents[gpu_id].agent),
                                              1, gpu_ptr);

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failed to lock memory (%p): 0x%x\n", ptr, status);
        exit(-1);
    }
}

void
rtn_rocm_memory_unlock(void *host_ptr)
{
    hsa_status_t status =  hsa_amd_memory_unlock(host_ptr);

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failed to unlock memory (%p): 0x%x\n", host_ptr, status);
        exit(-1);
     }
}

void
rtn_rocm_memory_lock_to_fine_grain(void *ptr, size_t size, void **gpu_ptr,
                                   int gpu_id)
{
    hsa_status_t status =
        hsa_amd_memory_lock_to_pool(ptr, size, &(gpu_agents[gpu_id].agent),
                                    1, cpu_agents[0].pool, 0, gpu_ptr);

    if (status != HSA_STATUS_SUCCESS) {
        printf("Failed to lock memory pool (%p): 0x%x\n", ptr, status);
        exit(-1);
    }
}
