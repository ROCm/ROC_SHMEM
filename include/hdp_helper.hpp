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

#ifndef HDP_HELPER_H
#define HDP_HELPER_H

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

int hdp_count;

inline void
hdp_flush(hsa_amd_hdp_flush_t *hdp_regs)
{
    for (int i = 0; i < hdp_count; i++) {
        *(hdp_regs[i].HDP_MEM_FLUSH_CNTL) = 0x1;
    }
}

inline void
hdp_read_inv(hsa_amd_hdp_flush_t *hdp_regs)
{
    hdp_flush(hdp_regs);
}

static hsa_status_t get_gpu_agent(hsa_agent_t agent, void *data) {
    hsa_status_t status;
    hsa_device_type_t device_type;
    status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
    if (HSA_STATUS_SUCCESS == status && HSA_DEVICE_TYPE_GPU == device_type) {
        hsa_agent_t* ret = (hsa_agent_t*)data;
        *ret = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

inline hsa_amd_hdp_flush_t *
hdp_map_all()
{
    hdp_count = 1;

    hsa_agent_t agent;
    hsa_iterate_agents(get_gpu_agent, &agent);

    hsa_amd_hdp_flush_t *hdp_regs;
    hipHostMalloc(&hdp_regs, sizeof(hsa_amd_hdp_flush_t) * hdp_count);

    hdp_regs[0].HDP_REG_FLUSH_CNTL = 0;
    hdp_regs[0].HDP_MEM_FLUSH_CNTL = 0;

    hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HDP_FLUSH), hdp_regs);

    printf("HDP REG flush %p MEM flush %p\n",
           hdp_regs[0].HDP_REG_FLUSH_CNTL,
           hdp_regs[0].HDP_MEM_FLUSH_CNTL);

    return hdp_regs;
}

#endif
