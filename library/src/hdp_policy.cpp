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

#include "config.h"

#include "hdp_policy.hpp"
#include "util.hpp"

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

HdpMapPolicy::HdpMapPolicy()
{
    hdp_flush_off = HDP_FLUSH % getpagesize();
    hdp_flush_pa_off = HDP_FLUSH - hdp_flush_off;
    hdp_read_off = HDP_READ % getpagesize();
    hdp_read_pa_off = HDP_READ - hdp_read_off;

    // TODO: Multi-device?
    int dev_id = 0;

    std::string name = "/dev/hdp_umap_";
    name += std::to_string(dev_id);

    if ((fd = open(name.c_str(), O_RDWR)) == -1) {
        printf("could not open the HDP dev module \n");
        exit(-1);
    }

    /*
     * Unclear why this needs to happen if it is unused, but IB won't
     * register cpu_hdp_flush later for RTN if this is not here.
     */
    mmap(nullptr, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd,
         hdp_read_pa_off);

    cpu_hdp_flush = (unsigned int*)
        mmap(nullptr, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd,
             hdp_flush_pa_off);

    cpu_hdp_flush += hdp_flush_off / 4;

    void* dev_ptr;
    rocm_memory_lock_to_fine_grain(cpu_hdp_flush, getpagesize(), &dev_ptr,
                                   dev_id);

    gpu_hdp_flush = (unsigned int *) dev_ptr;
}

HdpRocmPolicy::HdpRocmPolicy()
{
    // TODO: Multi-device?
    int dev_id = 0;

    hdp = rocm_hdp();
    cpu_hdp_flush = hdp[dev_id].HDP_MEM_FLUSH_CNTL;
    gpu_hdp_flush = cpu_hdp_flush;
}
