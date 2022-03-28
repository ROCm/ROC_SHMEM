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

/*
 * ROC_SHMEM roc_shmem_team_translate example to verify the team formed by even
 * ranked PEs from ROC_SHMEM_TEAM_WORLD using the team created from
 * roc_shmem_team_split_stride operation
 */

#include <roc_shmem.hpp>
#include <stdio.h>

using namespace rocshmem;

int main(void)
{
    int                  my_pe, npes, errors = 0;
    int                  t_pe_2, t_pe_3, t_pe_2_to_3, t_pe_3_to_2;
    roc_shmem_team_t        team_2s;
    roc_shmem_team_t        team_3s;
    roc_shmem_team_config_t *config;

    roc_shmem_init(1);
    config = NULL;
    my_pe  = roc_shmem_my_pe();
    npes   = roc_shmem_n_pes();

    roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD, 0, 2, ((npes-1)/2)+1, config, 0,
                             &team_2s);
    roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD, 0, 3, ((npes-1)/3)+1, config, 0,
                             &team_3s);

    t_pe_3 = roc_shmem_team_my_pe(team_3s);
    t_pe_2 = roc_shmem_team_my_pe(team_2s);
    t_pe_3_to_2 = roc_shmem_team_translate_pe(team_3s, t_pe_3, team_2s);
    t_pe_2_to_3 = roc_shmem_team_translate_pe(team_2s, t_pe_2, team_3s);

    if (my_pe % 2 == 0 && my_pe % 3 == 0) {
        if (t_pe_2 == -1 || t_pe_3 == -1 || t_pe_2_to_3 == -1 || t_pe_3_to_2 == -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n",
                   my_pe, t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    } else if (my_pe % 2 == 0) {
        if (t_pe_2 == -1 || t_pe_3 != -1 || t_pe_2_to_3 != -1 || t_pe_3_to_2 != -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n",
                   my_pe, t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    } else if (my_pe % 3 == 0){
        if (t_pe_2 != -1 || t_pe_3 == -1 || t_pe_2_to_3 != -1 || t_pe_3_to_2 != -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n",
                   my_pe, t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    } else {
        if (t_pe_2 != -1 || t_pe_3 != -1 || t_pe_2_to_3 != -1 || t_pe_3_to_2 != -1) {
            printf("ERROR: PE %d, t_pe_2=%d, t_pe_3=%d, t_pe_3_to_2=%d, t_pe_2_to_3=%d\n",
                   my_pe, t_pe_2, t_pe_3, t_pe_3_to_2, t_pe_2_to_3);
            ++errors;
        }
    }

    roc_shmem_finalize();
    return errors != 0;
}
