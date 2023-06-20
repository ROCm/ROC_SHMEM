# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#!/bin/bash

if [ $# -eq 0 ] ; then
    echo "This script must be run with at least 2 arguments."
    echo 'Usage: ${0} argument1 argument2 [argument3] [argument4] [argument5]'
    echo "  argument1 : path to the tester driver"
    echo "  argument2 : test type to run, e.g put"
    echo "  argument3 : directory to put the output logs"
    echo "  argument4 : enable gdb debug"
    echo "  argument5 : shmem context type"
    exit 1
fi

shm_ctx=4  # run tests with SHMEM_CONTEXT_WG_PRIVATE
if [ $# -eq 5 ] ; then
    shm_ctx=$5
fi

ENABLE_DBG=false
if [ $# -ge 4 ] ; then
    ENABLE_DBG=$4
fi

if [ "$ENABLE_DBG" = true ]
then
    gdb_cmd="xterm -e gdb -x gdbscript --args"
else
    gdb_cmd=""
fi

echo "Test Name ${2} (with shmem context: ${shm_ctx})"

check() {
    if [ $? -ne 0 ]
    then
        echo "Failed $1" >&2
        exit 1
    fi
}

case $2 in
    *"single_thread")
        echo "get"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 0 -x ${shm_ctx} > $3/get.log
        check get
        echo "get_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 0 -x ${shm_ctx} > $3/get_th.log
        check get_th
        echo "get_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 1 -x ${shm_ctx} > $3/get_nbi.log
        check get_nbi
        echo "get_nbi_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 1 -x ${shm_ctx} > $3/get_nbi_th.log
        check get_nbi_th
        echo "put"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 2 -x ${shm_ctx} > $3/put.log
        check put
        echo "put_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 2 -x ${shm_ctx} > $3/put_th.log
        check put_th
        echo "put_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 3 -x ${shm_ctx} > $3/put_nbi.log
        check put_nbi
        echo "put_nbi_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 3 -x ${shm_ctx} > $3/put_nbi_th.log
        check put_nbi_th
#        echo "team_ctx_infra"
#        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 42 -x ${shm_ctx} > $3/team_ctx_infra.log
#        check team_ctx_infra
        echo "team_ctx_put_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 41 -x ${shm_ctx} > $3/team_ctx_put_nbi.log
        check team_ctx_put_nbi
        echo "team_ctx_put_nbi_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 41 -x ${shm_ctx} > $3/team_ctx_put_nbi_th.log
        check team_ctx_put_nbi_th
        echo "team_ctx_get_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 39 -x ${shm_ctx} > $3/team_ctx_get_nbi.log
        check team_ctx_get_nbi
        echo "team_ctx_get_nbi_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 39 -x ${shm_ctx} > $3/team_ctx_get_nbi_th.log
        check team_ctx_get_nbi_th
        echo "reduction"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 5 -x ${shm_ctx} > $3/reduction.log
        check reduction
        echo "amo_fadd"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 6 -x ${shm_ctx} > $3/amo_fadd.log
        check amo_fadd
        echo "amo_fadd_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 6 -x ${shm_ctx} > $3/amo_fadd_th.log
        check amo_fadd_th
        echo "amo_finc"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 7 -x ${shm_ctx} > $3/amo_finc.log
        check amo_finc
        echo "amo_finc_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 7 -x ${shm_ctx} > $3/amo_finc_th.log
        check amo_finc_th
        echo "amo_fetch"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 8 -x ${shm_ctx} > $3/amo_fetch.log
        check amo_fetch
        echo "amo_fetch_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 8 -x ${shm_ctx} > $3/amo_fetch_th.log
        check amo_fetch_th
        echo "amo_fcswap"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 9 -x ${shm_ctx} > $3/amo_fcswap.log
        check amo_fcswap
        echo "amo_fcswap_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 9 -x ${shm_ctx} > $3/amo_fcswap_th.log
        check amo_fcswap_th
        echo "amo_add"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 10 -x ${shm_ctx} > $3/amo_add.log
        check amo_add
        echo "amo_add_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 10 -x ${shm_ctx} > $3/amo_add_th.log
        check amo_add_th
        echo "amo_set"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 44 -x ${shm_ctx} > $3/amo_set.log
        check amo_set
        echo "amo_set_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 44 -x ${shm_ctx} > $3/amo_set_th.log
        check amo_set_th
        echo "amo_inc"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 11 -x ${shm_ctx} > $3/amo_inc.log
        check amo_inc
        echo "amo_inc_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 11 -x ${shm_ctx} > $3/amo_inc_th.log
        check amo_inc_th
        echo "init"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 13 -x ${shm_ctx} > $3/init.log
        check init
        echo "ping_pong"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 14 -x ${shm_ctx} > $3/ping_pong.log
        check ping_pong
        echo "ping_pong_th"
        mpirun -np 2 ${gdb_cmd} $1 -t 1024 -w 25 -s 32768 -a 14 -x ${shm_ctx} > $3/ping_pong_th.log
        check ping_pong_th
        echo "barrier_all"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 17 -x ${shm_ctx} > $3/barrier_all.log
        check barrier_all
        echo "sync_all"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 18 -x ${shm_ctx} > $3/sync_all.log
        check sync_all
        echo "sync"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 19 -x ${shm_ctx} > $3/sync.log
        check sync
        echo "alltoall"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 23 -x ${shm_ctx} > $3/alltoall.log
        check alltoall
        echo "fcollect"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 22 -x ${shm_ctx} > $3/fcollect.log
        check fcollect
        ;;
    *"multi_thread")
        echo "get"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 0 -x ${shm_ctx} > $3/get.log
        check get
        echo "get_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 1 -x ${shm_ctx} > $3/get_nbi.log
        check get_nbi
        echo "put"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 2 -x ${shm_ctx} > $3/put.log
        check put
        echo "put_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 3 -x ${shm_ctx} > $3/put_nbi.log
        check put_nbi
#        echo "team_ctx_infra"
#        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 42 -x ${shm_ctx} > $3/team_ctx_infra.log
#        check team_ctx_infra
        echo "team_ctx_put_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 41 -x ${shm_ctx} > $3/team_ctx_put_nbi.log
        check team_ctx_put_nbi
        echo "team_ctx_get_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 39 -x ${shm_ctx} > $3/team_ctx_get_nbi.log
        check team_ctx_get_nbi
        echo "get_swarm"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 4 -x ${shm_ctx} > $3/get_swarm.log
        check get_swarm
        echo "reduction"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 5 -x ${shm_ctx} > $3/reduction.log
        check reduction
        echo "amo_fadd"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 6 -x ${shm_ctx} > $3/amo_fadd.log
        check amo_fadd
        echo "amo_finc"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 7 -x ${shm_ctx} > $3/amo_finc.log
        check amo_finc
        echo "amo_fetch"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 8 -x ${shm_ctx} > $3/amo_fetch.log
        check amo_fetch
        echo "amo_fcswap"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 9 -x ${shm_ctx} > $3/amo_fcswap.log
        check amo_fcswap
        echo "amo_add"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 10 -x ${shm_ctx} > $3/amo_add.log
        check amo_add
        echo "amo_set"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 44 -x ${shm_ctx} > $3/amo_set.log
        check amo_set
        echo "amo_inc"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 11 -x ${shm_ctx} > $3/amo_inc.log
        check amo_inc
        echo "init"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 13 -x ${shm_ctx} > $3/init.log
        check init
        echo "ping_pong"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 14 -x ${shm_ctx} > $3/ping_pong.log
        check ping_pong
        echo "barrier_all"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 17 -x ${shm_ctx} > $3/barrier_all.log
        check barrier_all
        echo "sync_all"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 18 -x ${shm_ctx} > $3/sync_all.log
        check sync_all
        echo "sync"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 19 -x ${shm_ctx} > $3/sync.log
        check sync
        echo "alltoall"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 23 -x ${shm_ctx} > $3/alltoall.log
        check alltoall
        echo "fcollect"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 22 -x ${shm_ctx} > $3/fcollect.log
        check fcollect
        ;;
    *"ro")
        echo "get"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 0 -x ${shm_ctx} > $3/get.log
        check get
        echo "get_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 1 -x ${shm_ctx} > $3/get_nbi.log
        check get_nbi
        echo "put"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 2 -x ${shm_ctx} > $3/put.log
        check put
        echo "put_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 3 -x ${shm_ctx} > $3/put_nbi.log
        check put_nbi
#        echo "team_ctx_infra"
#        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 42 -x ${shm_ctx} > $3/team_ctx_infra.log
#        check team_ctx_infra
        echo "team_ctx_put_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 41 -x ${shm_ctx} > $3/team_ctx_put_nbi.log
        check team_ctx_put_nbi
        echo "team_ctx_get_nbi"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 39 -x ${shm_ctx} > $3/team_ctx_get_nbi.log
        check team_ctx_get_nbi
#        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 4 -x ${shm_ctx} > $3/get_swarm.log
#        check get_swarm
        #mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 5 -x ${shm_ctx} > $3/reduction.log
        echo "amo_fadd"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 6 -x ${shm_ctx} > $3/amo_fadd.log
        check amo_fadd
        echo "amo_finc"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 7 -x ${shm_ctx} > $3/amo_finc.log
        check amo_finc
        echo "amo_fetch"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 8 -x ${shm_ctx} > $3/amo_fetch.log
        check amo_fetch
        echo "amo_fcswap"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 9 -x ${shm_ctx} > $3/amo_fcswap.log
        check amo_fcswap
        echo "amo_add"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 10 -x ${shm_ctx} > $3/amo_add.log
        check amo_add
        echo "amo_set"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 44 -x ${shm_ctx} > $3/amo_set.log
        check amo_set
        echo "amo_swap"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 45 -x ${shm_ctx} > $3/amo_swap.log
        check amo_swap
        echo "amo_fetch_and"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 46 -x ${shm_ctx} > $3/amo_fetch_and.log
        check amo_fetch_and
        echo "amo_and"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 49 -x ${shm_ctx} > $3/amo_and.log
        check amo_and
        echo "amo_fetch_or"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 47 -x ${shm_ctx} > $3/amo_fetch_or.log
        check amo_fetch_or
        echo "amo_or"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 50 -x ${shm_ctx} > $3/amo_or.log
        check amo_or
        echo "amo_fetch_xor"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 47 -x ${shm_ctx} > $3/amo_fetch_xor.log
        check amo_fetch_xor
        echo "amo_xor"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 50 -x ${shm_ctx} > $3/amo_xor.log
        check amo_xor
        echo "amo_inc"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 11 -x ${shm_ctx} > $3/amo_inc.log
        check amo_inc
        echo "init"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 13 -x ${shm_ctx} > $3/init.log
        check init
        echo "ping_pong"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 14 -x ${shm_ctx} > $3/ping_pong.log
        check ping_pong
        echo "barrier_all"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 17 -x ${shm_ctx} > $3/barrier_all.log
        check barrier_all
        echo "sync_all"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 18 -x ${shm_ctx} > $3/sync_all.log
        check sync_all
        echo "sync"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 19 -x ${shm_ctx} > $3/sync.log
        check sync
        echo "alltoall"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 23 -x ${shm_ctx} > $3/alltoall.log
        check alltoall
        echo "fcollect"
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 22 -x ${shm_ctx} > $3/fcollect.log
        check fcollect
        ;;
    *"team_ctx_get")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 38 -x ${shm_ctx}
        ;;
    *"team_ctx_get_nbi")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 39 -x ${shm_ctx}
        ;;
    *"team_ctx_put")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 40 -x ${shm_ctx}
        ;;
    *"team_ctx_put_nbi")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 41 -x ${shm_ctx}
        ;;
    *"get")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 0 -x ${shm_ctx}
        ;;
    *"get_nbi")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 1 -x ${shm_ctx}
        ;;
    *"put")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 2 -x ${shm_ctx}
        ;;
    *"put_nbi")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 3 -x ${shm_ctx}
        ;;
    *"get_swarm")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 4 -x ${shm_ctx}
        ;;
    *"team_reduction")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 37 -x ${shm_ctx}
        ;;
    *"reduction")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 5 -x ${shm_ctx}
        ;;
    *"amo_fadd")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 6 -x ${shm_ctx}
        ;;
    *"amo_finc")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 7 -x ${shm_ctx}
        ;;
    *"amo_fetch")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 8 -x ${shm_ctx}
        ;;
    *"amo_fcswap")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 9 -x ${shm_ctx}
        ;;
    *"amo_add")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 10 -x ${shm_ctx}
        ;;
    *"amo_set")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 44 -x ${shm_ctx}
        ;;
    *"amo_swap")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 45 -x ${shm_ctx}
        ;;
    *"amo_fetch_and")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 46 -x ${shm_ctx}
        ;;
    *"amo_and")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 49 -x ${shm_ctx}
        ;;
    *"amo_fetch_or")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 47 -x ${shm_ctx}
        ;;
    *"amo_or")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 50 -x ${shm_ctx}
        ;;
    *"amo_fetch_xor")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 48 -x ${shm_ctx}
        ;;
    *"amo_xor")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 51 -x ${shm_ctx}
        ;;
    *"amo_inc")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 11 -x ${shm_ctx}
        ;;
    *"init")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 13 -x ${shm_ctx}
        ;;
    *"ping_pong")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 14 -x ${shm_ctx}
        ;;
    *"team_broadcast")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 36 -x ${shm_ctx}
        ;;
    *"alltoall")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 23 -x ${shm_ctx}
    ;;
    *"fcollect")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 22 -x ${shm_ctx}
    ;;
    *"broadcast")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512 -a 20 -x ${shm_ctx}
        ;;
    *"barrier_all")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 17 -x ${shm_ctx}
        ;;
    *"sync_all")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 18 -x ${shm_ctx}
        ;;
    *"sync")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 19 -x ${shm_ctx}
        ;;
    *"ctx_infra")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8 -a 42 -x ${shm_ctx}
        ;;
    *)
        echo "UNKNOWN TEST TYPE: $2"
        exit -1
        ;;
esac

exit $?
