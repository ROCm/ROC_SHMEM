# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

shm_ctx=8  # run tests with SHMEM_CONTEXT_WG_PRIVATE
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
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 0  -x ${shm_ctx} > $3/get.log
        check get
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 1  -x ${shm_ctx} > $3/get_nbi.log
        check get_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 2  -x ${shm_ctx} > $3/put.log
        check put
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 3  -x ${shm_ctx} > $3/put_nbi.log
        check put_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8     -a 42 -x ${shm_ctx} > $3/team_ctx_infra.log
        check team_ctx_infra
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 41  -x ${shm_ctx} > $3/team_ctx_put_nbi.log
        check team_ctx_put_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 39  -x ${shm_ctx} > $3/team_ctx_get_nbi.log
        check team_ctx_get_nbi
        #mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512   -a 5  -x ${shm_ctx} > $3/reduction.log
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 6  -x ${shm_ctx} > $3/amo_fadd.log
        check amo_fadd
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 7  -x ${shm_ctx} > $3/amo_finc.log
        check amo_finc
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 8  -x ${shm_ctx} > $3/amo_fetch.log
        check amo_fetch
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 9  -x ${shm_ctx} > $3/amo_fcswap.log
        check amo_fcswap
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 10 -x ${shm_ctx} > $3/amo_add.log
        check amo_add
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 11 -x ${shm_ctx} > $3/amo_inc.log
        check amo_inc
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 13 -x ${shm_ctx} > $3/init.log
        check init
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 14 -x ${shm_ctx} > $3/ping_pong.log
        check ping_pong
        ;;
    *"multi_thread")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 0  -x ${shm_ctx} > $3/get.log
        check get
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 1  -x ${shm_ctx} > $3/get_nbi.log
        check get_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 2  -x ${shm_ctx} > $3/put.log
        check put
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 3  -x ${shm_ctx} > $3/put_nbi.log
        check put_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8     -a 42 -x ${shm_ctx} > $3/team_ctx_infra.log
        check team_ctx_infra
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 41  -x ${shm_ctx} > $3/team_ctx_put_nbi.log
        check team_ctx_put_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 39  -x ${shm_ctx} > $3/team_ctx_get_nbi.log
        check team_ctx_get_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 4  -x ${shm_ctx} > $3/get_swarm.log
        check get_swarm
        #mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512   -a 5  -x ${shm_ctx} > $3/reduction.log
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 6  -x ${shm_ctx} > $3/amo_fadd.log
        check amo_fadd
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 7  -x ${shm_ctx} > $3/amo_finc.log
        check amo_finc
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 8  -x ${shm_ctx} > $3/amo_fetch.log
        check amo_fetch
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 9  -x ${shm_ctx} > $3/amo_fcswap.log
        check amo_fcswap
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 10 -x ${shm_ctx} > $3/amo_add.log
        check amo_add
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 11 -x ${shm_ctx} > $3/amo_inc.log
        check amo_inc
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 13 -x ${shm_ctx} > $3/init.log
        check init
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 14 -x ${shm_ctx} > $3/ping_pong.log
        check ping_pong
        ;;
    *"ro")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 0  -x ${shm_ctx} > $3/get.log
        check get
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 1  -x ${shm_ctx} > $3/get_nbi.log
        check get_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 2  -x ${shm_ctx} > $3/put.log
        check put
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 3  -x ${shm_ctx} > $3/put_nbi.log
        check put_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8     -a 42 -x ${shm_ctx} > $3/team_ctx_infra.log
        check team_ctx_infra
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 41  -x ${shm_ctx} > $3/team_ctx_put_nbi.log
        check team_ctx_put_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 39  -x ${shm_ctx} > $3/team_ctx_get_nbi.log
        check team_ctx_get_nbi
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 4  -x ${shm_ctx} > $3/get_swarm.log
        check get_swarm
        #mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512   -a 5  -x ${shm_ctx} > $3/reduction.log
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 13 -x ${shm_ctx} > $3/init.log
        check init
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 32768 -a 14 -x ${shm_ctx} > $3/ping_pong.log
        check ping_pong
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
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512   -a 37 -x ${shm_ctx}
        ;;
    *"reduction")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512   -a 5 -x ${shm_ctx}
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
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512   -a 36 -x ${shm_ctx}
        ;;
    *"broadcast")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 512   -a 20 -x ${shm_ctx}
        ;;
    *"barrier_all")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8     -a 17 -x ${shm_ctx}
        ;;
    *"ctx_infra")
        mpirun -np 2 ${gdb_cmd} $1 -t 1 -w 1 -s 8     -a 42 -x ${shm_ctx}
        ;;
    *)
        echo "UNKNOWN TEST TYPE: $2"
        exit -1
        ;;
esac

exit $?
