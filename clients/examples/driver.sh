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

echo Test Name $2

case $2 in
    *"single_thread")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 0 > $3/get.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 1 > $3/get_nbi.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 2 > $3/put.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 3 > $3/put_nbi.log
        mpirun -np 2 $1 -t 1 -w 1 -s 512 -a 5 > $3/reduction.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 6 > $3/amo_fadd.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 7 > $3/amo_finc.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 8 > $3/amo_fetch.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 9 > $3/amo_fcswap.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 10 > $3/amo_add.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 11 > $3/amo_inc.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 12 > $3/amo_cswap.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 13 > $3/init.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 14 > $3/ping_pong.log
        ;;
    *"multi_thread")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 0 > $3/get.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 1 > $3/get_nbi.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 2 > $3/put.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 3 > $3/put_nbi.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 4 > $3/get_swarm.log
        mpirun -np 2 $1 -t 1 -w 1 -s 512 -a 5 > $3/reduction.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 6 > $3/amo_fadd.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 7 > $3/amo_finc.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 8 > $3/amo_fetch.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 9 > $3/amo_fcswap.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 10 > $3/amo_add.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 11 > $3/amo_inc.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 12 > $3/amo_cswap.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 13 > $3/init.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 14 > $3/ping_pong.log
        ;;
    *"ro")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 0 > $3/get.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 1 > $3/get_nbi.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 2 > $3/put.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 3 > $3/put_nbi.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 4 > $3/get_swarm.log
        mpirun -np 2 $1 -t 1 -w 1 -s 512 -a 5 > $3/reduction.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 13 > $3/init.log
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 14 > $3/ping_pong.log
        ;;
    *"get")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 0
        ;;
    *"get_nbi")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 1
        ;;
    *"put")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 2
        ;;
    *"put_nbi")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 3
        ;;
    *"get_swarm")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 4
        ;;
    *"reduction")
        mpirun -np 2 $1 -t 1 -w 1 -s 512 -a 5
        ;;
    *"amo_fadd")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 6
        ;;
    *"amo_finc")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 7
        ;;
    *"amo_fetch")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 8
        ;;
    *"amo_fcswap")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 9
        ;;
    *"amo_add")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 10
        ;;
    *"amo_inc")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 11
        ;;
    *"amo_cswap")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 12
        ;;
    *"init")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 13
        ;;
    *"ping_pong")
        mpirun -np 2 $1 -t 1 -w 1 -s 32768 -a 14
        ;;
    *)
        echo "UNKNOWN TEST TYPE: $2"
        exit -1
        ;;
esac

exit $?
