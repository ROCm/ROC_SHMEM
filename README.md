# ROCm OpenSHMEM (ROC_SHMEM)

The ROCm OpenSHMEM (ROC_SHMEM) runtime is part of an AMD Research
initiative to provide GPU-centric networking accessible from within a kernel.
ROC_SHMEM enables fine-grained GPU networking through a simple, OpenSHMEM-like
interface that does not require ending a kernel.  Intra-kernel networking
simplifies application code complexity as well as enabling more fine-grained
communication/computation overlap than traditional host-driven networking.

ROC_SHMEM has two different backend designs to implement intra-kernel
networking.  Both are always built and are selectable via an environment
variable at runtime.

The first design will be referred to as the the GPU InfiniBand (GPU-IB)
backend.  This backend implements a lightweight InfiniBand verbs interface
on the GPU.  The GPU itself is responsible with building commands and ringing
the doorbell on the NIC to send network commands.  GPU-IB is the default and
preferred backend design that offers the best performance.

The second design will be referred to as the Reverse Offload (RO) backend. With
the RO backend, the GPU runtime forwards ROC_SHMEM networking operations to the
host-side runtime, which calls into a traditional MPI or OpenSHMEM
implementation.  This forwarding of requests is transparent to the
programmer, who only sees the GPU-side interface.

## Limitations

ROC_SHMEM is an experimental prototype from AMD Research and not an official
ROCm product.  The software is provided as-is with no guarantees of support
from AMD or AMD Research.

ROC_SHMEM base requirements:
* ROCm version 2.10
    *  May work with other versions, but not tested
* AMD GFX9 GPUs (e.g.: MI25, Vega 56, Vega 64, MI50, MI60, Radeon VII)
* ROCm-aware MPI as described in
  [Building the Dependencies](#building-the-dependencies)
* InfiniBand adaptor compatable with ROCm RDMA technology
* UCX 1.6 or greater with ROCm support

ROC_SHMEM optional requirements
 * For Documentation:
     *  Doxygen

ROC_SHMEM only supports HIP applications. There are no plans to port to
OpenCL.

## Building and Installation

ROC_SHMEM uses the CMake build system. The CMakeLists file contains
additional details about library options.

To create an out-of-source build:

    cd library
    mkdir build
    cd build

Next, choose one configuration from the build_configs subdirectory. These
scripts pass configuration options to CMake to setup canonical builds which
are regularly tested:

    ../build_configs/dc_single
    ../build_configs/dc_multi
    ../build_configs/rc_single
    ../build_configs/rc_multi
    ../build_configs/rc_multi_wf_coal
    ../build_configs/ro_net_basic


## Compiling/linking and Running with ROC_SHMEM

ROC_SHMEM is built as a host and device side library that can be statically
linked to your application during compilation using hipcc.

During the compilation of your application, include the ROC_SHMEM header files
and the ROC_SHMEM library when using hipcc:

    -I/path-to-roc_shmem-installation/include
    -L/path-to-roc_shmem-installation/lib -lroc_shmem_

NOTE: As ROC_SHMEM depends on host MPI support, you need also to link to an
MPI runtime. Since you must use the hipcc compiler, the arguments for MPI
linkage must be added manually as opposed to using mpicc.

## Runtime Parameters

    ROC_SHMEM_RO        (default: not set)
                        Use the RO backend as opposed to the default GPU-IB
                        backend.

    ROC_SHMEM_DEBUG     (default: not set)
                        Enables verbose debug prints from the host if built
                        with --enable-debug.

    ROC_SHMEM_HEAP_SIZE (default : 1 GB)
                        Defines the size of the OpenSHMEM symmetric heap
                        Note the heap is on the GPU memory.

    ROC_SHMEM_SQ_SIZE   (default 1024)
                        Defines the size of the SQ as number of network
                        packet (WQE). Each WQE is 64B. This only for
                        GPU-IB conduit

    RTN_USE_CQ_GPU_MEM  (default : 1)
                        Set the placement of CQ on GPU memory (1)
                        or CPU memory (0)

    RTN_USE_SQ_GPU_MEM  (default : 1)
                        Set the placement of SQ on GPU memory (1)
                        or CPU memory (0)

    RO_NET_QUEUE_SIZE   (default: 64 elements)
                        Defines the size of the producer/consumer queue per
                        work-group (each element 128B). RO backend only.

    RO_NET_CPU_HEAP     (default: not set)
                        Force symmetric heap to be in CPU memory.  RO backend
                        only.

    RO_NET_CPU_QUEUE    (default: not set)
                        Force producer/consumer queues between CPU and GPU to
                        be in CPU memory. RO backend only.

ROC_SHMEM also requires the following environment variable be set for ROCm:

    export HSA_FORCE_FINE_GRAIN_PCIE=1

For GPU-side debug printf support, in addition to setting --enable-debug at
build time, the following environment variable must be set:

    export HCC_ENABLE_PRINTF=1

## Examples

ROC_SHMEM is similar to OpenSHMEM and should be familiar to programmers who
have experience with OpenSHMEM or other PGAS network programming APIs in the
context of CPUs. The best way to learn how to use ROC_SHMEM is to read the
autogenerated doxygen documentation for functions described in
include/roc_shmem.hpp, or to look at the provided sample applications in the
examples/ folder. ROC_SHMEM is shipped with a basic test suite for the
supported ROC_SHMEM API. The example folder tests Puts, Gets, nonblocking Puts,
nonblocking Gets, Quiets, and Reductions.

All example programs can be run using:

    make check

By default, mpirun will be used as the launcher for tests. However, this can
be overridden using the LAUNCH_CMD runtime environment variable:

e.g., `export LAUNCH_CMD=/usr/bin/mpirun -np 2 -env UCX_TLS=rc`

## Building the Dependencies

ROC_SHMEM requires an MPI runtime on the host that supports ROCm-Aware MPI.
Currently all ROCm-Aware MPI runtimes require the usage of ROCm-Aware UCX.

To build and configure ROCm-Aware UCX, you need to:
 1. Download the latest UCX
 2. Configure and build UCX with ROCm support: --with-rocm=/opt/rocm

Then, you need to build your MPI (OpenMPI or MPICH CH4) with UCX support.

For more information on OpenMPI-UCX support, please visit:
https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX

For more information on MPICH-UCX support, please visit:
https://www.mpich.org/about/news/
