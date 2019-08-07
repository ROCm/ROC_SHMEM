# Reverse Offload Networking (RO_NET)

The Reverse Offload Networking (RO_NET) runtime is part of an AMD Research
initiative to provide GPU-centric networking accessible from within a kernel.
RO_NET enables fine-grained GPU networking through a simple, OpenSHMEM-like
interface that does not require ending a kernel.  Intra-kernel networking
simplifies application code complexity as well as enabling more fine-grained
communication/computation overlap than traditional host-driven networking.

The core of RO_NET's design consists of device- and host-side runtimes that
are linked to the application.  The GPU runtime forwards RO_NET networking
operations to the host-side runtime, which calls into a traditional MPI or
OpenSHMEM implementation.  This forwarding of requests is transparent to the
programmer, who only sees the GPU-side interface.

## Limitations

RO_NET is an experimental prototype from AMD Research and not an official ROCm
product.  The software is provided as-is with no guarantees of support from AMD
or AMD Research.

RO_NET base requirements:
* ROCm version 2.6 or greater
* AMD GFX9 GPUs (e.g.: MI25, Vega 56, Vega 64, MI50, MI60, Radeon VII)
* Either MPI or OpenSHMEM host install, as described in
  [Building the Dependencies](#building-the-dependencies)

RO_NET optional requirements
 * For Documentation:
     *  Doxygen

 * For GPU memory heap and/or queues:
     *  InfiniBand adaptor compatable with ROCm RDMA technology
     *  UCX 1.6 or greater with ROCm support

Currently, RO_NET includes support for a small subset of OpenSHMEM 1.4
functionality (i.e., Put/Get, Quiet/Fence, Barrier_All, and a Reduction).

RO_NET only supports HIP applications.  There are no plans to port to OpenCL.

## Configuration

RO_NET uses autotools as its build system.  To generate the configure file,
first run:

    ./autogen.sh

Then, to configure the build environment:

    ./configure

RO_NET supports multiple configuration options to tune features and optimize
your installation:

    --enable-gpu-heap
        This option enables the allocation of the OpenSHMEM symmetric heap on
        the GPU memory. By default the heap is allocated on CPU memory.

        Note: Not supported in combination with --with-openshmem.

    --enable-gpu-queue
        This options allocates the RO_NET producer/consumers queues on GPU
        memory. By default the queues are on CPU memory.

    --enable-recycle-queues
        This options restricts the number of queues to equal the maximum number
        of work-groups that can be scheduled on the hardware at once.  This can
        improve memory utalization and performance in kernels that overprovision
        the hardware. By default the queues are allocated one per
        total work-group.

    --enable-profile
        This option builds the RO_NET runtime with profiling support. This
        supports includes API and mechanisms to accurately measure time on the
        GPUs as well as support for performance counters for CPU- and GPU-side
        RO_NET statistics.

    --enable-debug
        This option enables the support of debug printfs. By default debug
        support is disabled.

    --with-mpi
        Provide the path to the MPI installation on your system and configure
        RO_NET to use MPI as the networking backend.

    --with-openshmem
        Provide the path to the OpenSHMEM installation on your system and
        configure RO_NET to use OpenSHMEM as the networking backend.

        Note: The OpenSHMEM backend is experimental.  MPI is strongly
        recommended at this time.

## Building and Installation

After a succesfull configuration:

    make && make install

## Compiling/linking and Running with RO_NET

RO_NET is built as a host and device side library that can be statically linked
to your application during its compilation using hipcc.

Duing the compilation of your application, include the RO_NET header files
and the RO_NET library when using hipcc:

    -I/path-to-ro_net-installation/include
    -L/path-to-ro_net-installation/lib -lro_net

NOTE: As RO_NET depends on host MPI or OpenSHMEM support, you need also to link
to an MPI or OpenSHMEM runtime.  Since you must use the hipcc compiler, the
arguments for MPI/OpenSHMEM linkage must be added manually as opposed to using
mpicc/oshcc.

## Runtime Parameters

RO_NET uses some runtime parmeters to tune the size of the queues used for
CPU-GPU interaction and enable/disable debug prints:

    RO_NET_QUEUE_SIZE   (default: 64 elements)
                        Defines the size of the producer/consumer queue per
                        work-group (each element 128B).

    RO_NET_DEBUG        (not set)
                        Enables verbose debug prints if built with
                        --enable-debug

RO_NET also requires the following environment variable be set for ROCm if
using GPU memory for queues or the symmetric heap:

    export HSA_FORCE_FINE_GRAIN_PCIE=1

## Examples

RO_NET is similar to OpenSHMEM and should be familiar to programmers who have
experience with OpenSHMEM or other PGAS network programming APIs in the context
of CPUs.  The best way to learn how to use RO_NET is to read the autogenerated
doxygen documentation for functions described in include/ro_net.hpp, or to
look at the provided sample applications in the examples/ folder.
RO_NET is shipped with a basic test suite for the supported RO_NET API.
The example folder tests Puts, Gets, nonblocking Puts, nonblocking Gets,
Quiets, and Reductions.

All example programs can be run using:

    make check

By default, either mpirun or oshrun will be used as the launcher, depending
on how RO_NET was configured.  However, this can be overridden using the
LAUNCH_CMD runtime environment variable:

e.g., `export LAUNCH_CMD=/usr/bin/mpirun -np 2 -env UCX_TLS=rc`

## Building the Dependencies

RO_NET requires an OpenSHMEM or MPI runtime on the host.

Furthermore, if RO_NET is configured with `--enable-gpu-heap`, ROCm-Aware MPI
runtime support is required. Currently all ROCm-Aware MPI runtimes require the
usage of ROCm-Aware UCX.

To build and configure ROCm-Aware UCX, you need to:
 1. Download the latest UCX
 2. Configure and build UCX with ROCm support: --with-rocm=/opt/rocm

Then, you need to build your MPI (OpenMPI or MPICH CH4) with UCX support.

For more information on OpenMPI-UCX support, please visit:
https://github.com/openucx/ucx/wiki/OpenMPI-and-OpenSHMEM-installation-with-UCX

For more information on MPICH-UCX support, please visit:
https://www.mpich.org/about/news/
