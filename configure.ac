dnl Process this file with autoconf to produce a configure script.

AC_INIT([RO_NET], [1.0], [])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([config])
AM_INIT_AUTOMAKE([foreign])
AC_PROG_RANLIB
AC_PATH_PROG(CXX, [hipcc], [/opt/rocm/bin/hipcc])

AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN";
   then AC_MSG_WARN([Doxygen not found. Continuing without Doxygen support.])
fi

AM_CONDITIONAL([HAVE_DOXYGEN],
[test -n "$DOXYGEN"])AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])])

AC_ARG_ENABLE(
    [gpu_heap],
    [AC_HELP_STRING([--enable-gpu-heap],
                   [Symmetric heap on GPU memory (default=no)])],
    [AC_DEFINE([GPU_HEAP], [1], [enabling heap allocation on GPU])],
    [])

AC_ARG_ENABLE(
    [gpu_queue],
    [AC_HELP_STRING([--enable-gpu-queue],
                   [Control Queues on GPU memory (default=no)])],
    [AC_DEFINE([GPU_QUEUE], [1], [enabling control queues allocation on GPU])],
    [])

AC_ARG_ENABLE(
    [uc_device_allocator],
    [AC_HELP_STRING([--enable-uc-device-allocator],
                   [Enable UC allocator for GPU memory (default=no)])],
    [AC_DEFINE([UC_DEVICE_ALLOCATOR], [1], [enable uncacheable device allocator])],
    [])

AC_ARG_ENABLE(
    [profile],
    [AC_HELP_STRING([--enable-profile],
                   [Profiler counters (PVARs) and timing support (default=no)])],
    [AC_DEFINE([PROFILE], [1], [enabling profiling])],
    [])

AC_ARG_ENABLE(
    [recycle_queues],
    [AC_HELP_STRING([--enable-recycle-queues],
                   [Limits the number of queues to the max number of concurrently running WGs (default=no)])],
    [AC_DEFINE([RECYCLE_QUEUES], [1], [enabling queue recycling])],
    [])

AC_ARG_ENABLE(
    [debug],
    [AC_HELP_STRING([--enable-debug],
                   [enable debug prints (default=no)])],
    [AC_DEFINE([DEBUG], [1], [enabling debug])],
    [])

AC_ARG_ENABLE(
    [test],
    [AC_HELP_STRING([--enable-test],
                   [Build test programs (default=no)])],
    [enable_test=$enableval],
    [enable_test=no])
AM_CONDITIONAL(TEST_ENABLE, test x$enable_test = xyes)

AC_ARG_WITH([mpi],
    AC_HELP_STRING([--with-mpi], [ Set path to MPI installation ]))
AM_CONDITIONAL(MPI_TRANSPORT, test -d $with_mpi)

AC_ARG_WITH([openshmem],
    AC_HELP_STRING([--with-openshmem], [ Set path to OpenSHMEM installation ]))
AM_CONDITIONAL(OPENSHMEM_TRANSPORT, test -d $with_openshmem)

AS_IF(
    [!(test x$with_mpi = x || test x$with_mpi = xno) && test -d $with_mpi],
        [AC_DEFINE([MPI_TRANSPORT], [1], [enabling MPI transport])]
        using_mpi=1
        LAUNCH_CMD="$with_mpi/bin/mpirun -np 2 -env UCX_TLS=rc"
        AC_SUBST(LAUNCH_CMD)
        CPPFLAGS="$CPPFLAGS -I$with_mpi/include/"
        LDFLAGS="$LDFLAGS -L$with_mpi/lib -lmpi")

AS_IF(
    [!(test x$with_openshmem = x || test x$with_openshmem = xno) && test -d $with_openshmem],
        [AC_DEFINE([OPENSHMEM_TRANSPORT], [1], [enabling OpenSHMEM transport])]
        using_openshmem=1
        LAUNCH_CMD="$with_openshmem/bin/oshrun -np 2"
        AC_SUBST(LAUNCH_CMD)
        CPPFLAGS="$CPPFLAGS -I$with_openshmem/include/"
        LDFLAGS="$LDFLAGS -L$with_openshmem/lib -lsma -lpmi_simple")

AS_IF(
    [test x$using_mpi = x && test x$using_openshmem = x],
    [AC_MSG_ERROR([MPI or OpenSHMEM path required for ro_shmem.  Reconfigure using --with-mpi= or --with-openshmem=])])

AS_IF(
    [test x$using_mpi = x1 && test x$using_openshmem = x1],
    [AC_MSG_ERROR([Cannot enable both MPI and OpenSHMEM])])

AS_IF(
    [test x$using_mpi = x1 && test x$using_openshmem = x1],
    [AC_MSG_ERROR([Cannot enable both MPI and OpenSHMEM])])

AS_IF(
    [test x$GPU_HEAP = x1 && test x$using_openshmem = x1],
    [AC_MSG_ERROR([Cannot use GPU Heap with OpenSHMEM backend])])

dnl Checks for programs
AC_PROG_CXX([hipcc])

dnl Checks for header files.
AC_HEADER_STDC

AC_CONFIG_FILES([Makefile docs/Makefile src/Makefile examples/Makefile])
AC_OUTPUT
