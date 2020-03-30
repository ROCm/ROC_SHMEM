dnl Process this file with autoconf to produce a configure script.

AC_INIT([ROC_SHMEM], [1.0], [])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([config])
AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_PROG_RANLIB
AC_PATH_PROG(CXX, [hipcc], [/opt/rocm/bin/hipcc])

AC_CHECK_PROGS([DOXYGEN], [doxygen])
if test -z "$DOXYGEN";
   then AC_MSG_WARN([Doxygen not found. Continuing without Doxygen support.])
fi

AM_CONDITIONAL([HAVE_DOXYGEN],
[test -n "$DOXYGEN"])AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/Doxyfile])])

AC_ARG_ENABLE(
    [profile],
    [AC_HELP_STRING([--enable-profile],
                   [Profiler counters (PVARs) and timing support (default=no)])],
    [AC_DEFINE([PROFILE], [1], [enabling profiling])],
    [])

AC_ARG_ENABLE(
    [wf-coal],
    [AC_HELP_STRING([--enable-wf-coal],
                   [enable wavefront-level message coalescing (default=no)])],
    [AC_DEFINE([_WF_COAL_], [1], [enabling wavefront-level message coalescing])],
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

AC_ARG_ENABLE(
    [hdp_map],
    [AC_HELP_STRING([--enable-hdp-map],
                   [enable hdp-map support through hdp_ymap module (default=no)])],
    [AC_DEFINE([_USE_HDP_MAP_], [1], [enabling hdp-map kernel module support])],
    [])

AC_ARG_ENABLE(
    [ro_gpu_heap],
    [AC_HELP_STRING([--enable-ro-gpu-heap],
                   [Symmetric heap on GPU memory (default=no)])],
    [AC_DEFINE([GPU_HEAP], [1], [enabling heap allocation on GPU])],
    [])

AC_ARG_ENABLE(
    [ro_gpu_queue],
    [AC_HELP_STRING([--enable-ro-gpu-queue],
                   [Control Queues on GPU memory (default=no)])],
    [AC_DEFINE([GPU_QUEUE], [1], [enabling control queues allocation on GPU])],
    [])

AC_ARG_WITH([mpi],
    AC_HELP_STRING([--with-mpi], [ Set path to MPI installation ]))
AM_CONDITIONAL(MPI_TRANSPORT, test -d $with_mpi)

AC_ARG_WITH([openshmem],
    AC_HELP_STRING([--with-openshmem], [ Set path to OpenSHMEM installation ]))
AM_CONDITIONAL(OPENSHMEM_TRANSPORT, test -d $with_openshmem)

AC_ARG_ENABLE(
    [gpu_ib_dc],
    [AC_HELP_STRING([--enable-gpu-ib-dc],
                   [enable IB DC support (default=no)])],
    [AC_DEFINE([_USE_DC_], [1], [enabling DC Support])],
    [])

AC_ARG_ENABLE(
    [gpu_ib_cpu_template],
    [AC_HELP_STRING([--enable-gpu-ib-cpu-template],
                   [Rely on CPU to synchronize with GPU and build templates
                    for SQ fields. (default=no)])],
    [],
    [AC_DEFINE([_USE_GPU_UPDATE_SQ_], [1], [enabling GPU UPDATE SQ  Support])])

AC_ARG_ENABLE(
    [gpu_ib_ipc],
    [AC_HELP_STRING([--enable-gpu-ib-ipc],
                   [Enable IPC support through HIP,  (default=no)])],
    [AC_DEFINE([_USE_IPC_], [1], [enabling IPC support ])],
    [])

AC_ARG_ENABLE(
    [gpu_ib_static_queue],
    [AC_HELP_STRING([--enable-static-queue],
                   [Use static queue binding to work-groups instead of
                    recycling (default=no)])],
    [],
    [AC_DEFINE([_RECYCLE_QUEUE_], [1], [enabling Network Queue Recycling])])

AC_ARG_ENABLE(
    [gpu_ib_threads],
    [AC_HELP_STRING([--enable-gpu-ib-threads],
                   [enable multi-threads support (mult-threads of WG sharing the
                    network queues) (default=no)])],
    [AC_DEFINE([_USE_THREADS_], [1], [enabling multi-threads support])],
    [])

AC_ARG_WITH([libibverbs],
    AC_HELP_STRING([--with-libibverbs], [ Set path to libibverbs installation ]))
if test x$with_libibverbs = x || test x$with_libibverbs = xno; then
    want_libibverbs=no
else
    want_libibverbs=yes
    if test -d $with_libibverbs; then
        CPPFLAGS="$CPPFLAGS -I$with_libibverbs/include/infiniband -I/opt/rocm/include/hsa/"
        LDFLAGS="$LDFLAGS -L$with_libibverbs/lib -L$with_libibverbs/lib64 -L$with_libibverbs/lib/libibverbs  -lmlx5 -L/opt/rocm/lib/"
    fi
fi

dnl Checks for Verbs support
AC_CHECK_LIB(ibverbs, ibv_get_device_list, [],
    AC_MSG_ERROR([ibv_get_device_list() not found.  RTN requires libibverbs.]))

AC_CHECK_LIB(ibverbs, ibv_exp_create_qp,
    AC_MSG_ERROR([ibv_exp_create_qp not found. RTN requires verbs extension support.]))

AC_CHECK_LIB(mlx5, mlx5dv_init_obj, [],
    AC_MSG_ERROR([mlx5dv_init_obj not found. RTN requires Direct-Verbs extension support.]))

AC_CHECK_HEADER(infiniband/peer_ops.h, [],
    AC_MSG_ERROR([<infiniband/peer_ops.h> not found.  RTN requires verbs peer-direct support.]))
AC_HEADER_STDC

AS_IF(
    [!(test x$with_mpi = x || test x$with_mpi = xno) && test -d $with_mpi],
        [AC_DEFINE([MPI_TRANSPORT], [1], [enabling MPI transport])]
        using_mpi=1
        LAUNCH_CMD="$with_mpi/bin/mpirun -np 2 -env UCX_TLS=rc -env HSA_FORCE_FINE_GRAIN_PCIE=1"
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
