#!/bin/bash
###############################################
# Script : shmem_allLib_build_and_test.sh
# Description : the script is to build and execute test for 
#               given libraries as parameter to the script.
#               the script used in math_ci jenkins pipeline
# version 1.0 (Version 1)
# Assumption : the script is placed under clients/functional_tests
###############################################
echo ""
echo ""
echo "Starting script .."$0
echo ""
#check parameters. parameter of libs that need to be built and tested will be passed
if [ $# -ne 1 ]
then
  echo "Invalid Options for $0 ";
  echo "Usage : $0 <liblist> "
  echo "example : $0 RC_SINGLE,RC_MULTI "
  exit 1
fi

liblist=$1
tliblist=`echo $liblist | sed 's/,/ /g'` # replace , with space, so that can be passed for loop
echo "Parameters passed ==> $tliblist"
echo

for libi in $tliblist
do
  echo
  echo "==> Processing Lib...$libi"
  case $libi in
   RC_SINGLE)
      libnm=rc_single
      libBuildDir=$libnm"_build"
      ft_releaseDir=$libnm"_build"
      threadType="single_thread"
      ;;
   RC_MULTI)
      libnm=rc_multi
      libBuildDir=$libnm"_build"
      ft_releaseDir=$libnm"_build"
      threadType="multi_thread"
      ;;
   RC_MULTI_WF_COAL)
      libnm=rc_multi_wf_coal
      libBuildDir=$libnm"_build"
      ft_releaseDir=$libnm"_build"
      threadType="multi_thread"
      ;;
   DC_SINGLE)
      libnm=dc_single
      libBuildDir=$libnm"_build"
      ft_releaseDir=$libnm"_build"
      threadType="single_thread"
      ;;
   DC_MULTI)
      libnm=dc_multi
      libBuildDir=$libnm"_build"
      ft_releaseDir=$libnm"_build"
      threadType="multi_thread"
      ;;
   RO_NET)
      libnm=ro_net
      libBuildDir=$libnm"_build"
      ft_releaseDir=$libnm"_build"
      threadType="ro"
      ;;
   *)
     echo "Unable to find the option for $libi. Plesae check shmem admin"
     echo "** warning ** Skipping the process for $libi "
     ;;
  esac

  if [ "$libnm" != "" ] #process only if libname found
  then 
      echo "+-------------------------------------------------------------------------------------------------+"
      echo `date +%Y%m%d%H%M%S`" ==> Start | $threadType - build_configs/$libnm <=="
      echo "starting with params==> $libnm ; $libBuildDir ; $ft_releaseDir ; $threadType"
      echo
    
      echo "Library build at ==>library/"$libBuildDir

      mkdir library/$libBuildDir
      cd library/$libBuildDir
      ../build_configs/$libnm # from build directory generating the build

      cd ../../ # move to base directory

      echo "functional test release build at ==> clients/functional_tests/"$ft_releaseDir

      mkdir clients/functional_tests/$ft_releaseDir
      cd clients/functional_tests/$ft_releaseDir
      ../build_configs/release # from functional_tests release diretory executing the release build

      # test exeuction based on lib
      if [ "$libnm" == "ro_net" ]
      then
        cd ../  #from functional_tests executing test script
        ROC_SHMEM_RO=1
        ROC_NET_CPU_QUEUE=1
        UCX_TLS=rc
        #echo $ROC_SHMEM_RO"--"$ROC_NET_CPU_QUEUE "--"$UCX_TLS
        ./driver.sh $ft_releaseDir/rocshmem_example_driver ro $ft_releaseDir
      else
        cd ../  #from functional_tests executing test script
        ./driver.sh $ft_releaseDir/rocshmem_example_driver $threadType $ft_releaseDir
      fi

      if [ $? -ne 0 ]
      then
        echo "Lib $libnm functional Test exited with ** error $? **"
      else
        echo "Lib $libnm sucessfull "
      fi

      cd ../../ # move to base directory

      echo
      echo `date +%Y%m%d%H%M%S`" ==> End | $threadType - build_configs/$libnm <=="
      echo "+-------------------------------------------------------------------------------------------------+"
  fi

  libnm="" #reset the parameter after completion of build
  libBuildDir=""
  ft_releaseDir=""
  threadType=""
done
echo 
echo "Script execution ==> $0 <== is done"
echo 

