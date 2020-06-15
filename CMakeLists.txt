cmake_minimum_required(VERSION 3.13)
project(dawn_gt4py_test)

set(CMAKE_CXX_STANDARD 11)
include(GoogleTest)
enable_testing()

#include_directories(. ./include)
include_directories(. ../gridtools/include ../dawn/dawn/src)

add_executable(
    dawn_gt4py_test
    main.cpp
      /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__cxxopt_5a4b3474ff_pyext_BUILD/computation.cpp
      /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__cxxopt_5a4b3474ff_pyext_BUILD/unit_test.cpp
      /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__cxxopt_3523155bef_pyext_BUILD/computation.cpp
      /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__cxxopt_3523155bef_pyext_BUILD/unit_test.cpp
      /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_vstencil__cxxopt_26e0af8086_pyext_BUILD/computation.cpp
      /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_vstencil__cxxopt_26e0af8086_pyext_BUILD/unit_test.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qout_x_edge__dawncxxopt_f336c9ddb2_pyext_BUILD/computation.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qout_x_edge__dawncxxopt_f336c9ddb2_pyext_BUILD/unit_test.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qx_edge_west2__dawncxxopt_8f5b74d0e5_pyext_BUILD/computation.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qx_edge_west2__dawncxxopt_8f5b74d0e5_pyext_BUILD/unit_test.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_edge_profile__dawncxxopt_5074107e98_pyext_BUILD/computation.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_edge_profile__dawncxxopt_5074107e98_pyext_BUILD/unit_test.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncuda/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__dawncuda_19ce2f65ed_pyext_BUILD/computation.cu
#    ../fv3ser/.gt_cache/py37_1013/dawncuda/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__dawncuda_19ce2f65ed_pyext_BUILD/unit_test.cpp
)

target_link_libraries(dawn_gt4py_test gtest gtest_main pthread)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dawn_gt4py_test OpenMP::OpenMP_CXX)
endif()
