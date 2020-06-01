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
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__dawncxxopt_5d8517989a_pyext_BUILD/computation.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__dawncxxopt_5d8517989a_pyext_BUILD/unit_test.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__dawncxxopt_19ce2f65ed_pyext_BUILD/computation.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__dawncxxopt_19ce2f65ed_pyext_BUILD/unit_test.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_p_grad_c_vstencil__dawncxxopt_0d41010a82_pyext_BUILD/computation.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_p_grad_c_vstencil__dawncxxopt_0d41010a82_pyext_BUILD/unit_test.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qout_x_edge__dawncxxopt_f336c9ddb2_pyext_BUILD/computation.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qout_x_edge__dawncxxopt_f336c9ddb2_pyext_BUILD/unit_test.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qx_edge_west2__dawncxxopt_8f5b74d0e5_pyext_BUILD/computation.cpp
    ../fv3ser/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_qx_edge_west2__dawncxxopt_8f5b74d0e5_pyext_BUILD/unit_test.cpp
#    ../fv3ser/.gt_cache/py37_1013/dawncuda/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__dawncuda_19ce2f65ed_pyext_BUILD/computation.cu
#    ../fv3ser/.gt_cache/py37_1013/dawncuda/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__dawncuda_19ce2f65ed_pyext_BUILD/unit_test.cpp
)

target_link_libraries(dawn_gt4py_test gtest gtest_main pthread)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dawn_gt4py_test OpenMP::OpenMP_CXX)
endif()
