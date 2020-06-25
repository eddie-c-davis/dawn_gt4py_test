cmake_minimum_required(VERSION 3.13)
project(dawn_gt4py_test)

set(CMAKE_CXX_STANDARD 11)
include(GoogleTest)
enable_testing()

#include_directories(. ./include)
include_directories(. ../gridtools/include ../dawn/dawn/src ../dawn/install/include ../clang-gridtools ../clang-gridtools/build)

add_executable(
    dawn_gt4py_test
    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__cxxopt_7ada0bade9_pyext_BUILD/computation.cpp
    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__cxxopt_7ada0bade9_pyext_BUILD/unit_test.cpp
    /mnt/ram/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__dawncxxopt_7ada0bade9_pyext_BUILD/computation.cpp
    /mnt/ram/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_update_kinetic_energy__dawncxxopt_7ada0bade9_pyext_BUILD/unit_test.cpp
#    /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__cxxopt_3523155bef_pyext_BUILD/computation.cpp
#    /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_ustencil__cxxopt_3523155bef_pyext_BUILD/unit_test.cpp
#    /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_vstencil__cxxopt_26e0af8086_pyext_BUILD/computation.cpp
#    /mnt/ram/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_p_grad_c_vstencil__cxxopt_26e0af8086_pyext_BUILD/unit_test.cpp
#    /mnt/ram/.gt_cache/py37_1013/cxxopt/__main__/m_double_lap_mc__cxxopt_861c6224be_pyext_BUILD/computation.cpp
#    /mnt/ram/.gt_cache/py37_1013/cxxopt/__main__/m_double_lap_mc__cxxopt_861c6224be_pyext_BUILD/unit_test.cpp
#    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_set_vals_1__cxxopt_2de0648616_pyext_BUILD/computation.cpp
#    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_set_vals_1__cxxopt_2de0648616_pyext_BUILD/unit_test.cpp
#    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_set_extm__cxxopt_4eee958721_pyext_BUILD/computation.cpp
#    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_set_extm__cxxopt_4eee958721_pyext_BUILD/unit_test.cpp
    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_update_dz_c__cxxopt_f2cdfd7661_pyext_BUILD/computation.cpp
    /mnt/ram/.gt_cache/py37_1013/cxxopt/fv3/utils/gt4py_utils/m_update_dz_c__cxxopt_f2cdfd7661_pyext_BUILD/unit_test.cpp
    /mnt/ram/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_update_dz_c__dawncxxopt_f2cdfd7661_pyext_BUILD/computation.cpp
    /mnt/ram/.gt_cache/py37_1013/dawncxxopt/fv3/utils/gt4py_utils/m_update_dz_c__dawncxxopt_f2cdfd7661_pyext_BUILD/unit_test.cpp
    /home/eddied/Work/clang-gridtools/benchmarks/utils/benchmark_writer.cpp
    /home/eddied/Work/clang-gridtools/build/benchmarks/generated/fast_waves_sc_wpptp_c++-naive.cpp
    /home/eddied/Work/clang-gridtools/build/benchmarks/generated/fast_waves_sc_wpptp_gt.cpp
    /home/eddied/Work/clang-gridtools/benchmarks/fast_waves_sc_wpptp_benchmark.cpp
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
