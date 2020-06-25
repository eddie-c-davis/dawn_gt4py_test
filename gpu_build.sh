#!/bin/bash
nvcc computation.cu unit_test.cpp -DOPTBACKEND=cuda -x cu -arch=sm_35 -std=c++11 -O3 -g -I/home/eddied/boost_1_67_0 -lgtest -lgtest_main -o run_test
