#!/bin/bash
g++ computation.cpp unit_test.cpp -std=c++11 -O3 -g -I/home/eddied/boost_1_67_0 -lpthread -lgtest -lgtest_main -o run_test

