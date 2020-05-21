#include <gtest/gtest.h>

#ifndef _GTEST_MAIN_
int main(int argc, char* argv[]) {
  // Initialize GTest
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
} // main
#define _GTEST_MAIN_
#endif
