#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "ffcl/common/Utils.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

class UtilsErrorsTest : public ::testing::Test {
  public:
  protected:
};

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}