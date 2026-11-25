#include <gtest/gtest.h>

#include "ffcl/datastruct/matrix/PairwiseDistanceMatrix.hpp"

namespace {

using dType = float;

// 3 samples, 2 features, in a right angle at sample 0.
const std::vector<dType> kData = {
    0.0f, 0.0f,  // sample 0
    3.0f, 0.0f,  // sample 1  -> d(0,1) = 3
    0.0f, 4.0f,  // sample 2  -> d(0,2) = 4, d(1,2) = 5
};

TEST(PairwiseDistanceMatrixTest, EntriesMatchTheEuclideanDistance) {
    const auto matrix =
        ffcl::datastruct::PairwiseDistanceMatrix<decltype(kData.begin())>(kData.begin(), kData.end(), 2);

    EXPECT_NEAR(matrix(0, 1), 3.0f, 1e-5f);
    EXPECT_NEAR(matrix(0, 2), 4.0f, 1e-5f);
    EXPECT_NEAR(matrix(1, 2), 5.0f, 1e-5f);
}

TEST(PairwiseDistanceMatrixTest, MatrixIsSymmetric) {
    const auto matrix =
        ffcl::datastruct::PairwiseDistanceMatrix<decltype(kData.begin())>(kData.begin(), kData.end(), 2);

    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            EXPECT_NEAR(matrix(row, column), matrix(column, row), 1e-6f);
        }
    }
}

TEST(PairwiseDistanceMatrixTest, DistanceToSelfIsZero) {
    const auto matrix =
        ffcl::datastruct::PairwiseDistanceMatrix<decltype(kData.begin())>(kData.begin(), kData.end(), 2);

    for (std::size_t index = 0; index < 3; ++index) {
        EXPECT_NEAR(matrix(index, index), 0.0f, 1e-6f);
    }
}

}  // namespace
