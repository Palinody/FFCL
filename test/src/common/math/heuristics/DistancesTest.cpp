#include <gtest/gtest.h>

#include "ffcl/common/math/heuristics/Distances.hpp"

#include <cmath>

namespace {

const std::vector<float> kLeft  = {0.0f, 0.0f, 0.0f};
const std::vector<float> kRight = {3.0f, 4.0f, 0.0f};

namespace heuristics = ffcl::common::math::heuristics;

TEST(DistancesTest, SquaredEuclideanDistance) {
    const auto distance = heuristics::squared_euclidean_distance(kLeft.begin(), kLeft.end(), kRight.begin(), kRight.end());

    EXPECT_NEAR(distance, 25.0f, 1e-5f);
}

TEST(DistancesTest, EuclideanDistance) {
    const auto distance = heuristics::euclidean_distance(kLeft.begin(), kLeft.end(), kRight.begin(), kRight.end());

    EXPECT_NEAR(distance, 5.0f, 1e-5f);
}

TEST(DistancesTest, ManhattanDistance) {
    const auto distance = heuristics::manhattan_distance(kLeft.begin(), kLeft.end(), kRight.begin(), kRight.end());

    EXPECT_NEAR(distance, 7.0f, 1e-5f);
}

TEST(DistancesTest, DistanceFromOrigin) {
    EXPECT_NEAR(heuristics::euclidean_distance_from_origin(kRight.begin(), kRight.end()), 5.0f, 1e-5f);
    EXPECT_NEAR(heuristics::squared_euclidean_distance_from_origin(kRight.begin(), kRight.end()), 25.0f, 1e-5f);
}

TEST(DistancesTest, DistanceToSelfIsZero) {
    EXPECT_NEAR(heuristics::euclidean_distance(kLeft.begin(), kLeft.end(), kLeft.begin(), kLeft.end()), 0.0f, 1e-6f);
}

TEST(DistancesTest, DistanceIsSymmetric) {
    const auto forward =
        heuristics::euclidean_distance(kLeft.begin(), kLeft.end(), kRight.begin(), kRight.end());
    const auto backward =
        heuristics::euclidean_distance(kRight.begin(), kRight.end(), kLeft.begin(), kLeft.end());

    EXPECT_NEAR(forward, backward, 1e-6f);
}

// Cosine similarity of a vector with itself is 1; with the zero vector it is undefined but must not
// produce a NaN (the implementation guards the zero norm).
TEST(DistancesTest, CosineSimilarityOfOrthogonalVectorsIsZero) {
    const auto x = std::vector<float>{1.0f, 0.0f};
    const auto y = std::vector<float>{0.0f, 1.0f};

    EXPECT_NEAR(heuristics::cosine_similarity(x.begin(), x.end(), y.begin(), y.end()), 0.0f, 1e-5f);
}

}  // namespace
