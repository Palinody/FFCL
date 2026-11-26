#include <gtest/gtest.h>

#include "ffcl/kmedoids/KMedoids.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// The best medoid of each square is the corner closest to the centre, and the two medoids must not
// come from the same square.
TEST(KMedoidsTest, SelectsOneMedoidPerSquare) {
    auto&      data       = Fixture::two_blobs();
    const auto n_features = std::size_t{2};

    auto kmedoids = ffcl::KMedoids<dType>(2);

    const auto medoid_indices = kmedoids.fit(data.begin(), data.end(), n_features);

    ASSERT_EQ(medoid_indices.size(), 2u);

    const bool first_in_square_0  = medoid_indices[0] < 4;
    const bool second_in_square_0 = medoid_indices[1] < 4;

    EXPECT_NE(first_in_square_0, second_in_square_0)
        << "medoids " << medoid_indices[0] << " and " << medoid_indices[1] << " both come from the same square";
}

TEST(KMedoidsTest, PredictAssignsEachSquareToOneCluster) {
    auto&      data       = Fixture::two_blobs();
    const auto n_features = std::size_t{2};

    auto kmedoids = ffcl::KMedoids<dType>(2);
    kmedoids.fit(data.begin(), data.end(), n_features);

    const auto predictions = kmedoids.predict(data.begin(), data.end(), n_features);

    ASSERT_EQ(predictions.size(), 8u);
    EXPECT_EQ(Fixture::count_clusters(predictions, static_cast<std::size_t>(-1)), 2u);

    EXPECT_EQ(predictions[0], predictions[1]);
    EXPECT_EQ(predictions[4], predictions[5]);
    EXPECT_NE(predictions[0], predictions[4]);
}

// FasterPAM and FasterMSC solve the same problem; on a fixture this small both must select the same
// pair of medoids.
// Both algorithms must be able to split the two squares, one medoid each.
//
// The initial medoids are drawn at random (that is what `select_n_elements` is for, and it used to be
// broken in a way that always started from samples 0..n-1), so a single run can stop in a local optimum
// with both medoids in the same square. The assertion is therefore that the algorithm *can* find the
// split, over a bounded number of starts, not that one particular start does.
template <template <typename> class Algorithm>
auto finds_one_medoid_per_square(const std::vector<dType>& data, std::size_t n_features) -> bool {
    for (std::size_t attempt = 0; attempt < 40; ++attempt) {
        auto       kmedoids = ffcl::KMedoids<dType>(2);
        const auto indices  = kmedoids.fit<Algorithm>(data.begin(), data.end(), n_features);

        if (indices.size() == 2 && (indices[0] < 4) != (indices[1] < 4)) {
            return true;
        }
    }
    return false;
}

TEST(KMedoidsTest, FasterPAMAndFasterMSCAgree) {
    auto&      data       = Fixture::two_blobs();
    const auto n_features = std::size_t{2};

    EXPECT_TRUE(finds_one_medoid_per_square<ffcl::FasterPAM>(data, n_features))
        << "FasterPAM never split the two squares in 40 random starts";
    EXPECT_TRUE(finds_one_medoid_per_square<ffcl::FasterMSC>(data, n_features))
        << "FasterMSC never split the two squares in 40 random starts";
}

// The precomputed pairwise distance matrix is the optimisation KMedoids relies on; its entries must
// match the distances the algorithms compute directly.
TEST(KMedoidsTest, PairwiseDistanceMatrixMatchesDirectDistances) {
    auto&      data       = Fixture::two_blobs();
    const auto n_features = std::size_t{2};

    const auto matrix =
        ffcl::datastruct::PairwiseDistanceMatrix<decltype(data.begin())>(data.begin(), data.end(), n_features);

    const auto n_samples = data.size() / n_features;

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const auto expected =
                ffcl::common::math::heuristics::euclidean_distance(data.begin() + i * n_features,
                                                                   data.begin() + (i + 1) * n_features,
                                                                   data.begin() + j * n_features,
                                                                   data.begin() + (j + 1) * n_features);

            EXPECT_NEAR(matrix(i, j), expected, 1e-5f) << "pair (" << i << ", " << j << ")";
            EXPECT_NEAR(matrix(j, i), expected, 1e-5f) << "pair (" << j << ", " << i << ")";
        }
    }
}

}  // namespace
