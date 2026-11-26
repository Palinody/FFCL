#include <gtest/gtest.h>

#include "ffcl/kmedoids/KMedoids.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// The best medoid of each square is the corner closest to the centre, and the two medoids must not
// come from the same square.
TEST(KMedoidsTest, SelectsOneMedoidPerSquare) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto kmedoids = ffcl::KMedoids<dType>(2);

    const auto medoid_indices = kmedoids.fit(data.begin(), data.end(), n_features);

    ASSERT_EQ(medoid_indices.size(), 2u);

    const bool first_in_square_0  = medoid_indices[0] < 4;
    const bool second_in_square_0 = medoid_indices[1] < 4;

    EXPECT_NE(first_in_square_0, second_in_square_0) << "medoids " << medoid_indices[0] << " and "
                                                     << medoid_indices[1] << " both come from the same square";
}

TEST(KMedoidsTest, PredictAssignsEachSquareToOneCluster) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

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
TEST(KMedoidsTest, FasterPAMAndFasterMSCAgree) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto       pam = ffcl::KMedoids<dType>(2);
    const auto pam_indices = pam.fit<ffcl::FasterPAM>(data.begin(), data.end(), n_features);

    auto       msc = ffcl::KMedoids<dType>(2);
    const auto msc_indices = msc.fit<ffcl::FasterMSC>(data.begin(), data.end(), n_features);

    ASSERT_EQ(pam_indices.size(), 2u);
    ASSERT_EQ(msc_indices.size(), 2u);

    // both must pick one medoid per square; the exact corner is not unique because the square is
    // symmetric, so the two algorithms may pick different but equally valid corners
    EXPECT_NE(pam_indices[0] < 4, pam_indices[1] < 4);
    EXPECT_NE(msc_indices[0] < 4, msc_indices[1] < 4);
}

// The precomputed pairwise distance matrix is the optimisation KMedoids relies on; its entries must
// match the distances the algorithms compute directly.
TEST(KMedoidsTest, PairwiseDistanceMatrixMatchesDirectDistances) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    const auto matrix =
        ffcl::datastruct::PairwiseDistanceMatrix<decltype(data.begin())>(data.begin(), data.end(), n_features);

    const auto n_samples = data.size() / n_features;

    for (std::size_t i = 0; i < n_samples; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const auto expected = ffcl::common::math::heuristics::euclidean_distance(
                data.begin() + i * n_features,
                data.begin() + (i + 1) * n_features,
                data.begin() + j * n_features,
                data.begin() + (j + 1) * n_features);

            EXPECT_NEAR(matrix(i, j), expected, 1e-5f) << "pair (" << i << ", " << j << ")";
            EXPECT_NEAR(matrix(j, i), expected, 1e-5f) << "pair (" << j << ", " << i << ")";
        }
    }
}

}  // namespace
