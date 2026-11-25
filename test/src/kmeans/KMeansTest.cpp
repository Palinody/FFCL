#include <gtest/gtest.h>

#include "ffcl/kmeans/KMeans.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// With two centres and two well separated squares, the converged centroids are the centre of each
// square regardless of the algorithm or the initialisation.
TEST(KMeansTest, ConvergesToTheTwoSquareCentres) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto kmeans = ffcl::KMeans<dType>(2);

    const auto centroids = kmeans.fit(data.begin(), data.end(), n_features);

    ASSERT_EQ(centroids.size(), 4u);  // 2 centroids * 2 features

    // the two centroids are {0.375, 0.375} and {9.375, 9.375} in some order
    auto sorted_centroids = centroids;
    auto as_pairs         = std::vector<std::pair<dType, dType>>{};

    for (std::size_t i = 0; i + 1 < sorted_centroids.size(); i += 2) {
        as_pairs.emplace_back(sorted_centroids[i], sorted_centroids[i + 1]);
    }
    std::sort(as_pairs.begin(), as_pairs.end());

    EXPECT_NEAR(as_pairs[0].first, 0.375f, 1e-4f);
    EXPECT_NEAR(as_pairs[0].second, 0.375f, 1e-4f);
    EXPECT_NEAR(as_pairs[1].first, 9.375f, 1e-4f);
    EXPECT_NEAR(as_pairs[1].second, 9.375f, 1e-4f);
}

TEST(KMeansTest, PredictAssignsEachSquareToOneCluster) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto kmeans = ffcl::KMeans<dType>(2);
    kmeans.fit(data.begin(), data.end(), n_features);

    const auto predictions = kmeans.predict(data.begin(), data.end(), n_features);

    ASSERT_EQ(predictions.size(), 8u);
    EXPECT_EQ(Fixture::count_clusters(predictions, static_cast<std::size_t>(-1)), 2u);

    EXPECT_EQ(predictions[0], predictions[1]);
    EXPECT_EQ(predictions[0], predictions[3]);
    EXPECT_EQ(predictions[4], predictions[5]);
    EXPECT_NE(predictions[0], predictions[4]);
}

// Lloyd and Hamerly must agree on this fixture: it is the property that the fast variant relies on.
TEST(KMeansTest, LloydAndHamerlyAgree) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto lloyd = ffcl::KMeans<dType>(2);
    lloyd.fit<ffcl::Lloyd>(data.begin(), data.end(), n_features);
    const auto lloyd_predictions = lloyd.predict(data.begin(), data.end(), n_features);

    auto hamerly = ffcl::KMeans<dType>(2);
    hamerly.fit<ffcl::Hamerly>(data.begin(), data.end(), n_features);
    const auto hamerly_predictions = hamerly.predict(data.begin(), data.end(), n_features);

    ASSERT_EQ(lloyd_predictions.size(), hamerly_predictions.size());

    // cluster labels are arbitrary, so the two algorithms must agree on the partition rather than on
    // the label values: two samples share a cluster in one result iff they share it in the other
    for (std::size_t i = 0; i < lloyd_predictions.size(); ++i) {
        for (std::size_t j = 0; j < lloyd_predictions.size(); ++j) {
            const auto same_group_lloyd   = lloyd_predictions[i] == lloyd_predictions[j];
            const auto same_group_hamerly = hamerly_predictions[i] == hamerly_predictions[j];

            EXPECT_EQ(same_group_lloyd, same_group_hamerly) << "samples " << i << " and " << j;
        }
    }
}

}  // namespace
