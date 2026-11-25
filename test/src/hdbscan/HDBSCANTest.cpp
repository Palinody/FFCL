#include <gtest/gtest.h>

#include "ffcl/hdbscan/HDBSCAN.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// HDBSCAN with min_cluster_size 4 and k = 2 must recover the two squares as two clusters.
//
// k_nearest_neighbors counts the sample itself, so k = 2 is the distance to the nearest *other*
// sample; k = 1 would give all-zero core distances and reduce the tree to plain single linkage.
TEST(HDBSCANTest, RecoversTwoWellSeparatedSquares) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto hdbscan = ffcl::HDBSCAN<decltype(indexer)>();
    hdbscan.set_options(
        ffcl::HDBSCAN<decltype(indexer)>::Options().k_nearest_neighbors(2).min_cluster_size(4).allow_single_cluster(
            true));

    const auto predictions = hdbscan.predict(std::move(indexer));

    ASSERT_EQ(predictions.size(), 8u);

    EXPECT_EQ(Fixture::count_clusters(predictions, std::size_t{0}), 2u);

    // both squares are internally consistent and distinct from each other
    EXPECT_EQ(predictions[0], predictions[1]);
    EXPECT_EQ(predictions[0], predictions[3]);
    EXPECT_EQ(predictions[4], predictions[5]);
    EXPECT_NE(predictions[0], predictions[4]);
}

// A min_cluster_size larger than the whole dataset cannot produce two clusters: the condensed tree
// has no split that big, so everything collapses to a single cluster when allowed.
TEST(HDBSCANTest, LargeMinClusterSizeCollapsesToSingleCluster) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto hdbscan = ffcl::HDBSCAN<decltype(indexer)>();
    hdbscan.set_options(ffcl::HDBSCAN<decltype(indexer)>::Options()
                            .k_nearest_neighbors(2)
                            .min_cluster_size(data.size() / n_features + 1)
                            .allow_single_cluster(true));

    const auto predictions = hdbscan.predict(std::move(indexer));

    EXPECT_LE(Fixture::count_clusters(predictions, std::size_t{0}), 1u);
}

// The result must be a label per sample and must not depend on the index order beyond the label
// values themselves.
TEST(HDBSCANTest, ProducesOneLabelPerSampleDeterministically) {
    const auto n_features = std::size_t{2};

    auto first_run = [&] {
        auto& data = Fixture::two_blobs();
        auto        indices = Fixture::make_indices(data.size() / n_features);
        auto        indexer = Fixture::make_kdtree(data, indices, n_features);

        auto hdbscan = ffcl::HDBSCAN<decltype(indexer)>();
        hdbscan.set_options(ffcl::HDBSCAN<decltype(indexer)>::Options()
                                .k_nearest_neighbors(2)
                                .min_cluster_size(4)
                                .allow_single_cluster(true));

        return hdbscan.predict(std::move(indexer));
    };

    const auto labels_a = first_run();
    const auto labels_b = first_run();

    EXPECT_EQ(labels_a.size(), 8u);
    EXPECT_EQ(labels_a, labels_b);
}

}  // namespace
