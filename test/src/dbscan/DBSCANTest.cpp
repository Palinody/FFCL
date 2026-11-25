#include <gtest/gtest.h>

#include "ffcl/dbscan/DBSCAN.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// With radius 2 only the four points of a square are mutual neighbours: the distance between the
// corners of one square is sqrt(2) < 2 and the distance between the two squares is 8 > 2.
TEST(DBSCANTest, SeparatesTwoWellSeparatedSquares) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(2.0f).min_samples(3));

    const auto predictions = dbscan.predict(std::move(indexer));

    ASSERT_EQ(predictions.size(), 8u);

    // exactly two clusters, no noise
    EXPECT_EQ(Fixture::count_clusters(predictions, std::size_t{0}), 2u);
    EXPECT_EQ(std::count(predictions.begin(), predictions.end(), std::size_t{0}), 0);

    // all four points of a square share a label, and the two squares do not
    EXPECT_EQ(predictions[0], predictions[1]);
    EXPECT_EQ(predictions[0], predictions[2]);
    EXPECT_EQ(predictions[0], predictions[3]);

    EXPECT_EQ(predictions[4], predictions[5]);
    EXPECT_EQ(predictions[4], predictions[6]);
    EXPECT_EQ(predictions[4], predictions[7]);

    EXPECT_NE(predictions[0], predictions[4]);
}

// The isolated points are further than `radius` from everything else, so they cannot become core
// points and must be labelled noise (label 0).
TEST(DBSCANTest, IsolatedPointsAreNoise) {
    auto& data             = Fixture::two_blobs_with_outliers();
    const auto  n_features = std::size_t{2};

    constexpr std::size_t kLeftOutlier  = 8;
    constexpr std::size_t kRightOutlier = 9;

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(2.0f).min_samples(3));

    const auto predictions = dbscan.predict(std::move(indexer));

    ASSERT_EQ(predictions.size(), 10u);

    EXPECT_EQ(predictions[kLeftOutlier], 0u);
    EXPECT_EQ(predictions[kRightOutlier], 0u);

    // the two squares are still recovered
    EXPECT_EQ(predictions[0], predictions[1]);
    EXPECT_NE(predictions[0], predictions[4]);
    EXPECT_NE(predictions[0], 0u);
    EXPECT_NE(predictions[4], 0u);
}

// A radius large enough to bridge the two squares makes the whole fixture a single cluster.
TEST(DBSCANTest, LargeRadiusMergesEverything) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(20.0f).min_samples(2));

    const auto predictions = dbscan.predict(std::move(indexer));

    EXPECT_EQ(Fixture::count_clusters(predictions, std::size_t{0}), 1u);
}

// Every sample is its own label source: a radius with no neighbours at all must leave everything
// labelled noise.
TEST(DBSCANTest, TinyRadiusLabelsEverythingAsNoise) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(0.01f).min_samples(2));

    const auto predictions = dbscan.predict(std::move(indexer));

    EXPECT_EQ(Fixture::count_clusters(predictions, std::size_t{0}), 0u);
    EXPECT_EQ(std::count(predictions.begin(), predictions.end(), std::size_t{0}), predictions.size());
}

}  // namespace
