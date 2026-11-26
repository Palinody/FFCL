#include <gtest/gtest.h>

#include "ffcl/dbscan/DBSCAN.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// With radius 2 only the four points of a square are mutual neighbours: the distance between the
// corners of one square is sqrt(2) < 2 and the distance between the two squares is 8 > 2.
TEST(DBSCANTest, SeparatesTwoWellSeparatedSquares) {
    auto&      data       = Fixture::two_blobs();
    const auto n_features = std::size_t{2};

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
    auto&      data       = Fixture::two_blobs_with_outliers();
    const auto n_features = std::size_t{2};

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
    auto&      data       = Fixture::two_blobs();
    const auto n_features = std::size_t{2};

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
    auto&      data       = Fixture::two_blobs();
    const auto n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(0.01f).min_samples(2));

    const auto predictions = dbscan.predict(std::move(indexer));

    EXPECT_EQ(Fixture::count_clusters(predictions, std::size_t{0}), 0u);
    EXPECT_EQ(std::count(predictions.begin(), predictions.end(), std::size_t{0}), predictions.size());
}

// A reference DBSCAN over every pair, used to check the tree-accelerated one.
//
// The accelerated query prunes whole subtrees: a sibling is skipped as soon as its bounding box
// misses the radius. An error there does not crash, it silently loses neighbours, which changes the
// clusters. So the labels are compared against an implementation that cannot prune anything.
auto reference_dbscan(const std::vector<dType>& data, std::size_t n_features, dType radius, std::size_t min_samples)
    -> std::vector<std::size_t> {
    const std::size_t n_samples = data.size() / n_features;

    auto neighbours_of = [&](std::size_t sample_index) {
        auto neighbours = std::vector<std::size_t>{};
        for (std::size_t other_index = 0; other_index < n_samples; ++other_index) {
            auto squared_distance = dType{0};
            for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
                const auto difference =
                    data[sample_index * n_features + feature_index] - data[other_index * n_features + feature_index];
                squared_distance += difference * difference;
            }
            if (squared_distance < radius * radius) {
                neighbours.push_back(other_index);
            }
        }
        return neighbours;
    };

    auto        visited       = std::vector<bool>(n_samples, false);
    auto        predictions   = std::vector<std::size_t>(n_samples, 0);
    std::size_t cluster_label = 0;

    for (std::size_t candidate = 0; candidate < n_samples; ++candidate) {
        if (visited[candidate]) {
            continue;
        }
        visited[candidate] = true;

        auto neighbours = neighbours_of(candidate);

        // same comparison as the implementation under test
        if (neighbours.size() > min_samples) {
            predictions[candidate] = ++cluster_label;

            for (std::size_t k = 0; k < neighbours.size(); ++k) {
                const auto neighbour = neighbours[k];
                if (!visited[neighbour]) {
                    visited[neighbour] = true;
                    auto inner         = neighbours_of(neighbour);
                    if (inner.size() > min_samples) {
                        neighbours.insert(neighbours.end(), inner.begin(), inner.end());
                    }
                    predictions[neighbour] = cluster_label;
                } else if (predictions[neighbour] == 0) {
                    predictions[neighbour] = cluster_label;
                }
            }
        }
    }
    return predictions;
}

// Deterministic points spread over a grid with jitter, so the fixture exercises many clusters, noise
// and pruning decisions instead of a single clean split.
auto gapped_grid(std::size_t n_side) -> std::vector<dType> {
    auto data = std::vector<dType>{};

    std::size_t state = 12345;

    auto next_random = [&state]() {
        state = state * 1103515245 + 12345;
        return static_cast<dType>((state / 65536) % 1000) / 1000.0F;
    };

    for (std::size_t i = 0; i < n_side; ++i) {
        for (std::size_t j = 0; j < n_side; ++j) {
            data.push_back(static_cast<dType>(i) + next_random());
            data.push_back(static_cast<dType>(j) + next_random());
        }
    }
    return data;
}

// The accelerated labels must match the ones a pruning-free implementation produces, including which
// samples end up as noise.
TEST(DBSCANTest, MatchesReferenceOnAGridWithWidelySpreadPoints) {
    auto       data       = gapped_grid(/*n_side=*/18);  // 324 samples
    const auto n_features = std::size_t{2};
    const auto radius     = 1.05F;

    const auto expected = reference_dbscan(data, n_features, radius, 5);

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(radius).min_samples(5));

    const auto predictions = dbscan.predict(std::move(indexer));

    ASSERT_EQ(predictions.size(), expected.size());
    for (std::size_t sample_index = 0; sample_index < predictions.size(); ++sample_index) {
        EXPECT_EQ(predictions[sample_index] == 0, expected[sample_index] == 0)
            << "sample " << sample_index << " disagrees on being noise";
    }
    EXPECT_TRUE(Fixture::same_partition(predictions, expected)) << "the partitions differ";
    EXPECT_GT(Fixture::count_clusters(predictions, std::size_t{0}), 1u) << "the fixture must split";
}

// A radius large enough to reach across the whole grid: one cluster, no pruning can be too aggressive
// without changing that.
TEST(DBSCANTest, MatchesReferenceWithARadiusSpanningEveryPoint) {
    auto       data       = gapped_grid(/*n_side=*/12);  // 144 samples
    const auto n_features = std::size_t{2};
    const auto radius     = 50.0F;

    const auto expected = reference_dbscan(data, n_features, radius, 3);

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(radius).min_samples(3));

    const auto predictions = dbscan.predict(std::move(indexer));

    ASSERT_EQ(predictions.size(), expected.size());
    EXPECT_TRUE(Fixture::same_partition(predictions, expected));
    EXPECT_EQ(Fixture::count_clusters(predictions, std::size_t{0}), 1u);
}

// A border point visited and labelled noise before any cluster reaches it must still end up in the
// cluster whose core point covers it.
//
// The sample at index 0 is the border point: it sits within the radius of the second dense block but
// not of the first, so it is only reached once the expansion has already moved on to the second block.
// Being visited first, it is labelled noise and skipped by the frontier, which is the order that used
// to lose it.
TEST(DBSCANTest, BorderPointsVisitedBeforeTheirClusterAreNotLost) {
    const auto n_features  = std::size_t{2};
    const auto radius      = 0.5F;
    const auto min_samples = std::size_t{5};

    auto data = std::vector<dType>{
        // index 0: the border point, four neighbours, so not core at this minimum
        0.80F,
        0.00F,
        // indices 1-5: first dense block, core
        0.00F,
        0.00F,
        0.05F,
        0.00F,
        0.00F,
        0.05F,
        0.05F,
        0.05F,
        0.10F,
        0.00F,
        // indices 6-8: second dense block, core, and within reach of the border point
        0.40F,
        0.00F,
        0.42F,
        0.02F,
        0.42F,
        -0.02F,
        // index 9: genuine noise
        10.0F,
        10.0F,
    };

    const auto expected = reference_dbscan(data, n_features, radius, min_samples);

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto dbscan = ffcl::DBSCAN<decltype(indexer)>();
    dbscan.set_options(ffcl::DBSCAN<decltype(indexer)>::Options().radius(radius).min_samples(min_samples));

    const auto predictions = dbscan.predict(std::move(indexer));

    ASSERT_EQ(predictions.size(), expected.size());
    EXPECT_NE(predictions[0], 0u) << "the border point belongs to the block's cluster";
    EXPECT_EQ(predictions[0], predictions[1]) << "and carries that cluster's label";
    EXPECT_EQ(predictions[9], 0u) << "the far sample stays noise";
    EXPECT_TRUE(Fixture::same_partition(predictions, expected));
}

}  // namespace
