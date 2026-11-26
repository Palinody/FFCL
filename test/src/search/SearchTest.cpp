#include <gtest/gtest.h>

#include "ffcl/search/ClosestPairOfSamples.hpp"
#include "ffcl/search/Search.hpp"
#include "ffcl/search/buffer/Unsorted.hpp"
#include "ffcl/search/buffer/WithMemory.hpp"
#include "ffcl/search/buffer/WithUnionFind.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

using SamplesIterator = std::vector<dType>::iterator;

// The fixture is two squares: (0,0) (1,0) (0,1) (.5,.5) and the same shape at (9,9).
//
// The nearest neighbour of (0,0) is (1,0) or (0,1), both at distance 1; both are acceptable, so the
// assertions below check the distance rather than the index.
TEST(SearchTest, UniTreeKnnFindsTheNearestNeighbour) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto tree    = Fixture::make_kdtree(data, indices, n_features);

    auto searcher = ffcl::search::Searcher<decltype(tree)>(std::move(tree));

    constexpr std::size_t kQueryIndex = 0;

    auto buffer = searcher(ffcl::search::buffer::Unsorted<SamplesIterator>(
        searcher.features_range_first(kQueryIndex), searcher.features_range_last(kQueryIndex), std::size_t{2}));

    // the buffer holds the query itself (distance 0) and the nearest neighbour, which is the centre
    // of the square at distance sqrt(0.5) rather than a corner at distance 1
    ASSERT_EQ(buffer.size(), 2u);
    EXPECT_NEAR(buffer.closest_distance(), 0.0f, 1e-5f);
    EXPECT_NEAR(buffer.furthest_distance(), std::sqrt(dType{0.5f}), 1e-5f);
    EXPECT_NE(buffer.furthest_index(), kQueryIndex);
}

// A ball bound of radius 1.5 around (0,0) only reaches the other points of the first square.
TEST(SearchTest, RadiusSearchOnlyReturnsSamplesInsideTheBall) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto tree    = Fixture::make_kdtree(data, indices, n_features);

    auto searcher = ffcl::search::Searcher<decltype(tree)>(std::move(tree));

    constexpr std::size_t kQueryIndex = 0;
    constexpr auto        kRadius     = dType{1.5f};

    auto ball_view = ffcl::datastruct::bounds::BallView(searcher.features_range_first(kQueryIndex),
                                                        searcher.features_range_last(kQueryIndex),
                                                        kRadius);

    using BallViewType = ffcl::datastruct::bounds::BallView<SamplesIterator>;

    auto buffer = searcher(ffcl::search::buffer::Unsorted<SamplesIterator, BallViewType>(std::move(ball_view)));

    // (0,0) itself, (1,0), (0,1) and (.5,.5): everything of the first square, nothing of the second
    EXPECT_EQ(buffer.size(), 4u);

    for (const auto& index : buffer.indices()) {
        EXPECT_LT(index, 4u) << "sample " << index << " belongs to the second square";
    }
    for (const auto& distance : buffer.distances()) {
        EXPECT_LE(distance, kRadius);
    }
}

// WithMemory must skip the samples whose index is in the visited set.
TEST(SearchTest, WithMemoryBufferSkipsVisitedSamples) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto tree    = Fixture::make_kdtree(data, indices, n_features);

    auto searcher = ffcl::search::Searcher<decltype(tree)>(std::move(tree));

    constexpr std::size_t kQueryIndex = 0;

    // visit the query itself and its two nearest neighbours
    const auto visited = std::unordered_set<std::size_t>{0, 1, 2};

    auto buffer = searcher(ffcl::search::buffer::WithMemory<SamplesIterator>(
        searcher.features_range_first(kQueryIndex),
        searcher.features_range_last(kQueryIndex),
        visited,
        std::size_t{1}));

    ASSERT_EQ(buffer.size(), 1u);
    EXPECT_EQ(visited.count(buffer.closest_index()), 0u) << "a visited sample was returned";
    EXPECT_NEAR(buffer.closest_distance(), std::sqrt(dType{0.5f}), 1e-4f);  // (0,0) -> (.5,.5)
}

// WithUnionFind must skip the samples that belong to the query's component.
TEST(SearchTest, WithUnionFindBufferSkipsSameComponentSamples) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    const auto n_samples = data.size() / n_features;

    auto indices = Fixture::make_indices(n_samples);
    auto tree    = Fixture::make_kdtree(data, indices, n_features);

    auto searcher = ffcl::search::Searcher<decltype(tree)>(std::move(tree));

    // put the whole first square in one component
    auto union_find = ffcl::datastruct::UnionFind<std::size_t>(n_samples);

    union_find.merge(std::size_t{0}, std::size_t{1});
    union_find.merge(std::size_t{0}, std::size_t{2});
    union_find.merge(std::size_t{0}, std::size_t{3});

    constexpr std::size_t kQueryIndex = 0;

    auto buffer = searcher(ffcl::search::buffer::WithUnionFind<SamplesIterator>(
        searcher.features_range_first(kQueryIndex),
        searcher.features_range_last(kQueryIndex),
        union_find,
        union_find.find(kQueryIndex),
        std::size_t{1}));

    ASSERT_EQ(buffer.size(), 1u);
    EXPECT_GE(buffer.closest_index(), 4u) << "a sample of the query's own component was returned";
    EXPECT_NEAR(buffer.closest_distance(), std::sqrt(dType{2 * 9 * 9}), 1e-3f);  // (0,0) -> (9,9)
}

// The dual-tree traversal is the accelerated path of the Boruvka step: on a fixture this small it
// must agree with the brute-force answer.
TEST(SearchTest, BruteForceShortestEdgeIsExact) {
    auto&       data       = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    // disjoint sets: over a single set the traversal legitimately pairs a sample with itself
    auto query_indices     = Fixture::make_indices(4);
    auto reference_indices = std::vector<std::size_t>{4, 5, 6, 7};

    const auto brute_force = ffcl::search::algorithms::simple_dual_set_shortest_edge(query_indices.begin(),
                                                                                     query_indices.end(),
                                                                                     data.begin(),
                                                                                     data.end(),
                                                                                     n_features,
                                                                                     reference_indices.begin(),
                                                                                     reference_indices.end(),
                                                                                     data.begin(),
                                                                                     data.end(),
                                                                                     n_features);

    // the closest cross-square pair is (.5,.5) -> (9,9) at sqrt(144.5)
    EXPECT_NEAR(std::get<2>(brute_force), 12.020815280171307, 1e-4);
}

}  // namespace
