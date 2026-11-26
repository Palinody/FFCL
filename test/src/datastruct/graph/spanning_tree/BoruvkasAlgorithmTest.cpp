#include <gtest/gtest.h>

#include "ffcl/datastruct/graph/spanning_tree/BoruvkasAlgorithm.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// A minimum spanning tree over `n` samples has exactly `n - 1` edges and every edge weight is finite.
template <typename MstType>
void expect_valid_spanning_tree(const MstType& mst, std::size_t n_samples, double expected_total_weight = -1) {
    ASSERT_EQ(mst.size(), n_samples - 1);

    auto total_weight = double{};

    for (const auto& edge : mst) {
        const auto weight = static_cast<double>(std::get<2>(edge));

        EXPECT_TRUE(std::isfinite(weight)) << "non finite edge weight";
        EXPECT_GE(weight, 0.0) << "negative edge weight";

        total_weight += weight;
    }
    if (expected_total_weight >= 0) {
        EXPECT_NEAR(total_weight, expected_total_weight, 1e-4);
    }
}

// Two squares of four points each: the MST is the three edges that connect the corners of each
// square, plus one edge bridging the two squares.
//
//   square 0: (0,0) (1,0) (0,1) (.5,.5)  -> unit square edges 1, 1, sqrt(.5), sqrt(.5), sqrt(.5)
//   square 1: same shape translated by (9,9)
//   bridge  : (.5,.5) -> (9.5,9.5), length sqrt(2*9^2) = 9*sqrt(2)
//
// The minimum spanning tree is not unique here (several equal-length diagonals), so the test checks
// the edge count, that every weight is finite, and the total weight, which is unique.
TEST(BoruvkasAlgorithmTest, SpanningTreeOverTwoSquaresIsValid) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<decltype(indexer)>();
    boruvkas_algorithm.set_options(typename ffcl::BoruvkasAlgorithm<decltype(indexer)>::Options().k_nearest_neighbors(2));

    const auto mst = boruvkas_algorithm.make_tree_2(std::move(indexer));

    expect_valid_spanning_tree(mst, 8);
}

// make_tree (single-tree Boruvka) and make_tree_2 (dual-tree Boruvka) must build the same tree
// weight. They take different search paths to the same answer, so this is the cross-check that
// neither has regressed.
TEST(BoruvkasAlgorithmTest, SingleTreeAndDualTreeAgreeOnTotalWeight) {
    const auto n_features = std::size_t{2};
    const auto n_samples  = Fixture::two_blobs().size() / n_features;

    auto total_weight_of = [&](bool use_dual_tree) {
        auto& data          = Fixture::two_blobs();
        auto        indices = Fixture::make_indices(n_samples);
        auto        indexer = Fixture::make_kdtree(data, indices, n_features);

        auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<decltype(indexer)>();
        boruvkas_algorithm.set_options(
            typename ffcl::BoruvkasAlgorithm<decltype(indexer)>::Options().k_nearest_neighbors(2));

        const auto mst =
            use_dual_tree ? boruvkas_algorithm.make_tree_2(std::move(indexer)) : boruvkas_algorithm.make_tree(std::move(indexer));

        auto total = double{};
        for (const auto& edge : mst) {
            total += static_cast<double>(std::get<2>(edge));
        }
        return total;
    };

    EXPECT_NEAR(total_weight_of(false), total_weight_of(true), 1e-4);
}

// With k = 1 the core distance of every sample is zero, which is a degenerate configuration: the
// tree must still be a valid spanning tree rather than silently producing infinite weights.
TEST(BoruvkasAlgorithmTest, KEqualsOneStillProducesAFiniteTree) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto indices = Fixture::make_indices(data.size() / n_features);
    auto indexer = Fixture::make_kdtree(data, indices, n_features);

    auto boruvkas_algorithm = ffcl::BoruvkasAlgorithm<decltype(indexer)>();
    boruvkas_algorithm.set_options(typename ffcl::BoruvkasAlgorithm<decltype(indexer)>::Options().k_nearest_neighbors(1));

    const auto mst = boruvkas_algorithm.make_tree_2(std::move(indexer));

    expect_valid_spanning_tree(mst, 8);
}

// merging an edge whose endpoints already share a component must be a no-op, so merging the same
// edge twice cannot grow the tree or change the component count.
TEST(MSTBuilderTest, MergingTheSameEdgeTwiceIsANoOp) {
    auto builder = ffcl::MSTBuilder<std::size_t, dType>(4);

    EXPECT_EQ(builder.n_components(), 4u);

    const auto edge = ffcl::datastruct::mst::make_edge(std::size_t{0}, std::size_t{1}, dType{1.0f});

    builder.merge_components(edge);
    EXPECT_EQ(builder.n_components(), 3u);

    builder.merge_components(edge);
    EXPECT_EQ(builder.n_components(), 3u) << "merging an edge inside one component must be ignored";

    EXPECT_EQ(std::move(builder).minimum_spanning_tree().size(), 1u);
}

TEST(MSTBuilderTest, MergingDisjointComponentsBuildsATree) {
    auto builder = ffcl::MSTBuilder<std::size_t, dType>(4);

    builder.merge_components(ffcl::datastruct::mst::make_edge(std::size_t{0}, std::size_t{1}, dType{1.0f}));
    builder.merge_components(ffcl::datastruct::mst::make_edge(std::size_t{2}, std::size_t{3}, dType{2.0f}));
    builder.merge_components(ffcl::datastruct::mst::make_edge(std::size_t{1}, std::size_t{2}, dType{3.0f}));

    EXPECT_EQ(builder.n_components(), 1u);
    EXPECT_EQ(std::move(builder).minimum_spanning_tree().size(), 3u);
}

}  // namespace
