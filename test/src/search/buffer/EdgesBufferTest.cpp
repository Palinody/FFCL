#include <gtest/gtest.h>

#include "ffcl/datastruct/graph/spanning_tree/MinimumSpanningTree.hpp"
#include "ffcl/search/buffer/EdgesBuffer.hpp"
#include "ffcl/search/buffer/WithUnionFind.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// EdgesBuffer works in squared mutual reachability distances, so with all-zero core distances the
// buffered weight of an edge is the squared Euclidean distance.
TEST(EdgesBufferTest, IsConstructibleOverAFixtureTree) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};
    const auto  n_samples  = data.size() / n_features;

    auto indices = Fixture::make_indices(n_samples);
    auto tree    = Fixture::make_kdtree(data, indices, n_features);

    auto union_find = ffcl::datastruct::UnionFind<std::size_t>(n_samples);
    auto core       = std::vector<dType>(n_samples, 0.0f);

    auto edge_buffer = ffcl::search::buffer::make_edge_buffer(tree, tree, union_find, core.data());

    EXPECT_TRUE(edge_buffer.component_to_shortest_edge().empty())
        << "a fresh buffer has not found any edge yet";
}

// The buffer must never report an edge between two samples of the same component.
TEST(EdgesBufferTest, NeverBuffersAnEdgeInsideOneComponent) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};
    const auto  n_samples  = data.size() / n_features;

    auto indices = Fixture::make_indices(n_samples);
    auto tree    = Fixture::make_kdtree(data, indices, n_features);

    // put everything in a single component: there is no valid edge to find
    auto union_find = ffcl::datastruct::UnionFind<std::size_t>(n_samples);

    for (std::size_t index = 1; index < n_samples; ++index) {
        union_find.merge(std::size_t{0}, index);
    }
    auto core = std::vector<dType>(n_samples, 0.0f);

    auto edge_buffer = ffcl::search::buffer::make_edge_buffer(tree, tree, union_find, core.data());

    edge_buffer.base_case(tree.root(), tree.root());

    EXPECT_TRUE(edge_buffer.component_to_shortest_edge().empty())
        << "an edge was buffered although every sample belongs to the same component";
}

// With two disjoint components the buffer must find exactly one shortest edge, whose weight is the
// squared distance between the two closest samples.
TEST(EdgesBufferTest, FindsTheShortestCrossComponentEdge) {
    auto& data             = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};
    const auto  n_samples  = data.size() / n_features;

    auto indices = Fixture::make_indices(n_samples);
    // a bucket size of n_samples keeps the whole fixture in the root leaf, which is what base_case
    // operates on
    auto tree = Fixture::make_kdtree(data, indices, n_features, n_samples);

    // two components: the first square and the second square
    auto union_find = ffcl::datastruct::UnionFind<std::size_t>(n_samples);

    union_find.merge(std::size_t{0}, std::size_t{1});
    union_find.merge(std::size_t{0}, std::size_t{2});
    union_find.merge(std::size_t{0}, std::size_t{3});
    union_find.merge(std::size_t{4}, std::size_t{5});
    union_find.merge(std::size_t{4}, std::size_t{6});
    union_find.merge(std::size_t{4}, std::size_t{7});

    auto core = std::vector<dType>(n_samples, 0.0f);

    auto edge_buffer = ffcl::search::buffer::make_edge_buffer(tree, tree, union_find, core.data());

    edge_buffer.base_case(tree.root(), tree.root());

    const auto edges = edge_buffer.component_to_shortest_edge();

    ASSERT_FALSE(edges.empty());

    // the closest cross-component pair is (.5,.5) -> (9,9), squared distance 144.5
    auto shortest_squared_weight = ffcl::common::infinity<dType>();
    for (const auto& edge : edges) {
        shortest_squared_weight = std::min(shortest_squared_weight, std::get<2>(edge));
    }
    EXPECT_NEAR(shortest_squared_weight, 144.5f, 1e-3f);
}

}  // namespace
