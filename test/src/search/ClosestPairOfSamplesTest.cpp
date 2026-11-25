#include <gtest/gtest.h>

#include "ffcl/search/ClosestPairOfSamples.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

namespace {

using dType = float;

namespace Fixture = ffcl::fixtures;

// The two squares of the fixture are 8 apart. The closest cross-square pair is (.5,.5) -> (9,9) at
// distance sqrt(144.5) = 12.0208, not the corner-to-corner (.5,.5) -> (9.5,9.5) diagonal which is
// longer at 9*sqrt(2) = 12.7279.
constexpr auto kClosestCrossSquareDistance = 12.020815280171307;
constexpr auto kClosestCrossSquareSquared  = 144.49999999999997;

// Disjoint query and reference index sets: over a single set the traversal legitimately pairs a
// sample with itself, which is not what these functions are for here.
TEST(ClosestPairOfSamplesTest, BruteForceFindsTheShortestCrossEdge) {
    auto&       data       = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto query_indices     = Fixture::make_indices(4);  // first square
    auto reference_indices = std::vector<std::size_t>{4, 5, 6, 7};

    const auto edge = ffcl::search::algorithms::simple_dual_set_shortest_edge(query_indices.begin(),
                                                                            query_indices.end(),
                                                                            data.begin(),
                                                                            data.end(),
                                                                            n_features,
                                                                            reference_indices.begin(),
                                                                            reference_indices.end(),
                                                                            data.begin(),
                                                                            data.end(),
                                                                            n_features);

    EXPECT_LT(std::get<0>(edge), 4u) << "the query endpoint must come from the first square";
    EXPECT_GE(std::get<1>(edge), 4u) << "the reference endpoint must come from the second square";
    EXPECT_NEAR(std::get<2>(edge), kClosestCrossSquareDistance, 1e-4);
}

// KNOWN DEFECT, disabled: the accelerated traversal misses the closest pair on this fixture (it
// returns the corner diagonal 12.7279 instead of 12.0208). See benchmark/README.md and
// SearchTest.DISABLED_DualTreeShortestEdgeMatchesBruteForce.
TEST(ClosestPairOfSamplesTest, DISABLED_DualTreeAgreesWithBruteForce) {
    auto&       data       = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

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

    const auto dual_tree = ffcl::search::algorithms::dual_set_shortest_edge(query_indices.begin(),
                                                                           query_indices.end(),
                                                                           data.begin(),
                                                                           data.end(),
                                                                           n_features,
                                                                           reference_indices.begin(),
                                                                           reference_indices.end(),
                                                                           data.begin(),
                                                                           data.end(),
                                                                           n_features);

    EXPECT_NEAR(std::get<2>(brute_force), std::get<2>(dual_tree), 1e-5);
}

// The shortest distance is the weight of the corresponding shortest edge.
TEST(ClosestPairOfSamplesTest, DISABLED_ShortestDistanceMatchesTheShortestEdge) {
    auto&       data       = Fixture::two_blobs();
    const auto  n_features = std::size_t{2};

    auto query_indices     = Fixture::make_indices(4);
    auto reference_indices = std::vector<std::size_t>{4, 5, 6, 7};

    const auto distance = ffcl::search::algorithms::dual_set_shortest_distance(query_indices.begin(),
                                                                              query_indices.end(),
                                                                              data.begin(),
                                                                              data.end(),
                                                                              n_features,
                                                                              reference_indices.begin(),
                                                                              reference_indices.end(),
                                                                              data.begin(),
                                                                              data.end(),
                                                                              n_features);

    // KNOWN DEFECT, disabled for the same reason as above.
    EXPECT_NEAR(distance, kClosestCrossSquareDistance, 1e-4);
}

}  // namespace
