#include <gtest/gtest.h>

#include "ffcl/datastruct/tree/kdtree/policy/SplittingRulePolicy.hpp"

#include "base_fixtures/ClusteringFixtures.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace {

using IndicesIterator = std::vector<std::size_t>::iterator;
using SamplesIterator = std::vector<float>::iterator;

using MidpointRule = ffcl::datastruct::kdtree::policy::MidpointRange<IndicesIterator, SamplesIterator>;

// The contract the build relies on: the pivot is the last element of the left range, so it stays owned by
// the left child and the leaves still cover every sample exactly once; every value on the right is
// strictly greater than the pivot, so `query < pivot ? left : right` routes correctly.
TEST(MidpointRangeTest, PivotIsTheLargestValueOnTheLeftAndTheRightIsStrictlyGreater) {
    auto&      data       = ffcl::fixtures::two_blobs();
    const auto n_features = std::size_t{2};

    auto indices = ffcl::fixtures::make_indices(data.size() / n_features);

    const auto rule = MidpointRule{};

    const auto [pivot_index, left_range, median_range, right_range] =
        rule(indices.begin(), indices.end(), data.begin(), data.end(), n_features, /*feature_index=*/0);

    ASSERT_FALSE(left_range.first == left_range.second) << "the left side must not be empty";
    ASSERT_FALSE(right_range.first == right_range.second) << "the right side must not be empty";

    EXPECT_EQ(median_range.first, std::prev(left_range.second)) << "the pivot is the last element on the left";
    EXPECT_EQ(median_range.second, right_range.first);
    EXPECT_EQ(pivot_index, *median_range.first);

    const auto pivot_value = data[pivot_index * n_features];

    for (auto index_it = left_range.first; index_it != left_range.second; ++index_it) {
        EXPECT_LE(data[*index_it * n_features], pivot_value);
    }
    for (auto index_it = right_range.first; index_it != right_range.second; ++index_it) {
        EXPECT_GT(data[*index_it * n_features], pivot_value);
    }
}

// The three ranges must partition the input: nothing lost, nothing duplicated.
TEST(MidpointRangeTest, TheRangesPartitionTheInput) {
    auto&      data       = ffcl::fixtures::two_blobs();
    const auto n_features = std::size_t{2};

    auto indices = ffcl::fixtures::make_indices(data.size() / n_features);

    const auto rule = MidpointRule{};

    const auto [pivot_index, left_range, median_range, right_range] =
        rule(indices.begin(), indices.end(), data.begin(), data.end(), n_features, /*feature_index=*/0);

    (void)pivot_index;
    (void)median_range;

    EXPECT_EQ(std::distance(left_range.first, left_range.second) + std::distance(right_range.first, right_range.second),
              static_cast<std::ptrdiff_t>(indices.size()));
    EXPECT_EQ(left_range.second, right_range.first);
}

// All values equal on the axis: the midpoint has no extent, so it must fall back to the median rather than
// leaving one side empty.
TEST(MidpointRangeTest, FallsBackToTheMedianWhenTheAxisHasNoExtent) {
    auto       data       = std::vector<float>{0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 2.0F, 0.0F, 3.0F};
    const auto n_features = std::size_t{2};

    auto indices = ffcl::fixtures::make_indices(data.size() / n_features);

    const auto rule = MidpointRule{};

    const auto [pivot_index, left_range, median_range, right_range] =
        rule(indices.begin(), indices.end(), data.begin(), data.end(), n_features, /*feature_index=*/0);

    (void)pivot_index;
    (void)median_range;

    EXPECT_FALSE(left_range.first == left_range.second);
    EXPECT_FALSE(right_range.first == right_range.second);
    EXPECT_EQ(std::distance(left_range.first, left_range.second) + std::distance(right_range.first, right_range.second),
              static_cast<std::ptrdiff_t>(indices.size()));
}

}  // namespace

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
