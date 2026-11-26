#include <gtest/gtest.h>

#include "ffcl/common/math/random/Sampling.hpp"

#include <unordered_set>

namespace {

namespace sampling = ffcl::common::math::random;

TEST(SamplingTest, SelectNElementsReturnsRequestedCount) {
    auto indices = std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const auto selection = sampling::select_n_elements(indices.begin(), indices.end(), std::size_t{4});

    EXPECT_EQ(selection.size(), 4u);
}

TEST(SamplingTest, SelectNElementsReturnsDistinctIndices) {
    auto indices = std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    const auto selection = sampling::select_n_elements(indices.begin(), indices.end(), std::size_t{5});

    const auto distinct = std::unordered_set<std::size_t>(selection.begin(), selection.end());

    EXPECT_EQ(distinct.size(), selection.size()) << "the selection contains duplicates";
}

TEST(SamplingTest, SelectNElementsOnlyReturnsValuesFromTheRange) {
    auto indices = std::vector<std::size_t>{10, 11, 12, 13, 14};

    const auto selection = sampling::select_n_elements(indices.begin(), indices.end(), std::size_t{3});

    for (const auto& value : selection) {
        EXPECT_GE(value, 10u);
        EXPECT_LE(value, 14u);
    }
}

}  // namespace
