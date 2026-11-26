#include <gtest/gtest.h>

#include "ffcl/datastruct/vector/FeaturesMask.hpp"

namespace {

using FeaturesMask = ffcl::datastruct::FeaturesMask<std::size_t>;

TEST(FeaturesMaskTest, DefaultMaskIsTheIdentity) {
    const auto mask = FeaturesMask(3);

    EXPECT_EQ(mask.size(), 3u);
    EXPECT_EQ(mask[0], 0u);
    EXPECT_EQ(mask[1], 1u);
    EXPECT_EQ(mask[2], 2u);
}

TEST(FeaturesMaskTest, BuildFromInitializerList) {
    const auto mask = FeaturesMask({2, 0});

    EXPECT_EQ(mask.size(), 2u);
    EXPECT_EQ(mask[0], 2u);
    EXPECT_EQ(mask[1], 0u);
}

// The mask is used to select features in an arbitrary order, so it must preserve that order.
TEST(FeaturesMaskTest, PreservesTheRequestedOrder) {
    const auto samples    = std::vector<float>{10.0f, 20.0f, 30.0f};
    const auto mask       = FeaturesMask({2, 0});
    auto       selection  = std::vector<float>{};

    for (const auto& feature_index : mask) {
        selection.emplace_back(samples[feature_index]);
    }

    ASSERT_EQ(selection.size(), 2u);
    EXPECT_FLOAT_EQ(selection[0], 30.0f);
    EXPECT_FLOAT_EQ(selection[1], 10.0f);
}

}  // namespace
