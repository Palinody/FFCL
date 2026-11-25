#include <gtest/gtest.h>

#include "ffcl/common/math/random/VosesAliasMethod.hpp"

namespace {

TEST(VosesAliasMethodTest, AlwaysReturnsAnAliasInRange) {
    auto sampler = ffcl::common::math::random::VosesAliasMethod<float>({1.0f, 1.0f, 1.0f, 1.0f});

    for (std::size_t draw = 0; draw < 1000; ++draw) {
        EXPECT_LT(sampler(), 4u);
    }
}

// With a single non-zero weight the sampler must always return that index.
TEST(VosesAliasMethodTest, DegenerateWeightSelectsTheOnlyPossibleIndex) {
    auto sampler = ffcl::common::math::random::VosesAliasMethod<float>({0.0f, 0.0f, 1.0f});

    for (std::size_t draw = 0; draw < 100; ++draw) {
        EXPECT_EQ(sampler(), 2u);
    }
}

// A heavily skewed distribution must be sampled proportionally: with weights {9, 1} roughly 90% of
// the draws are index 0. The bound is loose so the test cannot flake.
TEST(VosesAliasMethodTest, SamplingFollowsTheWeights) {
    auto sampler = ffcl::common::math::random::VosesAliasMethod<float>({9.0f, 1.0f});

    auto counts = std::array<std::size_t, 2>{};

    constexpr std::size_t kDraws = 20000;

    for (std::size_t draw = 0; draw < kDraws; ++draw) {
        ++counts[sampler()];
    }

    const auto fraction_of_zero = static_cast<double>(counts[0]) / static_cast<double>(kDraws);

    EXPECT_GT(fraction_of_zero, 0.80);
    EXPECT_LT(fraction_of_zero, 0.98);
}

}  // namespace
