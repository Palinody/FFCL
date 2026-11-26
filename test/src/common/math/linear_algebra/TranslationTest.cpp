#include <gtest/gtest.h>

#include "ffcl/common/math/linear_algebra/Translation.hpp"

namespace {

namespace linear_algebra = ffcl::common::math::linear_algebra;

TEST(TranslationTest, TranslatesTheRightRangeToTheLeftRangeOrigin) {
    // left range is the origin of the translation, right range is shifted by (1, 2)
    const auto left  = std::vector<float>{0.0f, 0.0f};
    const auto right = std::vector<float>{1.0f, 2.0f};

    const auto translated =
        linear_algebra::translate_right_range_to_origin(left.begin(), left.end(), right.begin(), right.end());

    ASSERT_EQ(translated.size(), 2u);
    EXPECT_NEAR(translated[0], 1.0f, 1e-6f);
    EXPECT_NEAR(translated[1], 2.0f, 1e-6f);
}

TEST(TranslationTest, TranslatingAZeroVectorKeepsItZero) {
    const auto left  = std::vector<float>{3.0f, 3.0f};
    const auto right = std::vector<float>{3.0f, 3.0f};

    const auto translated =
        linear_algebra::translate_right_range_to_origin(left.begin(), left.end(), right.begin(), right.end());

    ASSERT_EQ(translated.size(), 2u);
    EXPECT_NEAR(translated[0], 0.0f, 1e-6f);
    EXPECT_NEAR(translated[1], 0.0f, 1e-6f);
}

TEST(TranslationTest, ResultSizeMatchesTheRightRange) {
    const auto left  = std::vector<float>{0.0f, 0.0f, 0.0f};
    const auto right = std::vector<float>{1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f};

    const auto translated =
        linear_algebra::translate_right_range_to_origin(left.begin(), left.end(), right.begin(), right.end());

    // one vector per sample of the right range
    EXPECT_EQ(translated.size(), 2u * 3u);
}

}  // namespace
