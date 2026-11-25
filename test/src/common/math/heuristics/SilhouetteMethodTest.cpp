#include <gtest/gtest.h>

#include "ffcl/common/math/heuristics/SilhouetteMethod.hpp"

namespace {

using dType = float;

// Two clusters that are far apart, each with a tight spread: every sample should score close to 1.
const std::vector<dType> kWellSeparated = {
    // clang-format off
    0.0f, 0.0f,  0.1f, 0.0f,  0.0f, 0.1f,
    10.0f, 10.0f, 10.1f, 10.0f, 10.0f, 10.1f,
    // clang-format on
};

const std::vector<std::size_t> kWellSeparatedLabels = {0, 0, 0, 1, 1, 1};

TEST(SilhouetteMethodTest, WellSeparatedClustersScoreNearOne) {
    const auto scores = ffcl::common::math::heuristics::silhouette(kWellSeparated.begin(),
                                                                  kWellSeparated.end(),
                                                                  2,
                                                                  kWellSeparatedLabels.begin(),
                                                                  kWellSeparatedLabels.end());

    ASSERT_EQ(scores.size(), kWellSeparatedLabels.size());

    const auto mean = ffcl::common::math::heuristics::get_average_silhouette(scores.begin(), scores.end());

    EXPECT_GT(mean, 0.9f);
    EXPECT_LE(mean, 1.0f);
}

// A single cluster has no sibling to compare against and the implementation divides by zero, so the
// mean silhouette is NaN. It cannot be asserted here because the library is built with -Ofast, which
// implies -ffinite-math-only and makes std::isnan() a constant false. Recorded in
// benchmark/README.md instead.

// Interleaving the clusters (alternate labels) must lower the score below the well separated case.
TEST(SilhouetteMethodTest, InterleavedLabelsScoreLower) {
    const auto interleaved_labels = std::vector<std::size_t>{0, 1, 0, 1, 0, 1};

    const auto well_separated_scores = ffcl::common::math::heuristics::silhouette(kWellSeparated.begin(),
                                                                                 kWellSeparated.end(),
                                                                                 2,
                                                                                 kWellSeparatedLabels.begin(),
                                                                                 kWellSeparatedLabels.end());

    const auto interleaved_scores = ffcl::common::math::heuristics::silhouette(
        kWellSeparated.begin(), kWellSeparated.end(), 2, interleaved_labels.begin(), interleaved_labels.end());

    const auto well_separated_mean =
        ffcl::common::math::heuristics::get_average_silhouette(well_separated_scores.begin(),
                                                               well_separated_scores.end());
    const auto interleaved_mean =
        ffcl::common::math::heuristics::get_average_silhouette(interleaved_scores.begin(), interleaved_scores.end());

    EXPECT_LT(interleaved_mean, well_separated_mean);
}

}  // namespace
