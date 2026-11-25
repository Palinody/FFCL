#include <gtest/gtest.h>

#include "ffcl/datastruct/Interval.hpp"

namespace {

using dType    = float;
using Segment  = ffcl::datastruct::bounds::segment::LowerBoundAndUpperBound<dType>;
using AABBType = ffcl::datastruct::bounds::AABBWithCentroid<Segment>;

// 4 samples, 2 features:  x in [0, 3], y in [-1, 2]
const std::vector<dType> kData = {
    0.0f, -1.0f,  // sample 0
    3.0f, 2.0f,   // sample 1
    1.0f, 0.5f,   // sample 2
    2.0f, 1.0f,   // sample 3
};

TEST(IntervalTest, TightSegmentCoversTheWholeData) {
    const auto segment = ffcl::datastruct::make_tight_segment(kData.begin(), kData.end(), 2, 0);

    EXPECT_NEAR(segment.lower_bound(), 0.0f, 1e-6f);
    EXPECT_NEAR(segment.upper_bound(), 3.0f, 1e-6f);
}

TEST(IntervalTest, TightSegmentOnASingleFeature) {
    const auto segment_x = ffcl::datastruct::make_tight_segment(kData.begin(), kData.end(), 2, 0);
    const auto segment_y = ffcl::datastruct::make_tight_segment(kData.begin(), kData.end(), 2, 1);

    EXPECT_NEAR(segment_x.lower_bound(), 0.0f, 1e-6f);
    EXPECT_NEAR(segment_x.upper_bound(), 3.0f, 1e-6f);
    EXPECT_NEAR(segment_y.lower_bound(), -1.0f, 1e-6f);
    EXPECT_NEAR(segment_y.upper_bound(), 2.0f, 1e-6f);
}

// Restricting to a subset of the indices must produce a bound that covers only that subset.
TEST(IntervalTest, TightSegmentOverASubsetOfIndices) {
    const auto indices = std::vector<std::size_t>{2, 3};  // x in [1, 2]

    const auto segment =
        ffcl::datastruct::make_tight_segment(indices.begin(), indices.end(), kData.begin(), kData.end(), 2, 0);

    EXPECT_NEAR(segment.lower_bound(), 1.0f, 1e-6f);
    EXPECT_NEAR(segment.upper_bound(), 2.0f, 1e-6f);
}

TEST(IntervalTest, TightBoundCoversEverySample) {
    const auto bound = ffcl::datastruct::make_tight_bound(kData.begin(), kData.end(), 2);

    EXPECT_NEAR(bound.segment_at(0).lower_bound(), 0.0f, 1e-6f);
    EXPECT_NEAR(bound.segment_at(0).upper_bound(), 3.0f, 1e-6f);
    EXPECT_NEAR(bound.segment_at(1).lower_bound(), -1.0f, 1e-6f);
    EXPECT_NEAR(bound.segment_at(1).upper_bound(), 2.0f, 1e-6f);

    for (std::size_t sample_index = 0; sample_index < kData.size() / 2; ++sample_index) {
        for (std::size_t feature_index = 0; feature_index < 2; ++feature_index) {
            const auto value = kData[sample_index * 2 + feature_index];
            EXPECT_GE(value, bound.segment_at(feature_index).lower_bound());
            EXPECT_LE(value, bound.segment_at(feature_index).upper_bound());
        }
    }
}

TEST(IntervalTest, BoundDiameterIsTheDiagonalOfTheBox) {
    const auto bound = ffcl::datastruct::make_tight_bound(kData.begin(), kData.end(), 2);

    // box sides 3 and 3 -> diagonal sqrt(18)
    EXPECT_NEAR(bound.diameter(), std::sqrt(dType{18.0f}), 1e-5f);
}

}  // namespace
