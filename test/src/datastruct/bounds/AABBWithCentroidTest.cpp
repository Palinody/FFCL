#include <gtest/gtest.h>

#include "ffcl/datastruct/bounds/AABBWithCentroid.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

#include <array>
#include <limits>

TEST(AABBWithCentroidTest, MinDistanceTest) {
    using ValueType = double;

    using SegmentType = ffcl::datastruct::bounds::segment::LowerBoundAndUpperBound<ValueType>;

    const auto segment1 = SegmentType(0, 1);
    const auto segment2 = SegmentType(0, 1);

    const auto segment3 = SegmentType(0, 10);
    const auto segment4 = SegmentType(2, 2);

    const auto segments_vector_1 = std::vector<SegmentType>{segment1, segment2};
    const auto segments_vector_2 = std::vector<SegmentType>{segment3, segment4};

    const auto aabb1 = ffcl::datastruct::bounds::AABBWithCentroid<SegmentType>{segments_vector_1};
    const auto aabb2 = ffcl::datastruct::bounds::AABBWithCentroid<SegmentType>{segments_vector_2};

    const ValueType min_dist = aabb1.min_distance(aabb2);

    // std::cout << min_dist << "\n";

    EXPECT_NEAR(min_dist, 1, std::numeric_limits<ValueType>::epsilon());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
