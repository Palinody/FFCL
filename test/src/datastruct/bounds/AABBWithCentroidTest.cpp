#include <gtest/gtest.h>

#include "ffcl/datastruct/bounds/AABB.hpp"
#include "ffcl/datastruct/bounds/AABBWithCentroid.hpp"
#include "ffcl/datastruct/bounds/Ball.hpp"
#include "ffcl/datastruct/bounds/UnboundedBall.hpp"
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

TEST(AABBWithCentroidTest, MinDistanceBallTest) {
    using ValueType = double;

    const auto ball_1 = ffcl::datastruct::bounds::Ball<ValueType>{{-10, 0}, 1};
    const auto ball_2 = ffcl::datastruct::bounds::Ball<ValueType>{{10, 0}, 5};

    const ValueType min_dist = ball_1.min_distance(ball_2);

    // std::cout << min_dist << "\n";

    EXPECT_NEAR(min_dist, 14, std::numeric_limits<ValueType>::epsilon());
}

TEST(AABBWithCentroidTest, MinDistanceUnboundedBallTest) {
    using ValueType = double;

    const auto unbounded_ball_1 = ffcl::datastruct::bounds::UnboundedBall<ValueType>{{-10, 0}};
    const auto unbounded_ball_2 = ffcl::datastruct::bounds::UnboundedBall<ValueType>{{10, 0}};

    const ValueType min_dist = unbounded_ball_1.min_distance(unbounded_ball_2);

    // std::cout << min_dist << "\n";

    EXPECT_NEAR(min_dist, 0, std::numeric_limits<ValueType>::epsilon());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}