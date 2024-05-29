#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/datastruct/bounds/AABB.hpp"
#include "ffcl/datastruct/bounds/AABBWithCentroid.hpp"
#include "ffcl/datastruct/bounds/Ball.hpp"
#include "ffcl/datastruct/bounds/UnboundedBall.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

#include "ffcl/datastruct/bounds/segment/distances/MinDistance.hpp"

namespace ffcl::datastruct::bounds {

template <typename FirstSegment, typename SecondSegment, std::size_t NFeatures = 0>
constexpr auto min_distance(const AABBWithCentroid<FirstSegment, NFeatures>&  first_aabb_with_centroids,
                            const AABBWithCentroid<SecondSegment, NFeatures>& second_aabb_with_centroids) {
    using ValueType = std::common_type_t<typename FirstSegment::ValueType, typename SecondSegment::ValueType>;

    ValueType inner_lengths_sum = 0;

    // Compute inner lengths using the min_distance method of each segment.
    for (std::size_t feature_index = 0; feature_index < second_aabb_with_centroids.n_features(); ++feature_index) {
        const auto& first_segment  = first_aabb_with_centroids.segment_at(feature_index);
        const auto& second_segment = second_aabb_with_centroids.segment_at(feature_index);

        const auto segments_min_distance = min_distance(first_segment, second_segment);

        inner_lengths_sum += segments_min_distance * segments_min_distance;
    }
    return std::sqrt(inner_lengths_sum);
}

template <typename FirstSegment, typename SecondSegment, std::size_t NFeatures = 0>
constexpr auto min_distance(const AABB<FirstSegment, NFeatures>&  first_aabb,
                            const AABB<SecondSegment, NFeatures>& second_aabb) {
    using ValueType = std::common_type_t<typename FirstSegment::ValueType, typename SecondSegment::ValueType>;

    ValueType inner_lengths_sum = 0;

    // Compute inner lengths using the min_distance method of each segment.
    for (std::size_t feature_index = 0; feature_index < second_aabb.n_features(); ++feature_index) {
        const auto& first_segment  = first_aabb.segment_at(feature_index);
        const auto& second_segment = second_aabb.segment_at(feature_index);

        const auto segments_min_distance = min_distance(first_segment, second_segment);

        inner_lengths_sum += segments_min_distance * segments_min_distance;
    }
    return std::sqrt(inner_lengths_sum);
}

template <typename FirstValue, typename SecondValue, std::size_t NFeatures = 0>
constexpr auto min_distance(const Ball<FirstValue, NFeatures>&  first_ball,
                            const Ball<SecondValue, NFeatures>& second_ball) {
    using ValueType = std::common_type_t<FirstValue, SecondValue>;

    const auto centroids_distance = common::math::heuristics::auto_distance(first_ball.centroid_begin(),
                                                                            first_ball.centroid_end(),
                                                                            second_ball.centroid_begin(),
                                                                            second_ball.centroid_begin());

    return std::max(static_cast<ValueType>(0),
                    centroids_distance - (first_ball.centroid_to_furthest_bound_distance() +
                                          second_ball.centroid_to_furthest_bound_distance()));
}

template <typename FirstFeaturesIterator, typename SecondFeaturesIterator>
constexpr auto min_distance(const BallView<FirstFeaturesIterator>&  first_ball_view,
                            const BallView<SecondFeaturesIterator>& second_ball_view) {
    using ValueType = std::common_type_t<typename std::iterator_traits<FirstFeaturesIterator>::value_type,
                                         typename std::iterator_traits<SecondFeaturesIterator>::value_type>;

    const auto centroids_distance = common::math::heuristics::auto_distance(first_ball_view.centroid_begin(),
                                                                            first_ball_view.centroid_end(),
                                                                            second_ball_view.centroid_begin(),
                                                                            second_ball_view.centroid_begin());

    return std::max(static_cast<ValueType>(0),
                    centroids_distance - (first_ball_view.centroid_to_furthest_bound_distance() +
                                          second_ball_view.centroid_to_furthest_bound_distance()));
}

template <typename FirstValue, typename SecondValue, std::size_t NFeatures = 0>
constexpr auto min_distance(const UnboundedBall<FirstValue, NFeatures>&, const UnboundedBall<SecondValue, NFeatures>&) {
    using ValueType = std::common_type_t<FirstValue, SecondValue>;

    return ValueType{};
}

template <typename FirstFeaturesIterator, typename SecondFeaturesIterator>
constexpr auto min_distance(const UnboundedBallView<FirstFeaturesIterator>&,
                            const UnboundedBallView<SecondFeaturesIterator>&) {
    using ValueType = std::common_type_t<typename std::iterator_traits<FirstFeaturesIterator>::value_type,
                                         typename std::iterator_traits<SecondFeaturesIterator>::value_type>;

    return ValueType{};
}

}  // namespace ffcl::datastruct::bounds