#pragma once

#include "ffcl/common/math/heuristics/Distances.hpp"

#include "ffcl/datastruct/bounds/AABB.hpp"
#include "ffcl/datastruct/bounds/AABBWithCentroid.hpp"
#include "ffcl/datastruct/bounds/Ball.hpp"
#include "ffcl/datastruct/bounds/UnboundedBall.hpp"
#include "ffcl/datastruct/bounds/segment/LowerBoundAndUpperBound.hpp"

#include "ffcl/datastruct/bounds/segment/distances/MinDistance.hpp"

#include <cmath>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace ffcl::datastruct::bounds {

// Squared variants. The dual-tree Boruvka pass only ever compares distances, so it works in squared space
// and takes the square root once per final edge instead of once per candidate.
template <typename FirstSegment, typename SecondSegment, std::size_t NFeatures = 0>
constexpr auto squared_min_distance(const AABBWithCentroid<FirstSegment, NFeatures>&  first_aabb_with_centroids,
                                    const AABBWithCentroid<SecondSegment, NFeatures>& second_aabb_with_centroids) {
    using ValueType = std::common_type_t<typename FirstSegment::ValueType, typename SecondSegment::ValueType>;

    ValueType inner_lengths_sum = 0;

    for (std::size_t feature_index = 0; feature_index < second_aabb_with_centroids.n_features(); ++feature_index) {
        const auto& first_segment  = first_aabb_with_centroids.segment_at(feature_index);
        const auto& second_segment = second_aabb_with_centroids.segment_at(feature_index);

        const auto segments_min_distance = min_distance(first_segment, second_segment);

        inner_lengths_sum += segments_min_distance * segments_min_distance;
    }
    return inner_lengths_sum;
}

template <typename FeaturesIterator, typename Segment, std::size_t NFeatures = 0>
constexpr auto squared_min_distance(const FeaturesIterator&                    features_range_first,
                                    const FeaturesIterator&                    features_range_last,
                                    const AABBWithCentroid<Segment, NFeatures>& aabb_with_centroid) {
    using ValueType = std::common_type_t<typename std::iterator_traits<FeaturesIterator>::value_type,
                                         typename Segment::ValueType>;

    ValueType   squared_distance = 0;
    std::size_t feature_index    = 0;

    for (auto features_it = features_range_first; features_it != features_range_last; ++features_it, ++feature_index) {
        const auto&     segment       = aabb_with_centroid.segment_at(feature_index);
        const ValueType feature_value = static_cast<ValueType>(*features_it);
        const ValueType lower_bound   = static_cast<ValueType>(segment.lower_bound());
        const ValueType upper_bound   = static_cast<ValueType>(segment.upper_bound());

        if (feature_value < lower_bound) {
            const auto delta = lower_bound - feature_value;
            squared_distance += delta * delta;
        } else if (feature_value > upper_bound) {
            const auto delta = feature_value - upper_bound;
            squared_distance += delta * delta;
        }
    }
    return squared_distance;
}

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

// Minimum distance between a single query point and an axis-aligned bounding box.
// Every feature is clamped to the segment of the box: a feature inside the segment
// contributes nothing, a feature outside contributes the distance to the closest
// bound. This is the `referenceNode.MinDistance(queryPoint)` primitive used by the
// dual-tree Boruvka base case to skip a whole reference node for a query point.
template <typename FeaturesIterator, typename Segment, std::size_t NFeatures = 0>
constexpr auto min_distance(const FeaturesIterator&         features_range_first,
                            const FeaturesIterator&         features_range_last,
                            const AABB<Segment, NFeatures>& aabb) {
    using ValueType =
        std::common_type_t<typename std::iterator_traits<FeaturesIterator>::value_type, typename Segment::ValueType>;

    ValueType   squared_distance = 0;
    std::size_t feature_index    = 0;

    for (auto features_it = features_range_first; features_it != features_range_last; ++features_it, ++feature_index) {
        const auto&     segment       = aabb.segment_at(feature_index);
        const ValueType feature_value = static_cast<ValueType>(*features_it);
        const ValueType lower_bound   = static_cast<ValueType>(segment.lower_bound());
        const ValueType upper_bound   = static_cast<ValueType>(segment.upper_bound());

        if (feature_value < lower_bound) {
            const auto delta = lower_bound - feature_value;
            squared_distance += delta * delta;

        } else if (feature_value > upper_bound) {
            const auto delta = feature_value - upper_bound;
            squared_distance += delta * delta;
        }
    }
    return std::sqrt(squared_distance);
}

template <typename FeaturesIterator, typename Segment, std::size_t NFeatures = 0>
constexpr auto min_distance(const FeaturesIterator&                     features_range_first,
                            const FeaturesIterator&                     features_range_last,
                            const AABBWithCentroid<Segment, NFeatures>& aabb_with_centroid) {
    using ValueType =
        std::common_type_t<typename std::iterator_traits<FeaturesIterator>::value_type, typename Segment::ValueType>;

    ValueType   squared_distance = 0;
    std::size_t feature_index    = 0;

    for (auto features_it = features_range_first; features_it != features_range_last; ++features_it, ++feature_index) {
        const auto&     segment       = aabb_with_centroid.segment_at(feature_index);
        const ValueType feature_value = static_cast<ValueType>(*features_it);
        const ValueType lower_bound   = static_cast<ValueType>(segment.lower_bound());
        const ValueType upper_bound   = static_cast<ValueType>(segment.upper_bound());

        if (feature_value < lower_bound) {
            const auto delta = lower_bound - feature_value;
            squared_distance += delta * delta;

        } else if (feature_value > upper_bound) {
            const auto delta = feature_value - upper_bound;
            squared_distance += delta * delta;
        }
    }
    return std::sqrt(squared_distance);
}

template <typename FirstFeaturesIterator, typename SecondFeaturesIterator>
constexpr auto min_distance(const UnboundedBallView<FirstFeaturesIterator>&,
                            const UnboundedBallView<SecondFeaturesIterator>&) {
    using ValueType = std::common_type_t<typename std::iterator_traits<FirstFeaturesIterator>::value_type,
                                         typename std::iterator_traits<SecondFeaturesIterator>::value_type>;

    return ValueType{};
}

}  // namespace ffcl::datastruct::bounds