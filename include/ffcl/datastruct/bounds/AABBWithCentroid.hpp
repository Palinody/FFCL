#pragma once

#include "ffcl/datastruct/bounds/StaticBoundWithCentroid.hpp"
#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace ffcl::datastruct::bounds {

template <typename Segment, std::size_t NFeatures = 0>
class AABBWithCentroid : public StaticBoundWithCentroid<AABBWithCentroid<Segment>> {
  public:
    using SegmentType  = Segment;
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = FeaturesVector<Segment, NFeatures>;
    using CentroidType = FeaturesVector<ValueType, NFeatures>;
    using IteratorType = typename CentroidType::Iterator;

    constexpr AABBWithCentroid(const SegmentsType& segments);
    constexpr AABBWithCentroid(const SegmentsType&& segments);
    constexpr AABBWithCentroid(const CentroidType& centroid, const SegmentsType& segments);
    constexpr AABBWithCentroid(CentroidType&& centroid, const SegmentsType&& segments) noexcept;

    constexpr std::size_t n_features_impl() const;

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& features_range_first,
                                     const OtherFeaturesIterator& features_range_last) const;

    template <typename OtherFeaturesIterator>
    constexpr auto distance_to_centroid_impl(const OtherFeaturesIterator& features_range_first,
                                             const OtherFeaturesIterator& features_range_last) const;

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_to_centroid_if_within_bounds_impl(
        const OtherFeaturesIterator& features_range_first,
        const OtherFeaturesIterator& features_range_last) const;

    template <typename OtherSegment>
    constexpr auto min_distance(const AABBWithCentroid<OtherSegment, NFeatures>& other_aabb_with_centroids) const;

    // ---

    constexpr auto centroid_to_furthest_bound_distance_impl() const;
    constexpr auto centroid_to_bound_distance_impl(std::size_t feature_index) const;

    const auto&    centroid_impl() const;
    constexpr auto centroid_begin_impl() const;
    constexpr auto centroid_end_impl() const;

    constexpr auto segments_begin() const;
    constexpr auto segments_end() const;

    auto&       centroid_at(std::size_t feature_index);
    const auto& centroid_at(std::size_t feature_index) const;

    auto&       segment_at(std::size_t feature_index);
    const auto& segment_at(std::size_t feature_index) const;

  private:
    CentroidType centroid_;
    SegmentsType segments_;
};

template <typename Segment, std::size_t NFeatures>
constexpr AABBWithCentroid<Segment, NFeatures>::AABBWithCentroid(const SegmentsType& segments)
  : centroid_{std::vector<ValueType>(segments.size())}
  , segments_{segments} {
    for (std::size_t feature_index = 0; feature_index < segments_.size(); ++feature_index) {
        centroid_[feature_index] = segments_[feature_index].centroid();
    }
}

template <typename Segment, std::size_t NFeatures>
constexpr AABBWithCentroid<Segment, NFeatures>::AABBWithCentroid(const SegmentsType&& segments)
  : centroid_{std::vector<ValueType>(segments.size())}
  , segments_{std::move(segments)} {
    for (std::size_t feature_index = 0; feature_index < segments_.size(); ++feature_index) {
        centroid_[feature_index] = segments_[feature_index].centroid();
    }
}

template <typename Segment, std::size_t NFeatures>
constexpr AABBWithCentroid<Segment, NFeatures>::AABBWithCentroid(const CentroidType& centroid,
                                                                 const SegmentsType& segments)
  : centroid_{centroid}
  , segments_{segments} {}

template <typename Segment, std::size_t NFeatures>
constexpr AABBWithCentroid<Segment, NFeatures>::AABBWithCentroid(CentroidType&&       centroid,
                                                                 const SegmentsType&& segments) noexcept
  : centroid_{std::move(centroid)}
  , segments_{std::move(segments)} {}

template <typename Segment, std::size_t NFeatures>
constexpr std::size_t AABBWithCentroid<Segment, NFeatures>::n_features_impl() const {
    return segments_.size();
}

template <typename Segment, std::size_t NFeatures>
template <typename OtherFeaturesIterator>
constexpr bool AABBWithCentroid<Segment, NFeatures>::is_in_bounds_impl(
    const OtherFeaturesIterator& features_range_first,
    const OtherFeaturesIterator& features_range_last) const {
    const std::size_t n_features = std::distance(features_range_first, features_range_last);

    for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
        if (!segments_[feature_index].contains_value(features_range_first[feature_index])) {
            return false;
        }
    }
    return true;
}

template <typename Segment, std::size_t NFeatures>
template <typename OtherFeaturesIterator>
constexpr auto AABBWithCentroid<Segment, NFeatures>::distance_to_centroid_impl(
    const OtherFeaturesIterator& features_range_first,
    const OtherFeaturesIterator& features_range_last) const {
    assert(centroid_.size() == std::distance(features_range_first, features_range_last));

    return common::math::heuristics::auto_distance(
        features_range_first, features_range_last, centroid_.begin(), centroid_.end());
}

template <typename Segment, std::size_t NFeatures>
template <typename OtherFeaturesIterator>
constexpr auto AABBWithCentroid<Segment, NFeatures>::compute_distance_to_centroid_if_within_bounds_impl(
    const OtherFeaturesIterator& features_range_first,
    const OtherFeaturesIterator& features_range_last) const {
    return is_in_bounds_impl(features_range_first, features_range_last)
               ? std::optional<ValueType>(distance_to_centroid_impl(features_range_first, features_range_last))
               : std::nullopt;
}

template <typename Segment, std::size_t NFeatures>
template <typename OtherSegment>
constexpr auto AABBWithCentroid<Segment, NFeatures>::min_distance(
    const AABBWithCentroid<OtherSegment, NFeatures>& other_aabb_with_centroids) const {
    ValueType inner_lengths_sum = 0;

    // Compute inner lengths using the min_distance method of each segment.
    for (std::size_t feature_index = 0; feature_index < other_aabb_with_centroids.n_features(); ++feature_index) {
        const auto& this_segment                       = this->segment_at(feature_index);
        const auto& other_segment                      = other_aabb_with_centroids.segment_at(feature_index);
        const auto  this_to_other_segment_min_distance = this_segment.min_distance(other_segment);

        inner_lengths_sum += this_to_other_segment_min_distance * this_to_other_segment_min_distance;
    }
    return std::sqrt(inner_lengths_sum);
}

template <typename Segment, std::size_t NFeatures>
constexpr auto AABBWithCentroid<Segment, NFeatures>::centroid_to_furthest_bound_distance_impl() const {
    auto furthest_distance = 0;

    for (const auto& segment : segments_) {
        const auto candidate_furthest_distance = segment.centroid_to_bound_distance();

        if (candidate_furthest_distance > furthest_distance) {
            furthest_distance = candidate_furthest_distance;
        }
    }
    return furthest_distance;
}

template <typename Segment, std::size_t NFeatures>
constexpr auto AABBWithCentroid<Segment, NFeatures>::centroid_to_bound_distance_impl(std::size_t feature_index) const {
    return segments_[feature_index].centroid_to_bound_distance();
}

template <typename Segment, std::size_t NFeatures>
const auto& AABBWithCentroid<Segment, NFeatures>::centroid_impl() const {
    return centroid_;
}

template <typename Segment, std::size_t NFeatures>
constexpr auto AABBWithCentroid<Segment, NFeatures>::centroid_begin_impl() const {
    return centroid_.begin();
}

template <typename Segment, std::size_t NFeatures>
constexpr auto AABBWithCentroid<Segment, NFeatures>::centroid_end_impl() const {
    return centroid_.end();
}

template <typename Segment, std::size_t NFeatures>
constexpr auto AABBWithCentroid<Segment, NFeatures>::segments_begin() const {
    return segments_.begin();
}

template <typename Segment, std::size_t NFeatures>
constexpr auto AABBWithCentroid<Segment, NFeatures>::segments_end() const {
    return segments_.end();
}

template <typename Segment, std::size_t NFeatures>
auto& AABBWithCentroid<Segment, NFeatures>::centroid_at(std::size_t feature_index) {
    return centroid_[feature_index];
}

template <typename Segment, std::size_t NFeatures>
const auto& AABBWithCentroid<Segment, NFeatures>::centroid_at(std::size_t feature_index) const {
    return centroid_[feature_index];
}

template <typename Segment, std::size_t NFeatures>
auto& AABBWithCentroid<Segment, NFeatures>::segment_at(std::size_t feature_index) {
    return segments_[feature_index];
}

template <typename Segment, std::size_t NFeatures>
const auto& AABBWithCentroid<Segment, NFeatures>::segment_at(std::size_t feature_index) const {
    return segments_[feature_index];
}

}  // namespace ffcl::datastruct::bounds
