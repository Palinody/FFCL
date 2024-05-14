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
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = FeaturesVector<Segment, 0>;
    using CentroidType = FeaturesVector<ValueType, NFeatures>;
    using IteratorType = typename CentroidType::IteratorType;

    constexpr AABBWithCentroid(const CentroidType& centroid, const SegmentsType& segments)
      : centroid_{centroid}
      , segments_{segments} {}

    constexpr AABBWithCentroid(CentroidType&& centroid, const SegmentsType&& segments) noexcept
      : centroid_{std::move(centroid)}
      , segments_{std::move(segments)} {}

    constexpr std::size_t n_features_impl() const {
        return segments_.size();
    }

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& features_range_first,
                                     const OtherFeaturesIterator& features_range_last) const {
        const std::size_t n_features = std::distance(features_range_first, features_range_last);

        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            // Checks if the feature value is outside of the segment interval for the specified feature index.
            if (!segments_[feature_index].contains_value(features_range_first[feature_index])) {
                return false;
            }
        }
        return true;
    }

    template <typename OtherFeaturesIterator>
    constexpr auto distance_to_centroid_impl(const OtherFeaturesIterator& features_range_first,
                                             const OtherFeaturesIterator& features_range_last) const {
        assert(centroid_.size() == std::distance(features_range_first, features_range_last));

        return common::math::heuristics::auto_distance(
            features_range_first, features_range_last, centroid_.begin(), centroid_.end());
    }

    template <typename OtherFeaturesIterator>
    constexpr auto compute_distance_to_centroid_if_within_bounds_impl(
        const OtherFeaturesIterator& features_range_first,
        const OtherFeaturesIterator& features_range_last) const {
        return is_in_bounds_impl(features_range_first, features_range_last)
                   ? std::optional<ValueType>(distance_to_centroid_impl(features_range_first, features_range_last))
                   : std::nullopt;
    }

    constexpr auto centroid_to_bound_length_impl() const {
        throw std::runtime_error("No half length to return if no feature dimension is specified for this bound.");
        return ValueType{};
    }

    constexpr auto centroid_to_bound_length_impl(std::size_t feature_index) const {
        return segments_[feature_index].centroid_to_bound_length();
    }

    constexpr auto centroid_begin_impl() const {
        return centroid_.begin();
    }

    constexpr auto centroid_end_impl() const {
        return centroid_.end();
    }

  private:
    CentroidType centroid_;
    // A bounding box represented as an array of 1D segments w.r.t. each feature index.
    SegmentsType segments_;
};

}  // namespace ffcl::datastruct::bounds