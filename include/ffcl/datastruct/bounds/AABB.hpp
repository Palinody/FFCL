#pragma once

#include "ffcl/datastruct/bounds/StaticBound.hpp"
#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace ffcl::datastruct::bounds {

template <typename Segment>
class AABB : public StaticBound<AABB<Segment>> {
  public:
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = FeaturesVector<Segment, 0>;

    constexpr AABB(const SegmentsType& segments)
      : segments_{segments} {}

    constexpr AABB(const SegmentsType&& segments) noexcept
      : segments_{std::move(segments)} {}

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
        const std::size_t n_features = std::distance(features_range_first, features_range_last);

        auto centroid = std::make_unique<ValueType[]>(n_features);

        for (std::size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            centroid[feature_index] = segments_[feature_index].centroid();
        }
        return common::math::heuristics::auto_distance(
            features_range_first, features_range_last, centroid.get(), centroid.get() + n_features);
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

  private:
    // A bounding box represented as an array of 1D segments w.r.t. each feature index.
    SegmentsType segments_;
};

}  // namespace ffcl::datastruct::bounds