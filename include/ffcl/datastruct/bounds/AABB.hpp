#pragma once

#include "ffcl/datastruct/bounds/StaticBound.hpp"
#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace ffcl::datastruct::bounds {

template <typename Segment, std::size_t NFeatures = 0>
class AABB : public StaticBound<AABB<Segment>> {
  public:
    using SegmentType  = Segment;
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = FeaturesVector<Segment, NFeatures>;
    using IteratorType = typename SegmentsType::Iterator;

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

    auto centroid_at(std::size_t feature_index) {
        return segments_[feature_index].centroid();
    }

    auto& segment_at(std::size_t feature_index) {
        return segments_[feature_index];
    }

    const auto& segment_at(std::size_t feature_index) const {
        return segments_[feature_index];
    }

  private:
    // A bounding box represented as an array of 1D segments w.r.t. each feature index.
    SegmentsType segments_;
};

}  // namespace ffcl::datastruct::bounds