#pragma once

#include "ffcl/datastruct/bounds/StaticBound.hpp"
#include "ffcl/datastruct/vector/FeaturesVector.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace ffcl::datastruct::bounds {

template <typename Segment, std::size_t NFeatures = 0>
class AABB : public StaticBound<AABB<Segment, NFeatures>> {
  public:
    using SegmentType  = Segment;
    using ValueType    = typename Segment::ValueType;
    using SegmentsType = FeaturesVector<Segment, NFeatures>;
    using IteratorType = typename SegmentsType::Iterator;

    constexpr AABB(const SegmentsType& segments);
    constexpr AABB(SegmentsType&& segments) noexcept;

    constexpr std::size_t n_features_impl() const;

    template <typename OtherFeaturesIterator>
    constexpr bool is_in_bounds_impl(const OtherFeaturesIterator& features_range_first,
                                     const OtherFeaturesIterator& features_range_last) const;

    auto        centroid_at(std::size_t feature_index);
    auto&       segment_at(std::size_t feature_index);
    const auto& segment_at(std::size_t feature_index) const;

  private:
    SegmentsType segments_;
};

template <typename Segment, std::size_t NFeatures>
constexpr AABB<Segment, NFeatures>::AABB(const SegmentsType& segments)
  : segments_{segments} {}

template <typename Segment, std::size_t NFeatures>
constexpr AABB<Segment, NFeatures>::AABB(SegmentsType&& segments) noexcept
  : segments_{std::move(segments)} {}

template <typename Segment, std::size_t NFeatures>
constexpr std::size_t AABB<Segment, NFeatures>::n_features_impl() const {
    return segments_.size();
}

template <typename Segment, std::size_t NFeatures>
template <typename OtherFeaturesIterator>
constexpr bool AABB<Segment, NFeatures>::is_in_bounds_impl(const OtherFeaturesIterator& features_range_first,
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
auto AABB<Segment, NFeatures>::centroid_at(std::size_t feature_index) {
    return segments_[feature_index].centroid();
}

template <typename Segment, std::size_t NFeatures>
auto& AABB<Segment, NFeatures>::segment_at(std::size_t feature_index) {
    return segments_[feature_index];
}

template <typename Segment, std::size_t NFeatures>
const auto& AABB<Segment, NFeatures>::segment_at(std::size_t feature_index) const {
    return segments_[feature_index];
}

}  // namespace ffcl::datastruct::bounds