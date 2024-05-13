#pragma once

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

namespace ffcl::datastruct::bounds::segment {

template <typename Value>
class LowerBoundAndLength : public StaticSegment<LowerBoundAndLength<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    LowerBoundAndLength(const ValueType& lower_bound, const ValueType& length)
      : LowerBoundAndLength(std::make_pair(lower_bound, length)) {}

    LowerBoundAndLength(const SegmentType& segment)
      : segment_representation_{segment} {}

    LowerBoundAndLength(SegmentType&& segment) noexcept
      : segment_representation_{std::move(segment)} {}

    constexpr auto read_only_first_impl() const {
        return segment_representation_.first;
    }

    auto& read_write_first_impl() {
        return segment_representation_.first;
    }

    constexpr auto read_only_second_impl() const {
        return segment_representation_.second;
    }

    auto& read_write_second_impl() {
        return segment_representation_.second;
    }

    constexpr auto centroid_impl() const {
        if constexpr (std::is_integral_v<ValueType>) {
            return common::compute_center_with_left_rounding(
                segment_representation_.first, segment_representation_.first + segment_representation_.second - 1);
        } else {
            return common::compute_center_with_left_rounding(
                segment_representation_.first, segment_representation_.first + segment_representation_.second);
        }
    }

    constexpr auto length_from_centroid_impl() const {
        if constexpr (std::is_integral_v<ValueType>) {
            return common::compute_size_from_center_with_left_rounding(
                segment_representation_.first, segment_representation_.first + segment_representation_.second - 1);
        } else {
            return common::compute_size_from_center_with_left_rounding(
                segment_representation_.first, segment_representation_.first + segment_representation_.second);
        }
    }

  private:
    // Segment represented as a reference lower bound and a length relative to that lower bound.
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment