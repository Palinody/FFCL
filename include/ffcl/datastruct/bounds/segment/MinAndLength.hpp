#pragma once

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

namespace ffcl::datastruct::bounds::segment {

template <typename Value>
class MinAndLength : public StaticSegment<MinAndLength<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    MinAndLength(const ValueType& position, const ValueType& length)
      : MinAndLength(std::make_pair(position, length)) {}

    MinAndLength(const SegmentType& segment)
      : segment_representation_{segment} {}

    MinAndLength(SegmentType&& segment) noexcept
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

    constexpr auto length_from_centroid_impl() const {
        return common::compute_size_from_middle_with_left_rounding(
            segment_representation_.first, segment_representation_.first + segment_representation_.second - 1);
    }

    constexpr auto centroid_impl() const {
        return common::compute_middle_with_left_rounding(
            segment_representation_.first, segment_representation_.first + segment_representation_.second - 1);
    }

  private:
    // segment represented as a reference position and a length relative to that position
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment