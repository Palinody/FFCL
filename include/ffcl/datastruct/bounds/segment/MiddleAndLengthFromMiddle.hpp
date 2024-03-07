#pragma once

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

namespace ffcl::datastruct::bounds::segment {

template <typename Value>
class MiddleAndLengthFromMiddle : public StaticSegment<MiddleAndLengthFromMiddle<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    MiddleAndLengthFromMiddle(const ValueType& middle, const ValueType& half_length)
      : MiddleAndLengthFromMiddle(std::make_pair(middle, half_length)) {}

    MiddleAndLengthFromMiddle(const SegmentType& segment)
      : segment_representation_{segment} {}

    MiddleAndLengthFromMiddle(SegmentType&& segment) noexcept
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
        return segment_representation_.second;
    }

    constexpr auto centroid_impl() const {
        return segment_representation_.first;
    }

  private:
    // segment represented as a length and a middle point that cuts it in half
    // this class actually stores half of the length
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment