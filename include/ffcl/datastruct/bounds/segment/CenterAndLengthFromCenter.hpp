#pragma once

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

namespace ffcl::datastruct::bounds::segment {

template <typename Value>
class CenterAndLengthFromCenter : public StaticSegment<CenterAndLengthFromCenter<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    CenterAndLengthFromCenter(const ValueType& middle, const ValueType& length_from_center)
      : CenterAndLengthFromCenter(std::make_pair(middle, length_from_center)) {}

    CenterAndLengthFromCenter(const SegmentType& segment)
      : segment_representation_{segment} {}

    CenterAndLengthFromCenter(SegmentType&& segment) noexcept
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
        return segment_representation_.first;
    }

    constexpr auto length_from_centroid_impl() const {
        return segment_representation_.second;
    }

  private:
    // segment represented as a length and a middle point that cuts it in half
    // this class actually stores half of the length
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment