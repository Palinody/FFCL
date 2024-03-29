#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

namespace ffcl::datastruct::bounds::segment {

template <typename Value>
class MinAndMax : public StaticSegment<MinAndMax<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    MinAndMax(const ValueType& min, const ValueType& max)
      : MinAndMax(std::make_pair(min, max)) {}

    MinAndMax(const SegmentType& segment)
      : segment_representation_{segment} {}

    MinAndMax(SegmentType&& segment) noexcept
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
        return common::compute_size_from_middle_with_left_rounding(segment_representation_.first,
                                                                   segment_representation_.second);
    }

    constexpr auto centroid_impl() const {
        return common::compute_middle_with_left_rounding(segment_representation_.first, segment_representation_.second);
    }

  private:
    // segment represented as a minimum and a maximum value
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment