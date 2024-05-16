#pragma once

#include "ffcl/common/Utils.hpp"

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

namespace ffcl::datastruct::bounds::segment {

template <typename Value>
class LowerBoundAndUpperBound : public StaticSegment<LowerBoundAndUpperBound<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    LowerBoundAndUpperBound(const ValueType& lower_bound, const ValueType& upper_bound)
      : LowerBoundAndUpperBound(std::make_pair(lower_bound, upper_bound)) {}

    LowerBoundAndUpperBound(const SegmentType& segment)
      : segment_representation_{segment} {}

    LowerBoundAndUpperBound(SegmentType&& segment) noexcept
      : segment_representation_{std::move(segment)} {}

    constexpr auto lower_bound_impl() const {
        return segment_representation_.first;
    }

    void update_lower_bound_impl(const ValueType& new_lower_bound) {
        segment_representation_.first = new_lower_bound;
    }

    constexpr auto upper_bound_impl() const {
        return segment_representation_.second;
    }

    void update_upper_bound_impl(const ValueType& new_upper_bound) {
        segment_representation_.second = new_upper_bound;
    }

    constexpr auto centroid_impl() const {
        return common::compute_center_with_left_rounding(segment_representation_.first, segment_representation_.second);
    }

    constexpr auto centroid_to_bound_length_impl() const {
        return common::compute_size_from_center_with_left_rounding(segment_representation_.first,
                                                                   segment_representation_.second);
    }

    constexpr bool contains_value_impl(const ValueType& value) const {
        return !(value < segment_representation_.first || segment_representation_.second < value);
    }

  private:
    // segment represented as a minimum and a maximum value
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment