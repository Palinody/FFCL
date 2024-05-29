#pragma once

#include "ffcl/datastruct/bounds/segment/StaticSegment.hpp"

namespace ffcl::datastruct::bounds::segment {

template <typename Value>
class LowerBoundAndLength : public StaticSegment<LowerBoundAndLength<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    LowerBoundAndLength(const ValueType& lower_bound, const ValueType& length);
    LowerBoundAndLength(const SegmentType& segment);
    LowerBoundAndLength(SegmentType&& segment) noexcept;

    constexpr auto lower_bound_impl() const;
    void           update_lower_bound_impl(const ValueType& new_lower_bound);

    constexpr auto upper_bound_impl() const;
    void           update_upper_bound_impl(const ValueType& new_upper_bound);

    constexpr auto centroid_impl() const;
    constexpr auto centroid_to_bound_distance_impl() const;
    constexpr bool contains_value_impl(const ValueType& value) const;

  private:
    SegmentType segment_representation_;
};

template <typename Value>
LowerBoundAndLength<Value>::LowerBoundAndLength(const ValueType& lower_bound, const ValueType& length)
  : LowerBoundAndLength(std::make_pair(lower_bound, length)) {}

template <typename Value>
LowerBoundAndLength<Value>::LowerBoundAndLength(const SegmentType& segment)
  : segment_representation_{segment} {}

template <typename Value>
LowerBoundAndLength<Value>::LowerBoundAndLength(SegmentType&& segment) noexcept
  : segment_representation_{std::move(segment)} {}

template <typename Value>
constexpr auto LowerBoundAndLength<Value>::lower_bound_impl() const {
    return segment_representation_.first;
}

template <typename Value>
void LowerBoundAndLength<Value>::update_lower_bound_impl(const ValueType& new_lower_bound) {
    segment_representation_.first = new_lower_bound;
}

template <typename Value>
constexpr auto LowerBoundAndLength<Value>::upper_bound_impl() const {
    return segment_representation_.first + segment_representation_.second;
}

template <typename Value>
void LowerBoundAndLength<Value>::update_upper_bound_impl(const ValueType& new_upper_bound) {
    segment_representation_.second = new_upper_bound - segment_representation_.first;
}

template <typename Value>
constexpr auto LowerBoundAndLength<Value>::centroid_impl() const {
    if constexpr (std::is_integral_v<ValueType>) {
        return common::compute_center_with_left_rounding(
            segment_representation_.first, segment_representation_.first + segment_representation_.second - 1);
    } else {
        return common::compute_center_with_left_rounding(
            segment_representation_.first, segment_representation_.first + segment_representation_.second);
    }
}

template <typename Value>
constexpr auto LowerBoundAndLength<Value>::centroid_to_bound_distance_impl() const {
    if constexpr (std::is_integral_v<ValueType>) {
        return common::compute_size_from_center_with_left_rounding(
            segment_representation_.first, segment_representation_.first + segment_representation_.second - 1);
    } else {
        return common::compute_size_from_center_with_left_rounding(
            segment_representation_.first, segment_representation_.first + segment_representation_.second);
    }
}

template <typename Value>
constexpr bool LowerBoundAndLength<Value>::contains_value_impl(const ValueType& value) const {
    if constexpr (std::is_integral_v<ValueType>) {
        return !(value < segment_representation_.first ||
                 segment_representation_.first + segment_representation_.second - 1 < value);
    } else {
        return !(value < segment_representation_.first ||
                 segment_representation_.first + segment_representation_.second < value);
    }
}

}  // namespace ffcl::datastruct::bounds::segment