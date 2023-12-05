#pragma once

#include "ffcl/datastruct/bounds/segment_representation/StaticSegmentRepresentation.hpp"

namespace ffcl::datastruct::bounds::segment_representation {

template <typename Value>
class MiddleAndHalfLength : public StaticSegmentRepresentation<MiddleAndHalfLength<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    MiddleAndHalfLength(const ValueType& middle, const ValueType& half_length)
      : MiddleAndHalfLength(std::make_pair(middle, half_length)) {}

    MiddleAndHalfLength(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    MiddleAndHalfLength(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

    constexpr auto length_from_centroid_impl() const {
        return segment_representation_.second;
    }

    constexpr auto centroid_impl() const {
        return segment_representation_.first;
    }

  private:
    // segment_representation represented as a length and a middle point that cuts it in half
    // this class actually stores half of the length
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation