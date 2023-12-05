#pragma once

#include "ffcl/datastruct/bounds/segment_representation/StaticSegmentRepresentation.hpp"

namespace ffcl::datastruct::bounds::segment_representation {

template <typename Value>
class PositionAndLength : public StaticSegmentRepresentation<PositionAndLength<Value>> {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<ValueType, ValueType>;

    PositionAndLength(const ValueType& position, const ValueType& length)
      : PositionAndLength(std::make_pair(position, length)) {}

    PositionAndLength(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    PositionAndLength(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

    constexpr auto length_from_centroid_impl() const {
        return segment_representation_.second / 2;
    }

    constexpr auto centroid_impl() const {
        return segment_representation_.first + length_from_centroid_impl();
    }

  private:
    // segment_representation represented as a reference position and a length relative to that position
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation