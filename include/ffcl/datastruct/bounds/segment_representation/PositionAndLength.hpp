#pragma once

namespace ffcl::datastruct::bounds::segment_representation {

template <typename Value>
class PositionAndLength {
  public:
    using ValueType   = Value;
    using SegmentType = std::pair<Value, Value>;

    PositionAndLength(const ValueType& position, const ValueType& length)
      : PositionAndLength(std::make_pair(position, length)) {}

    PositionAndLength(const SegmentType& segment_representation)
      : segment_representation_{segment_representation} {}

    PositionAndLength(SegmentType&& segment_representation) noexcept
      : segment_representation_{std::move(segment_representation)} {}

    constexpr Value length_from_centroid() const {
        return segment_representation_.second / 2;
    }

    constexpr Value centroid() const {
        return segment_representation_.first + length_from_centroid();
    }

  private:
    // segment_representation represented as a reference position and a length relative to that position
    SegmentType segment_representation_;
};

}  // namespace ffcl::datastruct::bounds::segment_representation